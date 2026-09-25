from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .features import add_rank_features, build_features, model_columns
from .metrics import predictions_at_threshold, truth_map, tune_threshold
from .retrieval import RetrievalConfig, generate_candidates


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)


def load_split(directory: Path, prefix: str):
    frames = [read_tsv(directory / f"{prefix}_source{i}.tsv") for i in (1, 2, 3)]
    return frames[0], pd.concat(frames[1:], ignore_index=True)


def labels_for_candidates(candidates: pd.DataFrame, ground_truth: pd.DataFrame) -> np.ndarray:
    truth = truth_map(ground_truth)
    return np.fromiter(
        (
            pair.candidate_entity_id in truth.get(pair.source1_entity_id, set())
            for pair in candidates.itertuples(index=False)
        ),
        dtype=np.int8,
        count=len(candidates),
    )


def candidate_recall(candidates: pd.DataFrame, ground_truth: pd.DataFrame):
    truth = truth_map(ground_truth)
    found = candidates.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()
    total_links = sum(len(x) for x in truth.values())
    captured = sum(len(ids & found.get(entity, set())) for entity, ids in truth.items())
    entities_with_all = np.mean([
        ids.issubset(found.get(entity, set())) for entity, ids in truth.items()
    ])
    return {
        "link_recall": captured / total_links if total_links else 1.0,
        "entities_with_all_matches": float(entities_with_all),
        "captured_links": int(captured),
        "total_links": int(total_links),
    }


def model_params(seed: int):
    return dict(
        objective="binary",
        n_estimators=900,
        learning_rate=0.04,
        num_leaves=47,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=0.8,
        class_weight={0: 1.0, 1: 4.0},
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def write_candidate_output(source1: pd.DataFrame, candidates: pd.DataFrame, path: Path):
    grouped = candidates.groupby("source1_entity_id")["candidate_entity_id"].agg(
        lambda x: ",".join(dict.fromkeys(x))
    ).to_dict()
    output = pd.DataFrame({
        "source1_entity_id": source1["entity_id"],
        "candidate_entity_ids": source1["entity_id"].map(grouped).fillna(""),
    })
    output.to_csv(path, sep="\t", index=False)


def write_matching_output(
    source1: pd.DataFrame,
    scored: pd.DataFrame,
    threshold: float,
    path: Path,
):
    predicted = predictions_at_threshold(scored, threshold)
    output = pd.DataFrame({
        "source1_entity_id": source1["entity_id"],
        "matched_entity_ids": source1["entity_id"].map(
            lambda x: ",".join(sorted(predicted.get(x, set())))
        ),
    })
    output.to_csv(path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--char-name-k", type=int, default=60)
    parser.add_argument("--char-address-k", type=int, default=60)
    parser.add_argument("--bm25-k", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-features", type=int, default=250000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    retrieval_config = RetrievalConfig(
        char_name_k=args.char_name_k,
        char_address_k=args.char_address_k,
        bm25_k=args.bm25_k,
        batch_size=args.batch_size,
        char_max_features=args.max_features,
        word_max_features=args.max_features,
    )

    train_s1, train_targets = load_split(args.train_dir, "train")
    ground_truth = read_tsv(args.train_dir / "train_ground_truth.tsv")
    print("Generating training candidates...")
    train_candidates = generate_candidates(train_s1, train_targets, retrieval_config)
    recall_stats = candidate_recall(train_candidates, ground_truth)
    print("Candidate recall:", json.dumps(recall_stats, indent=2))

    print("Building training features...")
    train_features = add_rank_features(build_features(train_candidates, train_s1, train_targets))
    train_features["label"] = labels_for_candidates(train_features, ground_truth)
    columns = model_columns(train_features)

    n_groups = train_features["source1_entity_id"].nunique()
    folds = min(args.folds, n_groups)
    splitter = GroupKFold(n_splits=folds)
    oof = train_features[["source1_entity_id", "candidate_entity_id"]].copy()
    oof["probability"] = 0.0
    fold_models = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(
        train_features, train_features["label"], groups=train_features["source1_entity_id"]
    )):
        model = lgb.LGBMClassifier(**model_params(2026 + fold))
        model.fit(
            train_features.iloc[train_idx][columns], train_features.iloc[train_idx]["label"],
            eval_set=[(train_features.iloc[valid_idx][columns], train_features.iloc[valid_idx]["label"])],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        oof.loc[valid_idx, "probability"] = model.predict_proba(
            train_features.iloc[valid_idx][columns]
        )[:, 1]
        fold_models.append(model)

    threshold, score, threshold_table = tune_threshold(oof, truth_map(ground_truth))
    print(f"OOF macro F0.5={score:.6f} at threshold={threshold:.4f}")
    threshold_table.to_csv(args.work_dir / "threshold_search.csv", index=False)

    print("Generating test candidates and features...")
    test_s1, test_targets = load_split(args.test_dir, "test")
    test_candidates = generate_candidates(test_s1, test_targets, retrieval_config)
    test_features = add_rank_features(build_features(test_candidates, test_s1, test_targets))
    probabilities = np.mean([
        model.predict_proba(test_features[columns])[:, 1] for model in fold_models
    ], axis=0)
    test_scored = test_features[["source1_entity_id", "candidate_entity_id"]].copy()
    test_scored["probability"] = probabilities

    write_candidate_output(test_s1, test_candidates, args.output_dir / "candidate_pairs.tsv")
    write_matching_output(test_s1, test_scored, threshold, args.output_dir / "matching_results.tsv")

    joblib.dump(fold_models, args.work_dir / "fold_models.joblib")
    (args.work_dir / "metadata.json").write_text(json.dumps({
        "threshold": threshold,
        "oof_macro_f0_5": score,
        "candidate_recall": recall_stats,
        "feature_columns": columns,
        "retrieval_config": retrieval_config.__dict__,
    }, indent=2), encoding="utf-8")
    print("Finished. Outputs written to", args.output_dir)


if __name__ == "__main__":
    main()
