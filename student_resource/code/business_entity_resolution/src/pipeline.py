from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .features import add_rank_features, build_features, model_columns
from .metrics import predictions_at_threshold, truth_map, tune_threshold
from .retrieval import RetrievalConfig, iter_candidates


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)


def load_split(directory: Path, prefix: str):
    frames = [read_tsv(directory / f"{prefix}_source{i}.tsv") for i in (1, 2, 3)]
    targets = pd.concat(frames[1:], ignore_index=True)
    for frame in (frames[0], targets):
        required = {"entity_id", "business_name", "business_address", "country"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing columns: {required - set(frame.columns)}")
        if frame.entity_id.eq("").any() or frame.entity_id.duplicated().any():
            raise ValueError("Entity IDs must be nonempty and unique within source1/combined targets")
    return frames[0], targets


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


def run_inference(args, retrieval_config, fold_models, columns, threshold):
    print("Generating test candidates and features...")
    test_s1, test_targets = load_split(args.test_dir, "test")
    target_lookup = test_targets.set_index("entity_id", drop=False)
    candidate_path = args.output_dir / "candidate_pairs.tsv.partial"
    matching_path = args.output_dir / "matching_results.tsv.partial"
    with candidate_path.open("w", encoding="utf-8", newline="") as candidate_file, matching_path.open("w", encoding="utf-8", newline="") as matching_file:
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
        matching_file.write("source1_entity_id\tmatched_entity_ids\n")
        for batch, pairs in iter_candidates(test_s1, test_targets, retrieval_config, args.work_dir):
            predicted = {}
            if len(pairs):
                selected = target_lookup.loc[pairs.candidate_entity_id.unique()]
                print(f"Scoring {len(pairs):,} candidate pairs", flush=True)
                features = add_rank_features(build_features(pairs, batch, selected))
                probabilities = np.zeros(len(features), dtype=np.float32)
                for model in fold_models:
                    probabilities += model.predict_proba(features[columns])[:, 1] / len(fold_models)
                scored = features[["source1_entity_id", "candidate_entity_id"]].copy()
                scored["probability"] = probabilities
                predicted = predictions_at_threshold(scored, threshold)
                del features, scored, probabilities
            grouped = pairs.groupby("source1_entity_id").candidate_entity_id.agg(list).to_dict()
            pd.DataFrame({"source1_entity_id": batch.entity_id,
                "candidate_entity_ids": [",".join(grouped.get(x, [])) for x in batch.entity_id]}).to_csv(candidate_file, sep="\t", index=False, header=False)
            pd.DataFrame({"source1_entity_id": batch.entity_id,
                "matched_entity_ids": [",".join(sorted(predicted.get(x, set()))) for x in batch.entity_id]}).to_csv(matching_file, sep="\t", index=False, header=False)
            candidate_file.flush()
            matching_file.flush()
    candidate_path.replace(args.output_dir / "candidate_pairs.tsv")
    matching_path.replace(args.output_dir / "matching_results.tsv")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--train-queries", type=int, default=2000)
    parser.add_argument("--fit-sample", type=int, default=5000)
    parser.add_argument("--document-batch", type=int, default=10000)
    parser.add_argument("--query-batch", type=int, default=256)
    parser.add_argument("--char-name-k", type=int, default=60)
    parser.add_argument("--char-address-k", type=int, default=60)
    parser.add_argument("--bm25-k", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-features", type=int, default=250000)
    parser.add_argument("--predict-only", action="store_true", help="Reuse the local trained_checkpoint.joblib; restart test inference")
    args = parser.parse_args()
    if args.train_queries < 2 or args.folds < 2:
        parser.error("Use at least two training queries and folds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    retrieval_config = RetrievalConfig(
        char_name_k=args.char_name_k,
        char_address_k=args.char_address_k,
        bm25_k=args.bm25_k,
        batch_size=args.batch_size,
        char_max_features=args.max_features,
        word_max_features=args.max_features,
        fit_sample=args.fit_sample,
        document_batch=args.document_batch,
        query_batch=args.query_batch,
    )

    if args.predict_only:
        saved = joblib.load(args.work_dir / "trained_checkpoint.joblib")
        config = RetrievalConfig(**saved["retrieval_config"])
        config.query_batch = args.query_batch
        config.document_batch = args.document_batch
        config.batch_size = args.batch_size
        run_inference(args, config, saved["models"], saved["columns"], saved["threshold"])
        return

    train_s1, train_targets = load_split(args.train_dir, "train")
    ground_truth = read_tsv(args.train_dir / "train_ground_truth.tsv")
    train_s1 = train_s1[train_s1.entity_id.isin(ground_truth.source1_entity_id)]
    train_s1 = train_s1.sample(n=min(args.train_queries, len(train_s1)), random_state=2026)
    ground_truth = ground_truth[ground_truth.source1_entity_id.isin(train_s1.entity_id)]
    print(f"First-submission baseline: training on {len(train_s1):,} labeled queries; ALL targets retained", flush=True)
    print("Generating training candidates...")
    target_lookup = train_targets.set_index("entity_id", drop=False)
    feature_parts, candidate_parts = [], []
    for batch, pairs in iter_candidates(train_s1, train_targets, retrieval_config, args.work_dir):
        candidate_parts.append(pairs)
        if len(pairs):
            selected = target_lookup.loc[pairs.candidate_entity_id.unique()]
            print(f"Building features for {len(pairs):,} pairs", flush=True)
            feature_parts.append(add_rank_features(build_features(pairs, batch, selected)))
    if not feature_parts:
        raise ValueError("No training candidates; check country fields and input text")
    train_candidates = pd.concat(candidate_parts, ignore_index=True)
    recall_stats = candidate_recall(train_candidates, ground_truth)
    print("Candidate recall:", json.dumps(recall_stats, indent=2))

    print("Building training features...")
    train_features = pd.concat(feature_parts, ignore_index=True)
    train_features["label"] = labels_for_candidates(train_features, ground_truth)
    columns = model_columns(train_features)
    if train_features.label.nunique() < 2:
        raise ValueError("Training candidates must contain both positive and negative labels")
    del feature_parts, candidate_parts, train_candidates, train_s1, train_targets, target_lookup, selected, batch, pairs
    gc.collect()

    n_groups = train_features["source1_entity_id"].nunique()
    folds = min(args.folds, n_groups)
    if folds < 2:
        raise ValueError("Not enough distinct query entities for validation")
    splitter = GroupKFold(n_splits=folds)
    oof = train_features[["source1_entity_id", "candidate_entity_id"]].copy()
    oof["probability"] = 0.0
    fold_models = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(
        train_features, train_features["label"], groups=train_features["source1_entity_id"]
    )):
        print(f"Training fold {fold + 1}/{folds}", flush=True)
        if train_features.iloc[train_idx].label.nunique() < 2:
            raise ValueError("A training fold has only one class; increase --train-queries")
        model = lgb.LGBMClassifier(**model_params(2026 + fold))
        model.fit(
            train_features.iloc[train_idx][columns], train_features.iloc[train_idx]["label"],
            eval_set=[(train_features.iloc[valid_idx][columns], train_features.iloc[valid_idx]["label"])],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(100)],
        )
        oof.loc[valid_idx, "probability"] = model.predict_proba(
            train_features.iloc[valid_idx][columns]
        )[:, 1]
        fold_models.append(model)

    threshold, score, threshold_table = tune_threshold(oof, truth_map(ground_truth))
    print(f"OOF macro F0.5={score:.6f} at threshold={threshold:.4f}")
    threshold_table.to_csv(args.work_dir / "threshold_search.csv", index=False)
    joblib.dump({"models": fold_models, "columns": columns, "threshold": threshold,
                 "retrieval_config": retrieval_config.__dict__}, args.work_dir / "trained_checkpoint.joblib")
    del train_features, oof
    gc.collect()

    run_inference(args, retrieval_config, fold_models, columns, threshold)

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
