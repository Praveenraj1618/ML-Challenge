from __future__ import annotations

import numpy as np
import pandas as pd


def truth_map(ground_truth: pd.DataFrame) -> dict[str, set[str]]:
    return {
        row.source1_entity_id: {x for x in str(row.matched_entity_ids).split(",") if x}
        for row in ground_truth.itertuples(index=False)
    }


def macro_fbeta(
    predictions: dict[str, set[str]],
    truth: dict[str, set[str]],
    beta: float = 0.5,
) -> float:
    beta2 = beta * beta
    scores = []
    for entity_id, actual in truth.items():
        predicted = predictions.get(entity_id, set())
        if not actual:
            scores.append(1.0 if not predicted else 0.0)
            continue
        if not predicted:
            scores.append(0.0)
            continue
        tp = len(actual & predicted)
        precision = tp / len(predicted)
        recall = tp / len(actual)
        denom = beta2 * precision + recall
        scores.append((1 + beta2) * precision * recall / denom if denom else 0.0)
    return float(np.mean(scores))


def predictions_at_threshold(scored: pd.DataFrame, threshold: float) -> dict[str, set[str]]:
    accepted = scored[scored["probability"] >= threshold]
    return accepted.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()


def tune_threshold(
    scored: pd.DataFrame,
    truth: dict[str, set[str]],
    thresholds=None,
) -> tuple[float, float, pd.DataFrame]:
    if thresholds is None:
        thresholds = np.unique(np.r_[np.linspace(0.1, 0.95, 86), np.linspace(0.955, 0.999, 45)])
    records = []
    for threshold in thresholds:
        score = macro_fbeta(predictions_at_threshold(scored, float(threshold)), truth)
        records.append({"threshold": float(threshold), "macro_f0_5": score})
    table = pd.DataFrame(records).sort_values("macro_f0_5", ascending=False)
    best = table.iloc[0]
    return float(best.threshold), float(best.macro_f0_5), table

