from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .normalize import digits, normalize_address, normalize_name, normalize_text


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _containment(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _ratio(left: str, right: str) -> float:
    longer = max(len(left), len(right))
    return min(len(left), len(right)) / longer if longer else 1.0


def build_features(
    candidate_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    left = source1.set_index("entity_id", drop=False)
    right = targets.set_index("entity_id", drop=False)
    rows = []
    for pair in candidate_pairs.itertuples(index=False):
        a = left.loc[pair.source1_entity_id]
        b = right.loc[pair.candidate_entity_id]
        name_a, name_b = normalize_name(a.business_name), normalize_name(b.business_name)
        core_a, core_b = normalize_name(a.business_name, True), normalize_name(b.business_name, True)
        addr_a, addr_b = normalize_address(a.business_address), normalize_address(b.business_address)
        original_a, original_b = normalize_text(a.business_name), normalize_text(b.business_name)
        name_tokens_a, name_tokens_b = set(core_a.split()), set(core_b.split())
        addr_tokens_a, addr_tokens_b = set(addr_a.split()), set(addr_b.split())
        digits_a, digits_b = digits(a.business_address), digits(b.business_address)
        rows.append({
            "source1_entity_id": pair.source1_entity_id,
            "candidate_entity_id": pair.candidate_entity_id,
            "char_name": pair.char_name,
            "char_address": pair.char_address,
            "bm25": pair.bm25,
            "retrieval_methods": pair.retrieval_methods,
            "name_ratio": fuzz.ratio(name_a, name_b) / 100,
            "name_wratio": fuzz.WRatio(name_a, name_b) / 100,
            "name_token_set": fuzz.token_set_ratio(core_a, core_b) / 100,
            "name_token_sort": fuzz.token_sort_ratio(core_a, core_b) / 100,
            "name_jaro": JaroWinkler.normalized_similarity(name_a, name_b),
            "name_edit": Levenshtein.normalized_similarity(name_a, name_b),
            "name_jaccard": _jaccard(name_tokens_a, name_tokens_b),
            "name_containment": _containment(name_tokens_a, name_tokens_b),
            "address_ratio": fuzz.ratio(addr_a, addr_b) / 100,
            "address_wratio": fuzz.WRatio(addr_a, addr_b) / 100,
            "address_token_set": fuzz.token_set_ratio(addr_a, addr_b) / 100,
            "address_jaro": JaroWinkler.normalized_similarity(addr_a, addr_b),
            "address_jaccard": _jaccard(addr_tokens_a, addr_tokens_b),
            "address_containment": _containment(addr_tokens_a, addr_tokens_b),
            "digit_jaccard": _jaccard(digits_a, digits_b),
            "digit_conflict": float(bool(digits_a and digits_b and not digits_a.intersection(digits_b))),
            "country_equal": float(str(a.country) == str(b.country)),
            "name_exact": float(name_a == name_b and bool(name_a)),
            "name_core_exact": float(core_a == core_b and bool(core_a)),
            "address_exact": float(addr_a == addr_b and bool(addr_a)),
            "address_missing": float(not addr_b),
            "name_length_ratio": _ratio(name_a, name_b),
            "address_length_ratio": _ratio(addr_a, addr_b),
            "original_script_name_ratio": fuzz.ratio(original_a, original_b) / 100,
            "same_target_source": 2.0 if str(pair.candidate_entity_id).startswith("S2-") else 3.0,
        })
    return pd.DataFrame(rows)


def add_rank_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for col in ["char_name", "char_address", "bm25"]:
        out[f"{col}_rank"] = out.groupby("source1_entity_id")[col].rank(
            method="min", ascending=False
        )
        best = out.groupby("source1_entity_id")[col].transform("max")
        out[f"{col}_from_best"] = best - out[col]
    out["candidate_count"] = out.groupby("source1_entity_id")[
        "candidate_entity_id"
    ].transform("count")
    return out


def model_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"source1_entity_id", "candidate_entity_id", "label", "fold", "probability"}
    return [c for c in df.columns if c not in excluded and np.issubdtype(df[c].dtype, np.number)]

