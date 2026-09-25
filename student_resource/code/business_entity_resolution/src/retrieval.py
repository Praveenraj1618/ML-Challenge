from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sparse_dot_topn import sp_matmul_topn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .normalize import normalize_address, normalize_name, normalize_text


@dataclass
class RetrievalConfig:
    char_name_k: int = 60
    char_address_k: int = 60
    bm25_k: int = 75
    batch_size: int = 256
    char_max_features: int = 250_000
    word_max_features: int = 250_000


def prepare_records(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["country"] = out["country"].fillna("").astype(str)
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["name_core"] = out["business_name"].map(lambda x: normalize_name(x, True))
    out["address_norm"] = out["business_address"].map(normalize_address)
    out["original_norm"] = (
        out["business_name"].map(normalize_text) + " " +
        out["business_address"].map(normalize_text)
    ).str.strip()
    out["combined_norm"] = (
        out["name_core"] + " " + out["name_norm"] + " " + out["address_norm"]
    ).str.strip()
    return out


def _topk_sparse_product(
    queries: sparse.csr_matrix,
    documents: sparse.csr_matrix,
    k: int,
    batch_size: int,
) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    k = min(k, documents.shape[0])
    if k <= 0:
        return
    for start in range(0, queries.shape[0], batch_size):
        # Crucial for million-row sources: ordinary sparse multiplication first
        # materializes every non-zero overlap, which can require many GiB even
        # though only K results per query are needed. sp_matmul_topn performs
        # the pruning during multiplication and bounds the result size.
        scores = sp_matmul_topn(
            queries[start:start + batch_size].tocsr(),
            documents.T,
            top_n=k,
            sort=True,
        ).tocsr()
        for local_row in range(scores.shape[0]):
            row = scores.getrow(local_row)
            if row.nnz == 0:
                yield start + local_row, np.array([], dtype=int), np.array([], dtype=float)
                continue
            yield start + local_row, row.indices, row.data


def _bm25_document_matrix(counts: sparse.csr_matrix, k1: float = 1.5, b: float = 0.75):
    counts = counts.tocsr().astype(np.float32)
    n_docs = counts.shape[0]
    doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)
    doc_len = np.asarray(counts.sum(axis=1)).ravel()
    avg_len = max(float(doc_len.mean()), 1.0)
    rows = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
    denom = counts.data + k1 * (1.0 - b + b * doc_len[rows] / avg_len)
    counts.data = counts.data * (k1 + 1.0) / denom
    counts = counts.multiply(idf).tocsr()
    return counts, idf


def _add_results(
    store: list[dict[str, dict[str, float]]],
    query_offset: int,
    target_ids: np.ndarray,
    iterator,
    score_name: str,
):
    for query_row, target_rows, values in iterator:
        bucket = store[query_offset + query_row]
        for target_row, score in zip(target_rows, values):
            target_id = str(target_ids[target_row])
            bucket.setdefault(target_id, {})[score_name] = float(score)


def generate_candidates(
    source1: pd.DataFrame,
    targets: pd.DataFrame,
    config: RetrievalConfig,
) -> pd.DataFrame:
    """Country-aware hybrid retrieval using char TF-IDF and BM25 weighting."""
    source1 = prepare_records(source1).reset_index(drop=True)
    targets = prepare_records(targets).reset_index(drop=True)
    all_results: list[dict[str, dict[str, float]]] = [dict() for _ in range(len(source1))]

    for country in source1["country"].drop_duplicates():
        q_idx = np.flatnonzero(source1["country"].to_numpy() == country)
        d_idx = np.flatnonzero(targets["country"].to_numpy() == country)
        if not len(q_idx) or not len(d_idx):
            continue
        q = source1.iloc[q_idx]
        d = targets.iloc[d_idx]
        local_store = [dict() for _ in range(len(q))]
        target_ids = d["entity_id"].to_numpy()

        for column, score_name, top_k, ngrams in [
            ("name_norm", "char_name", config.char_name_k, (2, 5)),
            ("address_norm", "char_address", config.char_address_k, (2, 5)),
        ]:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=ngrams, min_df=1,
                max_features=config.char_max_features, sublinear_tf=True,
                dtype=np.float32,
            )
            docs = vectorizer.fit_transform(d[column])
            queries = vectorizer.transform(q[column])
            iterator = _topk_sparse_product(queries, docs, top_k, config.batch_size)
            _add_results(local_store, 0, target_ids, iterator, score_name)

        word_vectorizer = CountVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1,
            max_features=config.word_max_features, binary=False,
        )
        doc_counts = word_vectorizer.fit_transform(d["combined_norm"])
        query_counts = word_vectorizer.transform(q["combined_norm"])
        bm25_docs, idf = _bm25_document_matrix(doc_counts)
        bm25_queries = (query_counts > 0).astype(np.float32).multiply(idf).tocsr()
        iterator = _topk_sparse_product(
            bm25_queries, bm25_docs, config.bm25_k, config.batch_size
        )
        _add_results(local_store, 0, target_ids, iterator, "bm25")

        for local_i, global_i in enumerate(q_idx):
            all_results[global_i] = local_store[local_i]

    rows = []
    for query_i, candidates in enumerate(all_results):
        source_id = source1.iloc[query_i]["entity_id"]
        for target_id, scores in candidates.items():
            rows.append({
                "source1_entity_id": source_id,
                "candidate_entity_id": target_id,
                "char_name": scores.get("char_name", 0.0),
                "char_address": scores.get("char_address", 0.0),
                "bm25": scores.get("bm25", 0.0),
                "retrieval_methods": len(scores),
            })
    return pd.DataFrame(rows)
