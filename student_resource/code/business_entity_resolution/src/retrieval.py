from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

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
    fit_sample: int = 5000
    document_batch: int = 10000
    query_batch: int = 512


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




PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id", "char_name",
                "char_address", "bm25", "retrieval_methods"]


def _save_matrix(path, matrix):
    # Separate arrays allow read-only memory mapping (no NPZ decompression/copy).
    for name in ("data", "indices", "indptr"):
        np.save(str(path) + f".{name}.npy", getattr(matrix, name))


def _load_matrix(path, shape):
    arrays = [np.load(str(path) + f".{name}.npy", mmap_mode="r")
              for name in ("data", "indices", "indptr")]
    return sparse.csr_matrix(tuple(arrays), shape=shape, copy=False)


def iter_candidates(source1, targets, config, work_dir=None):
    """Bounded sparse indexes on disk; vocabulary/IDF estimated on a fixed sample.

    All targets remain searchable. Each shard uses the SAME fitted vectorizer,
    so shard scores can be merged into a country-wide top K. Country blocking
    retains the original baseline assumption; cross-country matches are excluded.
    """
    if min(config.fit_sample, config.document_batch, config.query_batch,
           config.batch_size, config.char_name_k, config.char_address_k,
           config.bm25_k) < 1:
        raise ValueError("Batch sizes, sample size and retrieval K must be positive")
    source1 = source1.reset_index(drop=True)
    targets = targets.reset_index(drop=True)
    target_ids = targets.entity_id.to_numpy()
    qc = source1.country.fillna("").str.strip().str.casefold()
    dc = targets.country.fillna("").str.strip().str.casefold()
    started, completed = monotonic(), 0
    for country in qc.drop_duplicates():
        qrows = np.flatnonzero(qc.to_numpy() == country)
        drows = np.flatnonzero(dc.to_numpy() == country)
        print(f"Index country={country!r}: {len(qrows):,} queries / {len(drows):,} targets", flush=True)
        with TemporaryDirectory(prefix="retrieval-", dir=work_dir) as temporary:
            methods = []
            if len(drows):
                rng = np.random.default_rng(2026)
                sampled = rng.choice(drows, min(config.fit_sample, len(drows)), replace=False)
                sample = prepare_records(targets.iloc[sampled])
                for column, name, k in [("name_norm", "char_name", config.char_name_k),
                                         ("address_norm", "char_address", config.char_address_k),
                                         ("combined_norm", "bm25", config.bm25_k)]:
                    if name == "bm25":
                        vectorizer = CountVectorizer(ngram_range=(1, 2), dtype=np.float32,
                                                    max_features=config.word_max_features)
                    else:
                        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                            dtype=np.float32, sublinear_tf=True, max_features=config.char_max_features)
                    try:
                        fitted = vectorizer.fit_transform(sample[column])
                    except ValueError as exc:
                        if "empty vocabulary" not in str(exc):
                            raise
                        print(f"Skipping empty {name} index", flush=True)
                        continue
                    idf, avg_len = None, None
                    if name == "bm25":
                        df = np.asarray((fitted > 0).sum(axis=0)).ravel()
                        idf = np.log1p((len(sample) - df + .5) / (df + .5)).astype(np.float32)
                        avg_len = max(float(np.asarray(fitted.sum(axis=1)).mean()), 1.)
                    del fitted
                    shards = []
                    for start in range(0, len(drows), config.document_batch):
                        indices = drows[start:start + config.document_batch]
                        records = prepare_records(targets.iloc[indices])
                        matrix = vectorizer.transform(records[column]).tocsr()
                        if name == "bm25":
                            lengths = np.asarray(matrix.sum(axis=1)).ravel()
                            lengths = np.repeat(lengths, np.diff(matrix.indptr))
                            matrix.data *= 2.5 / (matrix.data + 1.5 * (.25 + .75 * lengths / avg_len))
                            matrix.data *= idf[matrix.indices]
                        path = Path(temporary) / f"{name}-{start}.npz"
                        _save_matrix(path, matrix)
                        shards.append((path, indices, matrix.shape))
                        print(f"  {name} index {min(start + config.document_batch, len(drows)):,}/{len(drows):,}", flush=True)
                        del records, matrix
                    methods.append((column, name, k, vectorizer, shards))
                del sample
            for start in range(0, len(qrows), config.query_batch):
                batch = source1.iloc[qrows[start:start + config.query_batch]]
                prepared = prepare_records(batch)
                merged = [{} for _ in range(len(batch))]
                for column, name, k, vectorizer, shards in methods:
                    queries = vectorizer.transform(prepared[column]).tocsr()
                    if name == "bm25":
                        # IDF is already in the document weights; do not square it.
                        queries.data[:] = 1.
                    best = [{} for _ in range(len(batch))]
                    for path, indices, shape in shards:
                        matrix = _load_matrix(path, shape)
                        for row, found, scores in _topk_sparse_product(queries, matrix, k, config.batch_size):
                            bucket = best[row]
                            bucket.update(zip(indices[found].tolist(), scores.tolist()))
                            if len(bucket) > k:
                                best[row] = dict(sorted(bucket.items(), key=lambda item: (-item[1], item[0]))[:k])
                        del matrix
                    for row, bucket in enumerate(best):
                        for target_row, score in bucket.items():
                            target_id = target_ids[target_row]
                            merged[row].setdefault(target_id, {})[name] = score
                rows = []
                for entity_id, candidates in zip(batch.entity_id, merged):
                    for target_id, scores in candidates.items():
                        rows.append((entity_id, target_id, scores.get("char_name", 0.),
                            scores.get("char_address", 0.), scores.get("bm25", 0.), len(scores)))
                completed += len(batch)
                elapsed = monotonic() - started
                eta = elapsed / max(completed, 1) * (len(source1) - completed)
                fraction = completed / len(source1)
                bar = "#" * int(20 * fraction) + "-" * (20 - int(20 * fraction))
                print(f"Retrieval [{bar}] {completed:,}/{len(source1):,} ({fraction:.1%}) | elapsed {elapsed/60:.1f}m | approximate ETA {eta/60:.1f}m", flush=True)
                yield batch, pd.DataFrame(rows, columns=PAIR_COLUMNS)


def generate_candidates(source1, targets, config):
    parts = [pairs for _, pairs in iter_candidates(source1, targets, config)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PAIR_COLUMNS)
