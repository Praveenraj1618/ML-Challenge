# Amazon ML Challenge 2026 — Business Entity Resolution

## Methodology

The system treats entity resolution as a two-stage retrieval and pair-classification
problem. Records are normalized without external lookup. Candidate records are
retrieved within the open-set country label using character TF-IDF and a BM25-style
word model. A supervised LightGBM model then scores the final candidate pairs.

## Candidate generation / blocking

For each Source 1 entity, the pipeline unions three candidate lists:

1. Character n-gram TF-IDF nearest records for normalized business name.
2. Character n-gram TF-IDF nearest records for normalized address.
3. BM25-weighted word retrieval over name and address.

Country is used only as an equality blocking value and is not hard-coded to a fixed
set, so unseen labels such as France are processed normally. The exported
`candidate_pairs.tsv` is the exact union passed to the matching model.

## Model and features

The LightGBM pair classifier uses retrieval scores and ranks, RapidFuzz string
similarities, Jaro-Winkler and Levenshtein similarities, token Jaccard and
containment, numeric agreement/conflict, missing-value indicators, length ratios,
legal-suffix-insensitive equality and source indicators.

Hard negatives arise naturally from the top-ranked non-matching retrieval results.
Training uses grouped folds so all candidate pairs belonging to one Source 1 entity
remain in the same fold.

## Validation

Out-of-fold probabilities are converted into sets of matches and evaluated using
the official entity-level macro F0.5 definition, including singleton entities. The
final probability threshold is selected directly by maximizing this metric.

## Fair play and licensing

The pipeline uses only the supplied challenge data and performs no external business
lookup, geocoding, or internet-based enrichment. All libraries are permissively
licensed; no model above the challenge parameter limit is used.

