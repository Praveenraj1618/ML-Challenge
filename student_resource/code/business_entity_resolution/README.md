# Business Entity Resolution Baseline

Competition-grade classical baseline for Amazon ML Challenge 2026. It uses:

- Unicode and transliteration-aware normalization
- Country-aware character TF-IDF retrieval for names and addresses
- BM25-weighted word retrieval
- Candidate union and retrieval-rank features
- String, token, numeric and contradiction features
- Grouped out-of-fold LightGBM training
- Direct optimization of entity-level macro F0.5
- Exact generation of `matching_results.tsv` and `candidate_pairs.tsv`
- Memory-bounded top-K sparse multiplication for million-row datasets

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On Linux/macOS, activation is `source .venv/bin/activate`.

## Expected data layout

```text
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── utils/validate_submission.py
└── code/business_entity_resolution/
```

## Run

Run from `student_resource/`:

```bash
python -m code.business_entity_resolution.src.pipeline \
  --train-dir dataset/train \
  --test-dir dataset/test \
  --output-dir output \
  --work-dir artifacts
```

Windows PowerShell:

```powershell
python -m code.business_entity_resolution.src.pipeline `
  --train-dir dataset/train `
  --test-dir dataset/test `
  --output-dir output `
  --work-dir artifacts
```

Validate:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Scaling controls

For memory pressure, lower `--max-features` or `--batch-size`. For higher
candidate recall, increase `--char-name-k`, `--char-address-k`, and `--bm25-k`.
Always measure candidate recall and OOF macro F0.5 after changing them.

Candidate multiplication uses `sparse-dot-topn`, so it never materializes the
full sparse similarity product. This is required for the complete challenge
data, where ordinary SciPy multiplication may attempt multi-gigabyte temporary
allocations.

## Sample-bundle verification

Using the supplied 5,000-entity labelled sample with grouped five-fold
validation, the default wide configuration achieved:

- Candidate link recall: `0.999367`
- Source 1 entities retaining every true match: `0.997800`
- Grouped out-of-fold macro F0.5: `0.992088`
- Selected probability threshold: `0.8100`

These are sample-validation results, not a leaderboard guarantee. Re-run the
pipeline on the complete challenge data and use the newly generated threshold.

This baseline performs no external identity lookup or external data enrichment.
