# First-submission recovery build

Run from this directory: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_full.ps1`

## Changes

- Fit vocabulary and IDF on a deterministic sample of at most 5,000 targets per country. Transform ALL targets in shards of 10,000. This avoids fitting a vocabulary on the entire corpus (max_features alone does not bound the fitting peak).
- Store sparse shards as memory-mapped arrays under artifacts, clean them on normal exit, and merge shard top-K scores. A common vectorizer per method/country makes shard scores comparable. Tied boundary IDs may vary with shard size.
- Retain only 256 queries' candidate dictionaries and test features at once; write prediction batches immediately. Candidate K remains 60/60/75. No target subsampling at retrieval time.
- Train on 2,000 randomly selected labeled queries, keeping ALL training targets as distractors, with three grouped folds. Increase --train-queries only after a successful full submission and a memory/runtime measurement.
- Cache repeated normalization within feature batches, avoid per-pair pandas lookups, and store numeric features as float32.
- Fix BM25's duplicate IDF weighting. Vocabulary/IDF sampling and this correction change model scores: retrain, do not reuse an older model.
- Log index counts, retrieval progress bars, elapsed time/rough ETA, feature counts and training iterations.
- Save trained_checkpoint.joblib before inference. To restart inference without retraining, run run_predict.ps1. This rebuilds test indexes and restarts prediction from the beginning; it does NOT resume halfway through a test run.
- Incomplete output uses .partial filenames; final TSVs are replaced only after inference finishes. Do not submit .partial files or older final outputs after a failed run.
- Reject missing columns, blank/duplicate IDs, invalid batch sizes and one-class training folds. Empty text and countries without targets produce explicit empty candidate rows.

## Remaining limits (important)

Raw TSV tables and their ID lookup indexes still reside in RAM. Disk free space must accommodate sparse indexes for the largest country and output TSVs. Disk-backed retrieval can be I/O-heavy: use a local SSD, not a network drive. This is not a promise that every RAM size or runtime budget will work.

Smaller fit samples can lose rare n-grams. Country blocking can miss cross-country matches; blank countries are matched only to blank countries. Query-grouped validation is not a guarantee of independence between related business entities. Reported sample scores are not leaderboard estimates.

The pipeline reports candidate recall and OOF macro F0.5. Inspect these on the full-target run. Benchmark the first test batches on your machine before estimating completion within a deadline. Keep the laptop plugged in and prevent sleep.

Tests: `python -m unittest discover -s tests -v`

Final validation (extra RAM needed for full ID checking):

```powershell
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Both TSVs are required in the submission. Only load trusted locally-created joblib checkpoints.
