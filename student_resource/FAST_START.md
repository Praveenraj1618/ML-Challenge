# Selective lookup recovery baseline

This is a separate baseline, not an acceleration with identical predictions. Keep the old model and outputs. The old `trained_checkpoint.joblib` is NOT used: retrieval features changed, so this path trains its own `artifacts_fast/fast_model.joblib` and retunes the threshold.

## Start on one laptop

In `student_resource`, using the existing `ml` environment:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fast.ps1 -Stage train
```

This streams targets into `artifacts_fast/train.sqlite`, trains on 2,000 sampled labeled queries and prints candidate recall and OOF F0.5. No full target tables are held in pandas. The index is persistent and ingestion resumes from its last committed batch if interrupted (10,000 records by default in the tuned version). The initial count and file-fingerprint passes can take time before the progress bar appears.

### Compatible indexing performance update

The tuned version can resume a database started by the original fast-recovery script. It keeps exactly the same keys, schema, fingerprints and scoring. Stop the Python run with Ctrl+C and wait for the PowerShell prompt before applying the update. Preserve the entire `artifacts_fast` folder, including any SQLite WAL/SHM files. Then rerun the same command; it rechecks inputs, skips committed records and continues. A current uncommitted batch may be repeated.

Writes are sorted by the index primary key, the SQLite cache grows from 64 MiB to 512 MiB, batches grow from 5,000 to 10,000 and WAL checkpoints are less frequent. SQLite FULL synchronization is retained. The cache is bounded; this is not an unlimited RAM setting. Override with `--index-cache-mb` and `--index-batch` when invoking the Python module directly. Index output now includes the latest batch rate and DB write duration, instead of only an average since startup. The first resumed batch includes time spent skipping earlier input rows, so use subsequent batches to assess speed.

On a synthetic local benchmark with an existing 150,000-record index, appending 30,000 records (20 keys each), including a final WAL checkpoint, took 2.34 seconds with the old writes and 1.42 seconds with the tuned writes. This measures only the database-write path on the development machine, NOT full-data preprocessing or prediction speed on your laptop. It does not guarantee a 1.65x end-to-end gain.

After successful training:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fast.ps1 -Stage benchmark
```

This builds/reuses `artifacts_fast/test.sqlite` and benchmarks a deterministic random sample of 2,048 queries across the full test file. Read the reported country distribution. Do not assume full-scale speed from the developer's small bundled sample. Use the measured queries/second and time left to decide whether to run prediction. Initial indexing time is separate from the reported prediction estimate. Cache warmth and country mix can affect speed.

Then:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fast.ps1 -Stage predict
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fast.ps1 -Stage merge
```

The merge stage runs the supplied validator. Live leaderboard upload is `output_fast/matching_results.tsv` only. For final packaging, include it and `output_fast/candidate_pairs.tsv` under `output/`, all source and requirements, reproduction instructions, and an updated methodology. Do not describe this model as BM25/TF-IDF: it uses selective blocking and fuzzy pair features.

## Restart

Rerun the SAME command. Completed index batches and completed prediction batches are reused. Leave data, model, batch size, partition settings and retrieval parameters unchanged. Changed input fingerprints cause an error; choose a new work/output directory instead of deleting old results. Index corruption is not automatically repaired. Keep SQLite files on a local SSD.

## Three laptops (optional)

Train once. Once no process is using the files, copy the same `fast_model.joblib`, completed `test.sqlite` and full test data to each laptop at the same relative paths. Alternatively each laptop builds its own identical test index. Do not copy an actively written SQLite database. Each machine keeps complete target data; only queries are partitioned.

```powershell
# Laptop 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fast.ps1 -Stage predict -Part 0 -Parts 3
# Laptop 2
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fast.ps1 -Stage predict -Part 1 -Parts 3
# Laptop 3
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fast.ps1 -Stage predict -Part 2 -Parts 3
```

Collect `output_fast/part-0`, `part-1`, `part-2` on one machine, then:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_fast.ps1 -Stage merge -Parts 3
```

Do not mix one-part and three-part runs in the same output directory. Merge rejects mismatched model/data/settings, duplicate prediction IDs and missing queries. The model must be identical on every laptop; do not separately retrain it on each.

## Method and limitations

Normalized full/sorted names, address strings, selected name tokens/prefixes, character anchors, address tokens, numbers and token-number combinations generate country-specific keys. SQLite maps those keys to target IDs. Keys with more than 256 hits are skipped; up to 1,500 records are shortlisted by shared-key count, then at most 60 are retained using name/address fuzzy similarity. These exact 60-or-fewer pairs are scored by LightGBM and exported in candidate_pairs.tsv. Each query performs bounded index lookups instead of scanning millions of target vectors.

Selective blocks and caps can miss matches, especially common names, translated names, heavily corrupted addresses and cross-country errors. Check candidate recall on the FULL target index before trusting scores. Increasing caps/candidates changes the trained retrieval distribution; use a new work directory and retrain. The candidate recall measured against the bundled sample is not full-data recall. Grouped OOF by source ID does not guarantee related businesses are independent, and threshold tuning on OOF can be optimistic.

This is intended to establish a timely baseline, not a claim of leaderboard quality or guaranteed runtime. No external business lookups or pretrained neural models are used. Existing MIT-licensed LightGBM is retained. The GPU is not used. Only load trusted local joblib files.

Tests: `python -m unittest discover -s tests -v`

For stronger final validation (uses additional RAM):

```powershell
python utils/validate_submission.py --matching output_fast/matching_results.tsv --candidate output_fast/candidate_pairs.tsv --test-dir dataset/test --check-ids
```
