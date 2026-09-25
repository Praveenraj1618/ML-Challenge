$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
# Only use a checkpoint created locally by run_full.ps1; joblib is not safe for untrusted files.
python -u -m code.business_entity_resolution.src.pipeline `
  --train-dir dataset/train `
  --test-dir dataset/test `
  --output-dir output `
  --work-dir artifacts `
  --query-batch 256 `
  --document-batch 10000 `
  --batch-size 64 `
  --predict-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir dataset/test
exit $LASTEXITCODE
