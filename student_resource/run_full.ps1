$ErrorActionPreference = "Stop"

python -m pip install -r code/business_entity_resolution/requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m code.business_entity_resolution.src.pipeline `
  --train-dir dataset/train `
  --test-dir dataset/test `
  --output-dir output `
  --work-dir artifacts `
  --folds 5 `
  --char-name-k 60 `
  --char-address-k 60 `
  --bm25-k 75 `
  --batch-size 256 `
  --max-features 250000
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir dataset/test
exit $LASTEXITCODE
