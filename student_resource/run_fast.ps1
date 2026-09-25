param(
  [ValidateSet("train", "benchmark", "predict", "merge")]
  [string]$Stage = "train",
  [int]$Part = 0,
  [int]$Parts = 1
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -u -m code.business_entity_resolution.src.fast_pipeline $Stage --part $Part --parts $Parts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($Stage -eq "merge") {
  python utils/validate_submission.py --matching output_fast/matching_results.tsv --candidate output_fast/candidate_pairs.tsv --test-dir dataset/test
  exit $LASTEXITCODE
}
