$ErrorActionPreference = "Stop"

$env:QT_UNIVERSE = "csi500"
$env:QT_FACTOR_SET = "full"

python csi500_daily_alpha_pipeline.py `
  --horizon 5 `
  --top-n 50 `
  --run-all-ml `
  --run-all-ml-optuna `
  --optuna-trials 12 `
  --optuna-val-days 63 `
  --optuna-retune-every 25 `
  --lgbm-max-train-rows 50000
