$ErrorActionPreference = "Stop"

$env:QT_UNIVERSE = "csi500"
$env:QT_FACTOR_SET = "full"

python csi500_daily_alpha_pipeline.py `
  --horizon 5 `
  --top-n 50 `
  --run-lightgbm `
  --run-lightgbm-optuna `
  --optuna-trials 20 `
  --optuna-val-days 63 `
  --optuna-retune-every 25
