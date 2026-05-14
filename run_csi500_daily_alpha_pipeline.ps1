$ErrorActionPreference = "Stop"

$env:QT_UNIVERSE = "csi500"
$env:QT_FACTOR_SET = "full"

python csi500_daily_alpha_pipeline.py --horizon 5 --top-n 50 --run-lightgbm
