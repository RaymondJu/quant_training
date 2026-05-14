$ErrorActionPreference = "Stop"

$env:QT_UNIVERSE = "csi500"
$env:QT_FACTOR_SET = "full"

python daily_tech_pipeline.py --horizon 5 --top-n 50 --run-lightgbm
