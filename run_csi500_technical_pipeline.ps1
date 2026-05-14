$ErrorActionPreference = "Stop"

$env:QT_UNIVERSE = "csi500"
$env:QT_FACTOR_SET = "technical"

python run_csi500_technical_pipeline.py
