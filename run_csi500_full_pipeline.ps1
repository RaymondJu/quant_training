$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$env:QT_UNIVERSE = "csi500"
if (-not $env:QT_DOWNLOAD_SLEEP) {
    $env:QT_DOWNLOAD_SLEEP = "0.5"
}

$steps = @(
    "python data/download.py",
    "python data/clean.py",
    "python factors/value.py",
    "python factors/momentum.py",
    "python factors/quality.py",
    "python factors/volatility.py",
    "python factors/liquidity.py",
    "python factors/additional.py",
    "python factors/risk.py",
    "python factors/preprocess.py",
    "python testing/ic_analysis.py",
    "python testing/quantile_backtest.py",
    "python testing/fama_macbeth.py",
    "python portfolio/backtest.py",
    "python ml/model_comparison.py",
    "python analysis/compare_strategies.py",
    "python analysis/ablation_top_risk.py",
    "python analysis/ablation_risk_v2.py"
)

foreach ($step in $steps) {
    Write-Output ("[{0}] START {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $step)
    Invoke-Expression $step
    if ($LASTEXITCODE -ne 0) {
        throw ("Step failed with exit code {0}: {1}" -f $LASTEXITCODE, $step)
    }
    Write-Output ("[{0}] DONE  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $step)
}

Write-Output ("[{0}] ALL DONE" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
