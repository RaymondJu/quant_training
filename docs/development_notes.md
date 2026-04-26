# Development Notes

## Current Final State

This file records the **current accepted project state** after code fixes and the latest full rerun.

### Scope

- Universe: static CSI 300 constituent list
- Benchmark: 510300 ETF cumulative NAV proxy
- Baseline window: `2015-07 ~ 2025-11`
- ML window: `2017-10 ~ 2025-11`

### Why This Scope Was Frozen

Historical CSI 300 constituent entry/exit reconstruction was attempted but did not stabilize into a reliable workflow. Instead of continuing to present a half-fixed dynamic universe, the project now keeps the static-universe version and explicitly acknowledges survivorship bias.

### Fixed Issues

- Momentum factor `MOM_12_1` timing corrected
- `standardize()` uses `ddof=0`
- `clean.py` documents approximate PIT risk on estimated announce dates
- ML turnover calculation matches the main backtest definition
- Benchmark uses 510300 cumulative NAV proxy rather than the old fixed-dividend approximation

### Latest Headline Results

| Strategy | Ann. Return | Sharpe | Max DD |
|---|---:|---:|---:|
| Equal-weight | 16.99% | 0.768 | -28.04% |
| IC-weight | 17.10% | 0.785 | -26.69% |
| ICIR-weight | 17.90% | 0.813 | -26.63% |
| XGBoost | 19.94% | 0.942 | -25.53% |
| RandomForest | 20.01% | 0.926 | -22.94% |
| Ridge | 19.89% | 0.994 | -21.64% |
| Benchmark | 3.19% | 0.184 | -33.24% |

### Current Interpretation

- Traditional baseline winner: `ICIR-weight`
- ML return winner: `RandomForest`
- ML stability winner: `Ridge`
- TopRisk helps mainly as a drawdown-control layer
- TopRisk v2 IC-weighted version is worse than v1 equal-weight

### Files To Trust

- `README.md`
- `PROJECT_SUMMARY_FOR_INTERVIEW.md`
- `docs/TOP_RISK_FILTER.md`
- `output/analysis/performance_table.csv`
- `output/ml/model_comparison.csv`
- `output/ablation/performance_comparison.csv`
- `output/ablation/v2_performance.csv`

If another document contradicts the above, treat that older document as outdated.
