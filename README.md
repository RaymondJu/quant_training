# A-Share Multi-Factor Stock Selection System

> A CSI 300 monthly multi-factor research pipeline covering factor construction, statistical testing, portfolio backtest, and ML comparison.

![Strategy Comparison](output/analysis/strategy_comparison.png)

---

## Final Scope

- Universe: static CSI 300 constituent list
- Frequency: monthly rebalance
- Baseline window: `2015-07 ~ 2025-11` (125 months)
- ML walk-forward window: `2017-10 ~ 2025-11` (98 months)
- Benchmark: `510300` ETF cumulative NAV proxy

This is the **final disclosed scope** for the current version. Historical CSI 300 constituent entry/exit reconstruction was attempted but not completed reliably, so the project keeps the static constituent-list version and **explicitly acknowledges survivorship bias**.

---

## Results Summary

### Baseline Portfolio

| Strategy | Ann. Return | Sharpe | Max DD | Excess Return | IR |
|---|---:|---:|---:|---:|---:|
| Equal-weight | 16.99% | 0.768 | -28.04% | 13.70% | 1.129 |
| IC-weight | 17.10% | 0.785 | -26.69% | 13.77% | 1.189 |
| **ICIR-weight** | **17.90%** | **0.813** | **-26.63%** | **14.56%** | **1.223** |
| CSI 300 / 510300 Benchmark | 3.19% | 0.184 | -33.24% | -- | -- |

### ML Enhancement

| Model | Ann. Return | Sharpe | Max DD | Excess Return | IR | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 13.22% | 0.621 | -22.94% | 10.14% | 0.920 | 56.0% |
| CatBoost | 15.34% | 0.698 | -28.22% | 12.35% | 1.111 | 55.8% |
| XGBoost | 19.94% | 0.942 | -25.53% | 16.62% | 1.440 | 54.4% |
| **RandomForest** | **20.01%** | 0.926 | -22.94% | **16.68%** | 1.354 | 39.6% |
| **Ridge** | 19.89% | **0.994** | **-21.64%** | 16.49% | **1.657** | **31.1%** |

### TopRisk Ablation

| Strategy | Ann. Return | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|---:|
| ICIR (no filter) | 18.96% | 0.940 | -24.01% | 0.790 |
| ICIR + TopRisk | 18.86% | 0.950 | -23.22% | 0.812 |
| Ridge (no filter) | 19.89% | 0.994 | -21.64% | 0.919 |
| Ridge + TopRisk | 18.15% | 0.905 | -19.81% | 0.917 |
| XGBoost (no filter) | 19.94% | 0.942 | -25.53% | 0.781 |
| XGBoost + TopRisk | 18.18% | 0.879 | -23.90% | 0.761 |

### TopRisk v1 vs v2

| Strategy | Ann. Return | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|---:|
| ICIR (no filter) | 18.96% | 0.940 | -24.01% | 0.790 |
| **ICIR + TopRisk v1 equal** | **18.86%** | **0.950** | **-23.22%** | **0.812** |
| ICIR + TopRisk v2 IC-weighted | 16.39% | 0.802 | -24.36% | 0.673 |

Current interpretation:
- Baseline: `ICIR-weight` is the strongest traditional multi-factor combination.
- ML: `RandomForest` has the highest annualized return, while `Ridge` has the best risk-adjusted profile.
- TopRisk: useful mainly as a drawdown-control layer, not a return-enhancement layer.
- v2 weighted risk filter: worse than v1 equal-weight, so v1 remains the production choice for this project version.

---

## Methodology

### Data

- Universe: static CSI 300 constituent list
- Benchmark: 510300 ETF cumulative NAV proxy
- Raw source: AKShare
- Label: `ret_next_month`
- Trading assumption: monthly Top-N equal-weight portfolio with 0.3% one-way transaction cost

### Key Design Choices

| Issue | Solution |
|---|---|
| Benchmark understates return | Prefer 510300 cumulative NAV proxy over price index + fixed 2% dividend approximation |
| Back-adjusted prices are not suitable for cross-stock market cap comparison | Market cap proxy = `turnover / volume * outstanding_share` |
| Cumulative YTD financial statements | TTM = `Q_t + Annual(Y-1) - Q_corr(Y-1)` |
| Financial look-ahead bias | Main PIT alignment uses `NOTICE_DATE` in factor construction utilities |
| Outliers | MAD winsorization |
| Industry concentration | Cross-sectional industry neutralization with SW Level-1 dummies |

### Walk-Forward ML

```text
Train [24m] | Val [3m] | -> Predict next period
             Train [24m] | Val [3m] | -> Predict next period
                          ...
```

No full-sample hyperparameter selection is used in the default ML comparison.

---

## Important Limitations

### 1. Survivorship Bias

This version uses a **static CSI 300 constituent list**, not a fully reconstructed historical constituent path. That means:

- stocks that later entered or remained in CSI 300 may appear in earlier periods when they were not actually in the index
- removed constituents are not fully modeled
- backtest results are therefore **research results with acknowledged survivorship bias**, not production-grade bias-free results

This limitation is deliberately disclosed rather than hidden.

### 2. Benchmark Is a Proxy

The benchmark now uses `510300` ETF cumulative NAV as the preferred local total-return proxy. This is materially better than the old fixed 2.0% dividend approximation, but it is still an ETF proxy with tracking error and fee effects.

### 3. Trading Frictions Are Simplified

Suspensions, limit-up/limit-down execution constraints, and market impact are not modeled.

---

## Architecture

```text
quant_training/
├── data/
│   ├── download.py
│   ├── clean.py
│   └── benchmark.py
├── factors/
│   ├── utils.py
│   ├── value.py
│   ├── momentum.py
│   ├── quality.py
│   ├── volatility.py
│   ├── liquidity.py
│   ├── additional.py
│   ├── risk.py
│   └── preprocess.py
├── testing/
│   ├── ic_analysis.py
│   ├── quantile_backtest.py
│   └── fama_macbeth.py
├── portfolio/
│   ├── combine.py
│   ├── backtest.py
│   └── performance.py
├── ml/
│   └── model_comparison.py
├── analysis/
│   ├── compare_strategies.py
│   ├── ablation_top_risk.py
│   └── ablation_risk_v2.py
└── output/
```

---

## Quick Start

```bash
python data/download.py
python data/clean.py
python factors/value.py
python factors/momentum.py
python factors/quality.py
python factors/volatility.py
python factors/liquidity.py
python factors/additional.py
python factors/risk.py
python factors/preprocess.py
python testing/ic_analysis.py
python testing/quantile_backtest.py
python testing/fama_macbeth.py
python portfolio/backtest.py
python ml/model_comparison.py
python analysis/compare_strategies.py
python analysis/ablation_top_risk.py
python analysis/ablation_risk_v2.py
```

---

## Output

Key files:

- `output/analysis/performance_table.csv`
- `output/analysis/strategy_comparison.png`
- `output/ml/model_comparison.csv`
- `output/ml/model_comparison_nav.png`
- `output/ablation/performance_comparison.csv`
- `output/ablation/v2_performance.csv`

---

## Takeaways

1. Traditional multi-factor baseline is already strong on this sample.
2. In the current static-CSI300 version, `RandomForest`, `XGBoost`, and `Ridge` all outperform the traditional baseline over the common ML window.
3. `Ridge` is the most stable ML model by Sharpe, IR, drawdown, and turnover.
4. The benchmark issue has been corrected by switching to the 510300 cumulative NAV proxy.
5. The project keeps the static-universe version and **honestly discloses survivorship bias** instead of overstating backtest rigor.
