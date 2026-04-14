# A-Share Multi-Factor Stock Selection System

> A full-stack quantitative research pipeline for CSI 300 covering factor construction, statistical testing, portfolio backtest, and ML enhancement.

![Strategy Comparison](output/analysis/strategy_comparison.png)

---

## Results Summary

### Baseline Portfolio (2015-07 ~ 2025-11, 125 months)

| Strategy | Ann. Return | Sharpe | Max DD | Excess Return | IR |
|---|---|---|---|---|---|
| Equal-weight | 16.97% | 0.755 | -28.0% | 12.55% | 0.98 |
| IC-weight | 18.25% | 0.833 | -26.7% | 13.75% | 1.17 |
| ICIR-weight | 17.37% | 0.796 | -26.7% | 12.87% | 1.09 |
| **CSI 300 (Benchmark)** | **3.93%** | **0.207** | **-36.3%** | -- | -- |

### ML Enhancement (Walk-Forward, 2017-10 ~ 2025-11, 98 months)

| Model | Ann. Return | Sharpe | Max DD | Excess Return | IR | Turnover |
|---|---|---|---|---|---|---|
| LightGBM | 10.94% | 0.538 | -29.3% | 6.98% | 0.973 | 63.6% |
| CatBoost | 10.63% | 0.507 | -26.1% | 6.80% | 0.931 | 60.3% |
| **XGBoost** | **13.09%** | **0.668** | -28.3% | **8.95%** | 1.276 | 64.2% |
| RandomForest | 13.00% | 0.624 | -29.9% | 9.09% | 1.245 | 48.6% |
| Ridge | 12.32% | 0.624 | **-22.4%** | 8.29% | **1.420** | 40.9% |

> All ML models use **strict walk-forward** (24-month train, 3-month val, predict next 3 months) with no look-ahead bias. Universe is filtered to time-varying CSI 300 constituents at each rebalance date.

---

## Architecture

```
quant_training/
├── data/
│   ├── download.py          # AKShare data pipeline (prices + financials)
│   ├── clean.py             # Deduplication, outlier removal, universe tagging
│   ├── benchmark.py         # Unified benchmark loader (total return proxy)
│   └── universe.py          # Time-varying CSI 300 constituent filter
├── factors/
│   ├── utils.py             # TTM calculation, market cap, monthly alignment
│   ├── preprocess.py        # MAD winsorize → Z-score → industry neutralize
│   ├── value.py             # EP, BP, SP
│   ├── momentum.py          # MOM_12_1, REV_1M
│   ├── quality.py           # ROE_TTM, GPM_change, OCF_QUALITY, ASSET_GROWTH
│   ├── volatility.py        # VOL_20D, IVOL, BETA_60D
│   └── liquidity.py         # TURN_1M, ABTURN_1M, AMIHUD, SIZE
├── testing/
│   ├── ic_analysis.py       # Rank IC / ICIR across all factors
│   ├── quantile_backtest.py # Q5-Q1 long-short backtest
│   └── fama_macbeth.py      # Two-pass FM regression (Newey-West t-stats)
├── portfolio/
│   ├── combine.py           # Factor score synthesis (equal/IC/ICIR weights)
│   ├── backtest.py          # Monthly rebalance, Top-N equal-weight, 0.3% cost
│   └── performance.py       # Sharpe, IR, max drawdown, win rate
├── ml/
│   ├── lgbm_model.py        # LightGBM walk-forward with early stopping
│   └── model_comparison.py  # Unified framework: LGBM / CatBoost / XGB / RF / Ridge
├── analysis/
│   └── compare_strategies.py  # Final 8-strategy + benchmark comparison chart
└── config.py                # Global parameters
```

---

## Factor Library (16 Factors)

| Category | Factors | Key Finding |
|---|---|---|
| Value | EP, BP, SP | **Negative alpha in A-shares** (small-cap speculation effect) |
| Momentum | MOM_12_1, REV_1M | Weak positive; ICIR ~ 0.14 |
| Quality | ROE_TTM, GPM_change, OCF_QUALITY, ASSET_GROWTH | OCF_QUALITY significant (t=1.91*) |
| Volatility | VOL_20D, IVOL, BETA_60D | IVOL & VOL_20D positive (FM t~2.3**) |
| Liquidity | TURN_1M, ABTURN_1M, AMIHUD, SIZE | **AMIHUD strongest** (ICIR=0.029, FM t>2**) |

---

## Methodology

### Data
- **Universe**: Time-varying CSI 300 constituents at each rebalance date (~99 stocks in 2015, growing to ~269 in 2025). Only stocks that had been admitted to the index by the rebalance date are eligible, reducing survivorship bias.
- **Benchmark**: CSI 300 Total Return proxy (price index + 2.0% annualized dividend yield). The CSI 300 price index excludes reinvested dividends; we add the historical average dividend yield (~2%) for apples-to-apples comparison with the strategy NAV.
- **Period**: 2015-07 to 2025-12 (monthly frequency)
- **Source**: AKShare (EastMoney backend for financials, adjusted prices)

### Key Design Choices
| Issue | Solution |
|---|---|
| Survivorship bias | Time-varying universe filter based on constituent entry dates (`data/universe.py`) |
| Benchmark understates return | Total return proxy: price index + 2.0% annual dividend yield (`data/benchmark.py`) |
| Back-adjusted prices invalid for cross-stock comparison | Market cap = `turnover/volume * outstanding_share` (VWAP proxy) |
| Cumulative YTD financial statements | TTM = `Q_t + Annual(Y-1) - Q_corr(Y-1)` |
| Look-ahead bias in financials | Use actual `NOTICE_DATE` not report period end |
| Outlier contamination | MAD winsorization: `median +/- 3 * 1.4826 * MAD` |
| Industry concentration | OLS residualization on 28 SW Level-1 industry dummies |

### Walk-Forward Backtest
```
Train [24m] | Val [3m] | -> Predict next 3m
             Train [24m] | Val [3m] | -> Predict next 3m
                          ...
```
No hyperparameter tuning on the full sample -- model selection done within each fold's validation window.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/RaymondJu/quant_training.git
cd quant_training

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

> **Note**: Raw data (~50MB) is not included in this repo. Run the download step first.

```bash
# Step 1: Download data from AKShare (~30 min, requires stable network)
python data/download.py

# Step 2: Clean, deduplicate, and tag time-varying universe
python data/clean.py

# Step 3: Build factor panel
python factors/preprocess.py

# Step 4: Factor testing
python testing/ic_analysis.py
python testing/quantile_backtest.py
python testing/fama_macbeth.py

# Step 5: Portfolio backtest (baseline)
python portfolio/combine.py
python portfolio/backtest.py

# Step 6: ML enhancement
python ml/model_comparison.py

# Step 7: Final comparison chart
python analysis/compare_strategies.py
```

---

## Output

All results are saved to `output/`:

```
output/
├── ic_analysis/          # IC series, IC decay, factor summary table
├── quantile_backtest/    # Q1-Q5 NAV charts per factor
├── fama_macbeth/         # FM coefficients, t-stats, factor correlation
├── portfolio/            # NAV curves and performance for 3 baseline strategies
├── ml/                   # Feature importance, NAV CSVs, model comparison
└── analysis/
    ├── strategy_comparison.png   # 8-strategy + benchmark NAV chart
    └── performance_table.csv     # Final results table
```

---

## Tech Stack

| Component | Library |
|---|---|
| Data | AKShare, pandas, pyarrow |
| Statistics | statsmodels (Newey-West), scipy |
| ML | LightGBM, XGBoost, CatBoost, scikit-learn |
| Visualization | matplotlib |

---

## Key Takeaways

1. **Liquidity premium dominates in A-shares**: AMIHUD illiquidity factor consistently outperforms all other single factors, driven by retail investor herding into liquid small-caps.
2. **Value factors are reversed**: BP/EP show *negative* long-short returns, opposite to developed markets. A-share retail speculation inflates low-BM (growth) stocks.
3. **Ridge achieves the best risk-adjusted return**: Among ML models, Ridge (Ann=12.32%, Sharpe=0.624, MaxDD=-22.4%) delivers the lowest drawdown and highest IR (1.420), confirming monthly factor signals are largely linear.
4. **Survivorship bias matters**: After filtering to time-varying CSI 300 constituents, ML model returns dropped ~7-8pp vs. the static-universe backtest, highlighting the importance of proper universe construction.

---

## Limitations & Future Work

- **Universe**: Restricted to CSI 300 (large-cap). Extending to CSI 500 / CSI 1000 would test factor robustness in mid/small-cap segments where mispricing is larger.
- **Survivorship bias (partial fix)**: Only constituent *entry* dates are available from AKShare; stocks removed from the index cannot be identified. The current fix eliminates "future entry" bias but not "past exit" bias.
- **Benchmark approximation**: The total return benchmark uses a flat 2.0% annual dividend yield added to the price index. Time-varying dividend yields or an ETF-based proxy (510300 back-adjusted) would be more precise.
- **Risk model**: Current portfolio uses simple equal-weight Top-N. A proper risk model (Barra-style multi-factor risk decomposition with optimization) would improve risk-adjusted returns.
- **Trading frictions**: Suspended stocks, daily price limits, and market impact are not modeled. Real-world implementation would require T+1 settlement and limit-up unfillable handling.
- **Factor coverage**: 16 factors across 5 categories. High-frequency factors (intraday volatility, order flow imbalance) and alternative data factors are not included.
- **ML overfitting risk**: Walk-forward mitigates but does not eliminate overfitting. Cross-market validation (e.g., applying same factors to HK or US markets) would provide stronger evidence.
