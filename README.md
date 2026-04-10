# A-Share Multi-Factor Stock Selection System

> A full-stack quantitative research pipeline for CSI 300 (沪深300) covering factor construction → statistical testing → portfolio backtest → ML enhancement.

---

## Results Summary

### Baseline Portfolio (2015-07 ~ 2025-11, 125 months)

| Strategy | Ann. Return | Sharpe | Max DD | Excess Return | IR |
|---|---|---|---|---|---|
| Equal-weight | 16.97% | 0.755 | -28.0% | 14.80% | 1.16 |
| IC-weight | 18.25% | 0.833 | -26.7% | 16.03% | 1.37 |
| ICIR-weight | 17.37% | 0.796 | -26.7% | 15.13% | 1.28 |
| **CSI 300 (Benchmark)** | **1.87%** | **0.098** | **-39.9%** | — | — |

### ML Enhancement (Walk-Forward, 2017-10 ~ 2025-11, 98 months)

| Model | Ann. Return | Sharpe | Max DD | Excess Return | IR | Turnover |
|---|---|---|---|---|---|---|
| LightGBM | 18.50% | 0.880 | -22.9% | 16.39% | 1.541 | 71.8% |
| CatBoost | 20.13% | 0.914 | -24.8% | 18.21% | 1.700 | 69.1% |
| XGBoost | 21.09% | 1.000 | -22.5% | 18.95% | 1.764 | 70.1% |
| **RandomForest** | **21.50%** | 1.009 | -22.9% | **19.33%** | 1.693 | 55.4% |
| **Ridge** | 20.20% | **1.018** | **-20.9%** | 17.87% | **1.771** | 46.1% |

> All ML models use **strict walk-forward** (24-month train, 3-month val, predict next 3 months) with no look-ahead bias.

---

## Architecture

```
quant_training/
├── data/
│   ├── download.py          # AKShare data pipeline (prices + financials)
│   └── clean.py             # Deduplication, outlier removal
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
│   └── compare_strategies.py  # Final 6-strategy + benchmark comparison chart
└── config.py                # Global parameters
```

---

## Factor Library (16 Factors)

| Category | Factors | Key Finding |
|---|---|---|
| Value | EP, BP, SP | **Negative alpha in A-shares** (small-cap speculation effect) |
| Momentum | MOM_12_1, REV_1M | Weak positive; ICIR ≈ 0.14 |
| Quality | ROE_TTM, GPM_change, OCF_QUALITY, ASSET_GROWTH | OCF_QUALITY significant (t=1.91*) |
| Volatility | VOL_20D, IVOL, BETA_60D | IVOL & VOL_20D positive (FM t≈2.3**) |
| Liquidity | TURN_1M, ABTURN_1M, AMIHUD, SIZE | **AMIHUD strongest** (ICIR=0.227, FM t=3.85***) |

### Factor Validation Highlights

- **AMIHUD** (illiquidity): IC mean=0.025, ICIR=0.227, long-short Ann=13.77%, Sharpe=1.055 — strongest single factor
- **Fama-MacBeth multi-factor**: SIZE (***), BP (−***), SP (**), AMIHUD (**) jointly significant
- **Value reversal**: BP long-short Ann=-10.8%, confirming A-share value factor anomaly

---

## Methodology

### Data
- Universe: CSI 300 constituents (~279 stocks)
- Period: 2015-07 to 2025-12 (monthly frequency)
- Source: AKShare (EastMoney backend for financials, adjusted prices)

### Key Design Choices
| Issue | Solution |
|---|---|
| Back-adjusted prices invalid for cross-stock comparison | Market cap = `turnover/volume × outstanding_share` (VWAP proxy) |
| Cumulative YTD financial statements | TTM = `Q_t + Annual(Y-1) - Q_corr(Y-1)` |
| Look-ahead bias in financials | Use actual `NOTICE_DATE` not report period end |
| Outlier contamination | MAD winsorization: `median ± 3 × 1.4826 × MAD` |
| Industry concentration | OLS residualization on 28 SW Level-1 industry dummies |

### Walk-Forward Backtest
```
Train [24m] | Val [3m] | → Predict next 3m
             Train [24m] | Val [3m] | → Predict next 3m
                          ...
```
No hyperparameter tuning on the full sample — model selection done within each fold's validation window.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/quant-factor-model.git
cd quant-factor-model

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

# Step 2: Clean and deduplicate
python data/clean.py

# Step 3: Build factor panel
python -c "
from factors.value import build as v
from factors.momentum import build as m
from factors.quality import build as q
from factors.volatility import build as vol
from factors.liquidity import build as liq
from factors.preprocess import build_factor_panel
build_factor_panel()
"

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
    ├── strategy_comparison.png   # 6-strategy + benchmark NAV chart
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
3. **Linear models generalize well**: Ridge regression achieves the best risk-adjusted return (Sharpe=1.018, IR=1.771), confirming monthly factor signals are largely linear.
4. **ML adds alpha over baseline**: All 5 ML models outperform the ICIR-weighted baseline (~17.4% Ann), with XGBoost/RandomForest topping 21% annual return.
