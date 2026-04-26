# Top Risk Filter

> Document date: 2026-04-25  
> Experiment window: `2017-10 ~ 2025-11`  
> Universe: **static CSI 300 constituent list**  
> Known limitation: survivorship bias is present and explicitly acknowledged  
> Benchmark: `510300 ETF` cumulative NAV proxy

---

## 1. Purpose

The TopRisk layer is a **post-selection filter**, not an alpha model.

Workflow:

1. Select Top-N stocks from the baseline alpha or ML model.
2. Compute `TOP_RISK_SCORE`.
3. Remove the highest-risk tail of the selected names.
4. Reweight the remaining names equally.

The intent is to reduce exposure to short-term overheated stocks rather than to improve the core signal model itself.

---

## 2. Risk Factors

The current risk score uses four monthly risk descriptors:

| Factor | Meaning |
|---|---|
| `BIAS_20` | price deviation from 20-day moving average |
| `UPSHADOW_20` | recent upper-shadow pressure |
| `VOL_SPIKE_6M` | abnormal turnover spike vs. recent history |
| `RET_6M` | trailing 6-month cumulative return |

v1 risk score:

```text
TOP_RISK_SCORE_v1 = mean(BIAS_20_z, UPSHADOW_20_z, VOL_SPIKE_6M_z, RET_6M_z)
```

The risk factors are shifted by one period before use so the filter does not consume future information.

---

## 3. Current Ablation Result

Source: `output/ablation/performance_comparison.csv`

| Strategy | Ann. Return | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|---:|
| ICIR (no filter) | 18.96% | 0.940 | -24.01% | 0.790 |
| ICIR + TopRisk | 18.86% | 0.950 | -23.22% | 0.812 |
| Ridge (no filter) | 19.89% | 0.994 | -21.64% | 0.919 |
| Ridge + TopRisk | 18.15% | 0.905 | -19.81% | 0.917 |
| XGBoost (no filter) | 19.94% | 0.942 | -25.53% | 0.781 |
| XGBoost + TopRisk | 18.18% | 0.879 | -23.90% | 0.761 |

Interpretation:

- `ICIR + TopRisk` improves Sharpe, drawdown, and Calmar slightly, but does not improve return.
- `Ridge + TopRisk` reduces drawdown but gives up too much return and Sharpe.
- `XGBoost + TopRisk` also trades return for a moderate drawdown improvement.

So in the **current final version**, TopRisk should be framed as a **drawdown-control layer**, not as a return-enhancement layer.

---

## 4. Conditional Recommendation

### ICIR

TopRisk is acceptable for `ICIR` because it slightly improves risk-adjusted metrics without a large performance collapse.

### Ridge

TopRisk is **not universally recommended** for Ridge.

Current comparison:

- `Ridge`: 19.89% / Sharpe 0.994 / MaxDD -21.64%
- `Ridge + TopRisk`: 18.15% / Sharpe 0.905 / MaxDD -19.81%

Interpretation:

- If the investor values **return efficiency and Sharpe**, keep Ridge without TopRisk.
- If the investor has a **harder drawdown preference**, TopRisk can still be described as a conditional option.

### XGBoost

The same logic applies to XGBoost:

- without TopRisk: stronger return
- with TopRisk: somewhat lower tail risk

So this should also be presented as a conditional choice rather than a universally better configuration.

---

## 5. v1 Equal vs v2 IC-Weighted

Source: `output/ablation/v2_performance.csv`

| Strategy | Ann. Return | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|---:|
| ICIR (no filter) | 18.96% | 0.940 | -24.01% | 0.790 |
| **ICIR + TopRisk v1 equal** | **18.86%** | **0.950** | **-23.22%** | **0.812** |
| ICIR + TopRisk v2 IC-weighted | 16.39% | 0.802 | -24.36% | 0.673 |

Conclusion:

- v2 IC-weighted risk aggregation underperforms v1 equal-weight.
- The more complicated weighting rule did not improve the final portfolio.
- The official project conclusion remains: **equal-weight is the more robust choice**.

---

## 6. What To Say Externally

Recommended wording:

> I added a TopRisk post-filter to remove the hottest and most overextended names after the main model selected Top-N stocks. In the final static-CSI300 version, this layer mainly helped drawdown control for the ICIR baseline, but it did not consistently improve return. For Ridge and XGBoost it became a conditional risk-preference choice rather than a universally better setting. I also tested a more complex IC-weighted v2 risk score, but it underperformed the simple equal-weight version.

This is the correct final positioning for this document. Do not describe TopRisk as a universally return-improving layer.
