# CSI500 日频价量 Alpha 研究

> 基于静态中证 500 股票池的日频价量 Alpha 研究流程，覆盖日频特征构造、Rank IC 检验、IC 加权组合、Ridge / LightGBM / Optuna-tuned LightGBM walk-forward 横向比较、市值中性化和交易成本敏感性分析。

本分支只展示 `csi500-daily-alpha` 实验结果。旧版月频图表和结果文件已从本分支的 Git 追踪中移除，避免混淆。

---

## 实验口径

- 股票池：静态中证 500 成分股，共 499 只可用股票。
- 数据频率：日频行情，每个交易日、每只股票一条样本。
- 特征来源：纯价量 / 技术面特征，不使用财务基本面因子。
- 预测标签：未来 5 个交易日收益。
- 执行假设：第 t 日收盘后生成信号，第 t+1 日开盘买入，持有 5 个交易日。
- 调仓频率：每 5 个交易日调仓一次。
- 基准：中证 500 / 510500 ETF 净值 proxy。
- 交易成本：主表使用 30 bps，并额外输出 30 / 60 / 100 bps 敏感性。
- 重要限制：股票池是静态名单，存在幸存者偏差；当前结果是研究 demo，不是生产级无偏回测。

---

## 日频 Alpha 特征

| 类别 | 特征 |
|---|---|
| 短期反转 | `REV_1D` |
| 动量 | `MOM_5D`, `MOM_20D`, `MOM_60D` |
| 波动率 | `VOL_20D`, `RANGE_20D` |
| 流动性 | `TURN_5D`, `TURN_20D`, `AMIHUD_20D`, `VOLUME_RATIO_5_20` |
| 偏离度 | `BIAS_20` |
| 风险 / 规模 | `SIZE`, `BETA_60D` |

所有特征在每日横截面上做缩尾和标准化。模型训练和组合构建都只使用历史已知数据，避免 look-ahead。

---

## 主要结果

| 策略 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 胜率 | 调仓期数 | 平均换手 | 超额收益 | IR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IC-weight | 15.05% | 19.87% | 0.757 | -34.60% | 57.0% | 472 | 34.2% | 12.17% | 1.154 |
| IC-weight size-neutral | 13.75% | 19.87% | 0.692 | -31.42% | 56.6% | 472 | 34.8% | 10.95% | 1.082 |
| Ridge | 36.47% | 27.22% | 1.340 | -30.93% | 58.5% | 383 | 35.6% | 32.85% | 2.618 |
| Ridge size-neutral | 26.87% | 26.41% | 1.017 | -33.17% | 55.9% | 383 | 44.2% | 23.34% | 1.939 |
| LightGBM | 40.71% | 30.64% | 1.328 | -28.45% | 57.7% | 383 | 58.0% | 37.91% | 2.641 |
| LightGBM Optuna | 41.72% | 28.89% | 1.444 | -22.84% | 58.5% | 383 | 61.0% | 38.55% | 3.015 |

结果文件：

- `output/csi500/daily_alpha/performance_summary.csv`
- `output/csi500/daily_alpha/daily_ic_summary.csv`
- `output/csi500/daily_alpha/cost_sensitivity.csv`
- `output/csi500/daily_alpha/ridge_returns.csv`
- `output/csi500/daily_alpha/lightgbm_returns.csv`
- `output/csi500/daily_alpha/lightgbm_optuna_returns.csv`
- `output/csi500/daily_alpha/lightgbm_optuna_params.csv`

大文件 `data/processed/csi500/daily_alpha/daily_alpha_panel.parquet` 未上传到 GitHub，需要本地运行脚本重新生成。

---

## 成本敏感性

| 策略 | 30 bps 年化 | 60 bps 年化 | 100 bps 年化 | 结论 |
|---|---:|---:|---:|---|
| IC-weight | 15.05% | 9.27% | 1.99% | 成本敏感，换手成本上升后吸引力明显下降 |
| Ridge | 36.47% | 29.34% | 20.40% | 成本鲁棒性较好 |
| Ridge size-neutral | 26.87% | 18.68% | 8.56% | 中性化后仍有 alpha，但成本压力更明显 |
| LightGBM | 40.71% | 28.95% | 14.77% | 毛收益高，但换手最高，成本上升后优势收窄 |
| LightGBM Optuna | 41.72% | 29.31% | 14.39% | 调参后风险收益改善，但换手进一步上升 |

30 bps 偏乐观，60 bps 更接近中性假设，100 bps 是压力测试。由于 CSI500 中盘股流动性并不总是充裕，不能只看 30 bps 结果。

---

## 运行方式

需要先准备 CSI500 原始日频行情、行业和基准数据。已有本地数据时，直接运行：

```powershell
$env:QT_UNIVERSE = "csi500"
python csi500_daily_alpha_pipeline.py --horizon 5 --top-n 50 --run-lightgbm --run-lightgbm-optuna
```

或使用脚本：

```powershell
.\run_csi500_daily_alpha_pipeline.ps1
```

输出目录：

```text
output/csi500/daily_alpha/
```

---

## 项目结构

```text
quant_training/
|-- csi500_daily_alpha_pipeline.py
|-- run_csi500_daily_alpha_pipeline.ps1
|-- config.py
|-- data/
|-- factors/
|-- portfolio/
|-- ml/
`-- output/csi500/daily_alpha/
```

---

## 主要结论

1. 中证 500 日频价量特征中，`AMIHUD_20D` 和 `SIZE` 的 Rank IC 为正，短中期动量、波动率、换手率类因子多为负 IC，说明该样本中更偏向非流动性、小市值、低波低换手和短期反转逻辑。
2. Ridge 的原始表现较强，但市值中性化后年化从 36.47% 降到 26.87%，说明收益中存在明显小市值暴露。
3. LightGBM 毛收益最高，但平均换手约 58%，对交易成本更敏感。
4. Optuna-tuned LightGBM 使用 walk-forward 内部验证集调参，每 25 个调仓期重新调参一次，每次 20 个 TPE trial；调参后 Sharpe 和回撤改善，但换手升至约 61%，因此仍需结合成本敏感性判断。
5. 若用于面试展示，建议同时展示原始版本、市值中性版本、调参版本和成本敏感性，避免只展示最乐观口径。
