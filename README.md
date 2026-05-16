# CSI500 日频价量 Alpha 研究

> 基于静态中证 500 股票池的日频价量 Alpha 研究流程，覆盖日频特征构造、Rank IC 检验、IC 加权组合、Ridge / RidgeCV / LightGBM / XGBoost / CatBoost / RandomForest 及 Optuna 调参版本的 walk-forward 横向比较、市值中性化和交易成本敏感性分析。

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

所有特征在每日横截面上做缩尾和标准化。模型训练、调参和组合构建都只使用历史已知数据，避免 look-ahead。

---

## 主要结果

| 策略 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 胜率 | 调仓期数 | 平均换手 | 超额收益 | IR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IC-weight | 15.05% | 19.87% | 0.757 | -34.60% | 57.0% | 472 | 34.2% | 12.17% | 1.154 |
| IC-weight size-neutral | 13.75% | 19.87% | 0.692 | -31.42% | 56.6% | 472 | 34.8% | 10.95% | 1.082 |
| Ridge | 36.47% | 27.22% | 1.340 | -30.93% | 58.5% | 383 | 35.6% | 32.85% | 2.618 |
| Ridge size-neutral | 26.87% | 26.41% | 1.017 | -33.17% | 55.9% | 383 | 44.2% | 23.34% | 1.939 |
| RidgeCV | 36.60% | 27.23% | 1.344 | -30.93% | 58.5% | 383 | 35.6% | 32.98% | 2.623 |
| LightGBM | 34.66% | 29.26% | 1.184 | -29.97% | 56.9% | 383 | 59.0% | 31.65% | 2.334 |
| XGBoost | 42.04% | 30.27% | 1.389 | -26.17% | 56.9% | 383 | 58.3% | 39.08% | 2.736 |
| CatBoost | 42.80% | 29.95% | 1.429 | -28.94% | 57.7% | 383 | 54.3% | 39.74% | 2.776 |
| RandomForest | 34.71% | 30.11% | 1.153 | -30.69% | 57.4% | 383 | 52.3% | 31.80% | 2.171 |
| LightGBM Optuna | 38.79% | 28.42% | 1.365 | -26.91% | 57.2% | 383 | 62.3% | 35.59% | 2.863 |
| XGBoost Optuna | 46.68% | 29.43% | 1.586 | -26.18% | 56.7% | 383 | 60.2% | 43.45% | 3.212 |
| CatBoost Optuna | 45.46% | 28.78% | 1.580 | -26.24% | 58.2% | 383 | 55.5% | 42.08% | 3.149 |
| RandomForest Optuna | 39.46% | 29.02% | 1.360 | -29.99% | 55.4% | 383 | 53.6% | 36.34% | 2.750 |

结果文件：

- `output/csi500/daily_alpha/performance_summary.csv`
- `output/csi500/daily_alpha/daily_ic_summary.csv`
- `output/csi500/daily_alpha/cost_sensitivity.csv`
- `output/csi500/daily_alpha/*_returns.csv`
- `output/csi500/daily_alpha/*_optuna_params.csv`

大文件 `data/processed/csi500/daily_alpha/daily_alpha_panel.parquet` 未上传到 GitHub，需要本地运行脚本重新生成。

---

## 成本敏感性

| 策略 | 30 bps 年化 | 60 bps 年化 | 100 bps 年化 | 结论 |
|---|---:|---:|---:|---|
| RidgeCV | 36.60% | 29.45% | 20.48% | 线性模型中最稳健，调参效果接近 Ridge |
| XGBoost Optuna | 46.68% | 34.01% | 18.77% | 当前综合表现最强，但仍受换手成本影响 |
| CatBoost Optuna | 45.46% | 33.83% | 19.74% | 风险收益接近 XGBoost Optuna，换手略低 |
| RandomForest Optuna | 39.46% | 28.67% | 15.55% | 调参后改善明显，但成本压力仍存在 |
| LightGBM Optuna | 38.79% | 26.37% | 11.49% | 调参后优于默认版，但换手偏高 |
| IC-weight | 15.05% | 9.27% | 1.99% | 成本敏感，作为传统 baseline 更合适 |

30 bps 偏乐观，60 bps 更接近中性假设，100 bps 是压力测试。由于 CSI500 中盘股流动性并不总是充裕，不能只看 30 bps 结果。

---

## 调参设计

- 调参方法：Optuna TPE sampler，不使用网格穷举。
- 验证方式：walk-forward 内部验证，只使用当前调仓日前的历史数据。
- 验证集：训练窗口最后 63 个交易日。
- 重调频率：每 25 个调仓期重新调参。
- 每次 trial 数：12。
- 训练抽样：树模型默认最多 50,000 行；RandomForest 单独限制为 15,000 行，避免日频滚动训练过慢。

---

## 运行方式

需要先准备 CSI500 原始日频行情、行业和基准数据。已有本地数据时，直接运行：

```powershell
$env:QT_UNIVERSE = "csi500"
python csi500_daily_alpha_pipeline.py --horizon 5 --top-n 50 --run-all-ml --run-all-ml-optuna --optuna-trials 12 --optuna-val-days 63 --optuna-retune-every 25 --lgbm-max-train-rows 50000
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
3. 在非线性模型中，Optuna 调参后的 XGBoost 和 CatBoost 表现最好，Sharpe 分别为 1.586 和 1.580。
4. 调参能改善树模型的风险收益，但也需要结合换手和成本敏感性判断；100 bps 压力测试下，收益显著收缩。
5. 若用于面试展示，建议同时展示传统 IC baseline、线性模型、树模型默认版、Optuna 调参版、市值中性版本和成本敏感性，避免只展示最乐观口径。
