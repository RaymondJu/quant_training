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
- 可交易性过滤：建仓日（t+1 开盘）涨停（按板块/日期区分 ±10%/±20%）、停牌、上市不足 120 个交易日的标的不纳入选股。
- 基准：中证 500 / 510500 ETF 净值 proxy。
- 交易成本：主表使用 30 bps，并额外输出 30 / 60 / 100 bps 敏感性。
- 重要限制：股票池是静态名单，存在幸存者偏差；当前结果是研究 demo，不是生产级无偏回测。

> 本表数据为修复标签可得性泄露（见 `NO_LEAKAGE_FIX_REPORT.md`）、补充 Optuna train/val embargo 与可交易性过滤（见 `LEAKAGE_REVIEW_AND_TRADEABILITY_FIX.md`）后的结果，相比早期未修复版本收益显著下降，更接近现实。

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
| IC-weight | 10.31% | 19.78% | 0.521 | -35.87% | 55.9% | 472 | 36.7% | 7.52% | 0.707 |
| IC-weight size-neutral | 10.03% | 19.70% | 0.509 | -33.39% | 56.8% | 472 | 37.0% | 7.26% | 0.698 |
| Ridge | 21.39% | 26.70% | 0.801 | -36.54% | 56.7% | 383 | 35.4% | 18.17% | 1.555 |
| Ridge size-neutral | 12.81% | 25.76% | 0.497 | -38.93% | 54.0% | 383 | 43.6% | 9.61% | 0.851 |
| RidgeCV | 21.30% | 26.70% | 0.798 | -36.48% | 56.7% | 383 | 35.4% | 18.08% | 1.547 |
| LightGBM | 21.07% | 28.89% | 0.729 | -31.60% | 55.9% | 383 | 59.1% | 18.38% | 1.442 |
| XGBoost | 19.17% | 28.67% | 0.668 | -38.06% | 55.6% | 383 | 59.1% | 16.48% | 1.290 |
| CatBoost | 20.78% | 28.87% | 0.720 | -32.95% | 56.4% | 383 | 54.9% | 18.00% | 1.342 |
| RandomForest | 19.78% | 28.53% | 0.693 | -33.23% | 56.9% | 383 | 53.7% | 17.00% | 1.311 |
| LightGBM Optuna | 17.96% | 28.08% | 0.640 | -36.25% | 56.4% | 383 | 61.0% | 15.24% | 1.287 |
| XGBoost Optuna | 20.22% | 28.61% | 0.707 | -34.00% | 55.9% | 383 | 59.2% | 17.53% | 1.434 |
| CatBoost Optuna | 20.80% | 28.37% | 0.733 | -38.18% | 55.9% | 383 | 54.9% | 18.05% | 1.467 |
| RandomForest Optuna | 19.29% | 28.38% | 0.680 | -34.07% | 56.7% | 383 | 54.0% | 16.55% | 1.295 |

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
| RidgeCV | 21.30% | 15.00% | 7.09% | 换手低（~35%），100 bps 下仍为正，成本承受力最强 |
| Ridge | 21.39% | 15.08% | 7.16% | 与 RidgeCV 基本一致，线性模型最稳健 |
| CatBoost Optuna | 20.80% | 11.21% | -0.42% | 60 bps 大幅缩水，100 bps 基本归零 |
| XGBoost Optuna | 20.22% | 9.95% | -2.42% | 换手高（~59%），100 bps 转负 |
| RandomForest Optuna | 19.29% | 9.95% | -1.39% | 100 bps 转负，成本压力明显 |
| LightGBM Optuna | 17.96% | 7.58% | -4.88% | 换手最高（~61%），对成本最敏感，100 bps 跌至 -4.88% |
| IC-weight | 10.31% | 4.36% | -3.09% | 成本敏感，作为传统 baseline 更合适 |

30 bps 偏乐观，60 bps 更接近中性假设，100 bps 是压力测试。由于 CSI500 中盘股流动性并不总是充裕，不能只看 30 bps 结果。加入可交易性过滤后，高换手的树模型在 100 bps 下普遍转负，仅低换手的线性模型（Ridge / RidgeCV）仍能保持正收益。

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
2. 修复标签泄露并加入可交易性过滤后，各策略年化普遍下降 5–7pp；Ridge 从早期泄露口径的 36.47% 回落到 21.39%，市值中性化后进一步降到 12.81%，说明原始收益中既有近端标签泄露，也有明显的小市值与不可成交涨停股暴露。
3. 修复后线性模型反而最稳健：Ridge / RidgeCV 的 Sharpe（约 0.80）高于全部树模型与 Optuna 版本（约 0.64–0.73），且换手低、成本承受力最强。
4. 树模型换手普遍偏高（约 55–61%），在可交易性过滤 + 100 bps 成本下大多转负；Optuna 调参并未带来稳定优势，说明日频价量信号以线性结构为主。
5. IR 列仍偏高（如 Ridge 1.55），主要因基准（中证 500，约 2.86%）偏低且股票池为静态名单存在幸存者偏差——衡量真实 alpha 应以等权全样本为基准，详见 `LEAKAGE_REVIEW_AND_TRADEABILITY_FIX.md`。
6. 若用于面试展示，建议同时展示传统 IC baseline、线性模型、树模型默认版、Optuna 调参版、市值中性版本和成本敏感性，并主动说明幸存者偏差等局限，避免只展示最乐观口径。
