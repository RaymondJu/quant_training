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
- 买入约束：建仓日（t+1 开盘）涨停（按板块/日期区分 ±10%/±20%）、停牌或上市不足 120 个交易日的标的不纳入选股。
- 卖出约束：已有持仓若在调仓日开盘跌停或停牌，则不能卖出并继续占用组合名额；只用剩余名额买入新标的。
- 停牌处理：按全市场交易日历补齐股票缺失交易日，停牌期间使用最近可用价格盯市，不再由单股票 `shift()` 跳过停牌期。
- 基准：中证 500 / 510500 ETF 净值 proxy。
- 交易成本：主表使用 30 bps，并额外输出 30 / 60 / 100 bps 敏感性。
- 重要限制：本地没有历史 ST / *ST 状态和交易所官方涨跌停价。ST 过滤尚未实现，涨跌停使用板块和日期阈值近似识别。
- 重要限制：股票池是静态名单，存在幸存者偏差；当前结果是研究 demo，不是生产级无偏回测。

> 当前版本在标签可得性、Optuna embargo 和买入过滤基础上，进一步加入跌停无法卖出与停牌锁仓。实现与数据缺口见 `CSI500_DAILY_TRADE_CONSTRAINT_FIX_REPORT.md`。

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

## 最新交易约束结果

本轮使用严格市场交易日历重跑四个代表策略。参数仍为 2015-2025、5 日调仓、Top 50、30 bps。

| 策略 | 修复前年化 | 最新年化 | 变化 | 最新波动 | 最新 Sharpe | 最新最大回撤 | 锁仓调仓次数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IC-weight | 10.31% | 10.19% | -0.13pp | 19.62% | 0.519 | -34.55% | 153 |
| IC-weight size-neutral | 10.03% | 8.58% | -1.45pp | 19.33% | 0.444 | -33.41% | 153 |
| Ridge | 21.39% | 21.14% | -0.26pp | 26.67% | 0.793 | -35.37% | 52 |
| Ridge size-neutral | 12.81% | 11.74% | -1.07pp | 25.77% | 0.456 | -40.86% | 56 |

执行网格共识别 20,805 条次日停牌/无报价记录和 1,664 条次日跌停不可卖记录。普通 IC-weight 和 Ridge 变化较小，市值中性版本受锁仓影响更明显。

明细结果：

- `output/csi500/daily_alpha/trade_constraint_validation/performance_before_after.csv`
- `output/csi500/daily_alpha/trade_constraint_validation/constraint_coverage.csv`
- `CSI500_DAILY_TRADE_CONSTRAINT_FIX_REPORT.md`

### 上一轮完整模型比较

以下树模型、RidgeCV 和 Optuna 数字尚未按最新卖出约束完整重跑，仅保留作为上一轮“买入端可交易性过滤”基线，不应与上表混作同一最终口径。

| 策略 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 胜率 | 调仓期数 | 平均换手 | 超额收益 | IR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
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

> 下表尚未按最新停牌/跌停卖出约束重跑，是上一轮完整模型的成本敏感性基线。最新四策略的 30 bps 结果以上方主表为准。

| 策略 | 30 bps 年化 | 60 bps 年化 | 100 bps 年化 | 结论 |
|---|---:|---:|---:|---|
| RidgeCV | 21.30% | 15.00% | 7.09% | 换手低（~35%），100 bps 下仍为正，成本承受力最强 |
| Ridge | 21.39% | 15.08% | 7.16% | 与 RidgeCV 基本一致，线性模型最稳健 |
| CatBoost Optuna | 20.80% | 11.21% | -0.42% | 60 bps 大幅缩水，100 bps 基本归零 |
| XGBoost Optuna | 20.22% | 9.95% | -2.42% | 换手高（~59%），100 bps 转负 |
| RandomForest Optuna | 19.29% | 9.95% | -1.39% | 100 bps 转负，成本压力明显 |
| LightGBM Optuna | 17.96% | 7.58% | -4.88% | 换手最高（~61%），对成本最敏感，100 bps 跌至 -4.88% |
| IC-weight | 10.31% | 4.36% | -3.09% | 成本敏感，作为传统 baseline 更合适 |

30 bps 偏乐观，60 bps 更接近中性假设，100 bps 是压力测试。由于 CSI500 中盘股流动性并不总是充裕，不能只看 30 bps 结果。上一轮加入买入端可交易性过滤后，高换手的树模型在 100 bps 下普遍转负；最新卖出约束结果仍需在完整模型重跑后更新该表。

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
2. 修复标签泄露和买入端不可成交问题后，Ridge 从早期泄露口径的 36.47% 回落到 21.39%；进一步加入停牌/跌停卖出约束后为 21.14%，说明普通 Ridge 对新增约束相对稳健。
3. 市值中性策略更受卖出约束影响：IC-weight size-neutral 从 10.03% 降至 8.58%，Ridge size-neutral 从 12.81% 降至 11.74%，表明其更容易持有停牌或跌停锁仓标的。
4. 上一轮完整比较中线性模型的 Sharpe 高于树模型与 Optuna 版本，且换手更低；树模型尚需按最新卖出约束完整重跑后再形成最终横向结论。
5. 上一轮完整模型表中的 IR 仍偏高（如 Ridge 1.55），主要因基准（中证 500，约 2.86%）偏低且股票池为静态名单存在幸存者偏差；衡量真实 alpha 应以等权全样本为基准，详见 `LEAKAGE_REVIEW_AND_TRADEABILITY_FIX.md`。
6. 历史 ST / *ST 状态仍是明确数据缺口。对外展示时必须同时说明静态成分股幸存者偏差、ST 过滤缺失和官方涨跌停价缺失，避免把当前结果描述为生产级无偏回测。
