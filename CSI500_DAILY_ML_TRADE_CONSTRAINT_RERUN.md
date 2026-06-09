# CSI500 日频机器学习全模型补跑报告

生成日期：2026-06-09
执行面板：`data/processed/csi500/daily_alpha/daily_alpha_panel_trade_constraints.parquet`
承接报告：`CSI500_DAILY_TRADE_CONSTRAINT_FIX_REPORT.md`

## 1. 补跑范围

在 2026-06-07 已完成的 IC-weight、Ridge 及市值中性版本基础上，本次补跑：

- RidgeCV
- LightGBM、XGBoost、CatBoost、RandomForest
- 上述四类树模型的 Optuna 版本

所有模型统一使用同一套次日买入过滤、停牌/跌停卖出锁仓、停牌期间盯市和换手成本逻辑。

## 2. 参数口径

| 参数 | 数值 |
|---|---:|
| 预测周期与调仓间隔 | 5 个交易日 |
| 持仓数量 | Top 50 等权 |
| 训练窗口 | 504 个交易日 |
| 首次预测前暖启动 | 756 个交易日 |
| 基准交易成本 | 30 bps |
| 树模型最大训练样本 | 50,000 行 |
| RandomForest 最大训练样本 | 15,000 行 |
| Optuna trials | 12 |
| Optuna 验证窗口 | 63 个交易日 |
| Optuna 重调频率 | 每 25 次调仓 |
| Embargo | 5 个交易日 |

执行面板共 1,109,061 行、2,668 个信号日、499 只股票。机器学习模型均产生 383 个调仓期。

## 3. 最新绩效

| 策略 | 年化收益 | Sharpe | 最大回撤 | 平均换手 | 锁仓调仓次数 |
|---|---:|---:|---:|---:|---:|
| Ridge | 21.14% | 0.793 | -35.37% | 35.45% | 52 |
| RidgeCV | 20.96% | 0.786 | -35.25% | 35.44% | 52 |
| LightGBM | 23.80% | 0.850 | -32.85% | 58.72% | 63 |
| XGBoost | 22.86% | 0.803 | **-27.90%** | 58.10% | 64 |
| **CatBoost** | **25.45%** | **0.912** | -29.33% | 54.02% | 56 |
| RandomForest | 24.37% | 0.810 | -29.37% | 53.52% | 54 |
| LightGBM Optuna | 21.13% | 0.750 | -29.36% | 60.81% | 62 |
| XGBoost Optuna | 18.18% | 0.653 | -38.99% | 60.51% | 64 |
| CatBoost Optuna | 21.24% | 0.753 | -33.23% | 55.51% | 47 |
| RandomForest Optuna | 19.32% | 0.680 | -30.37% | 55.27% | 58 |

30 bps 假设下，默认 CatBoost 的收益和 Sharpe 最高，XGBoost 的最大回撤最低。Ridge 收益略低，但换手显著低于树模型。

## 4. Optuna 结论

四类 Optuna 模型均未超过对应默认参数：

| 模型 | 默认参数年化 | Optuna 年化 | 差值 |
|---|---:|---:|---:|
| LightGBM | 23.80% | 21.13% | -2.67pp |
| XGBoost | 22.86% | 18.18% | -4.68pp |
| CatBoost | 25.45% | 21.24% | -4.20pp |
| RandomForest | 24.37% | 19.32% | -5.04pp |

当前 Optuna 目标最大化验证期日均 Rank IC，没有直接约束 Top 50 组合收益、换手、回撤或锁仓暴露。结果表明该目标与最终组合表现并不完全一致。

## 5. 成本敏感性

| 策略 | 30 bps | 60 bps | 100 bps |
|---|---:|---:|---:|
| Ridge | 21.14% | 14.83% | **6.91%** |
| LightGBM | 23.80% | 13.31% | 0.67% |
| XGBoost | 22.86% | 12.55% | 0.12% |
| CatBoost | **25.45%** | **15.63%** | 3.71% |
| RandomForest | 24.37% | 14.73% | 3.01% |

树模型在 30 bps 下领先，但高换手使优势随成本快速收窄。100 bps 假设下，Ridge 的年化收益最高。

## 6. 版本对比解释

相对 2026-05-27 输出，当前版本同时改变了：

1. 全市场交易日历上的目标与执行收益；
2. 停牌期间的价格盯市；
3. 跌停或停牌时无法卖出的锁仓状态。

因此版本前后的收益变化不是单一“卖出约束效应”，尤其不能把树模型收益上升解释为更严格约束带来的因果改善。

## 7. 验证与输出

- `testing/test_daily_trade_constraints.py`：2 项通过。
- 13 个收益文件均通过必需字段、日期顺序、重复日期和缺失值检查。
- 错误日志为空；全部模型正常完成。

主要输出：

- `output/csi500/daily_alpha/trade_constraint_validation/performance_all_models.csv`
- `output/csi500/daily_alpha/trade_constraint_validation/performance_before_after_all_models.csv`
- `output/csi500/daily_alpha/trade_constraint_validation/cost_sensitivity_all_models.csv`
- `output/csi500/daily_alpha/trade_constraint_validation/*_returns.csv`
- `output/csi500/daily_alpha/trade_constraint_validation/*_optuna_params.csv`

仍然存在静态成分股幸存者偏差、历史 ST / *ST 状态缺失、涨跌停价近似和冲击成本简化等限制。
