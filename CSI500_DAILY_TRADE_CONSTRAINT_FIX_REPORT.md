# CSI500 日频交易约束补充修复报告

生成日期：2026-06-07
目标文件：`csi500_daily_alpha_pipeline.py`
承接报告：`NO_LEAKAGE_FIX_REPORT.md`、`LEAKAGE_REVIEW_AND_TRADEABILITY_FIX.md`

## 1. 本轮范围

本轮不重复标签泄漏排查，只处理上一轮报告遗留的三项交易可行性问题：

1. ST / *ST 过滤
2. 跌停无法卖出
3. 更严格的停牌处理

## 2. 本地数据字段审计

### 2.1 历史 ST / *ST：缺失，未伪造

`data/processed/csi500/daily_prices.parquet` 只有：

`date, open, high, low, close, volume, turnover, outstanding_share, turnover_rate, stock_code, pct_change`

本地没有历史 `is_st`、证券简称、交易状态或官方涨跌停价字段。
`data/raw/csi500/index_constituents.csv` 虽有 `stock_name`，但全部记录日期都是
2026-05-09，只是当前静态成分股名称，不能用于 2015-2025 年历史 ST 判断。
当前静态名单中也没有名称匹配 ST / *ST 的股票。

因此本轮没有加入不可靠的历史 ST 过滤。要补齐该项，至少需要按交易日提供
`stock_code, date, is_st`，或可还原历史简称的证券状态表。

### 2.2 停牌与涨跌停：可从 OHLCV 推导

行情共有 499 只股票、2,674 个市场交易日。逐股票首末日期之间共有 20,803 个
缺失交易日，另有 2 条零成交记录。原实现按单只股票直接 `shift()`，会跳过这些
缺失日，把停牌期间错误地压缩掉。

本轮使用全市场交易日历重建执行网格，并基于开盘价、前一可用收盘价、成交量和
板块涨跌幅阈值推导开盘可买/可卖状态。

## 3. 实现

- 标签收益改为严格使用全市场日历上的 `open[t+1]` 和 `open[t+h+1]`，不再跨过
  缺失停牌日。
- 次日无报价或零成交时禁止买入；原有新股 120 个交易日过滤继续保留。
- 次日开盘跌幅达到对应板块阈值时，已有持仓禁止卖出并强制锁仓。
- 次日停牌或无报价时，已有持仓同样强制锁仓。
- 锁仓占用组合名额，只用剩余名额选择新的可买股票。
- 停牌期间使用最近可用价格进行组合盯市，不把缺失日当作已经完成交易。
- IC、Ridge、RidgeCV、LightGBM、XGBoost、CatBoost、RandomForest 及 Optuna
  路径统一走同一个执行约束 helper。
- 绩效输出新增平均锁仓数、发生锁仓的调仓次数、跌停锁仓数和停牌锁仓数。

## 4. 最小验证

- AST 语法解析：通过。
- `testing/test_daily_trade_constraints.py`：覆盖跌停锁仓和停牌锁仓，2 项通过。
- 新执行面板：1,109,061 行，2,668 个有效信号日，499 只股票。
- 次日停牌/无报价标记：20,805 行。
- 次日跌停不可卖标记：1,664 行。
- 可买样本比例：92.54%。

## 5. 修复前后绩效

参数沿用上一轮基线：2015-2025、5 日调仓、Top 50、成本 30 bps。

| 策略 | 修复前年化 | 修复后年化 | 变化 | 修复前 Sharpe | 修复后 Sharpe | 锁仓调仓次数 |
|---|---:|---:|---:|---:|---:|---:|
| IC-weight | 10.31% | 10.19% | -0.13pp | 0.521 | 0.519 | 153 |
| IC-weight size-neutral | 10.03% | 8.58% | -1.45pp | 0.509 | 0.444 | 153 |
| Ridge | 21.39% | 21.14% | -0.26pp | 0.801 | 0.793 | 52 |
| Ridge size-neutral | 12.81% | 11.74% | -1.07pp | 0.497 | 0.456 | 56 |

普通 IC-weight 和 Ridge 受影响较小；规模中性版本下降更明显，说明其持仓更容易
落入停牌或跌停锁仓状态。本轮只跑了代表性的四条基线路径，没有重跑全部树模型和
Optuna 组合。

明细输出：

- `output/csi500/daily_alpha/trade_constraint_validation/performance_before_after.csv`
- `output/csi500/daily_alpha/trade_constraint_validation/constraint_coverage.csv`
- `data/processed/csi500/daily_alpha/daily_alpha_panel_trade_constraints.parquet`

## 6. 仍存在的数据缺口

1. 历史 ST / *ST 状态缺失，尚不能可靠过滤。
2. 没有交易所官方涨跌停价，本轮沿用板块和日期阈值近似识别。
3. 当前仍是 2026-05-09 静态 CSI500 成分股回溯，历史成分股偏差不在本轮范围。
