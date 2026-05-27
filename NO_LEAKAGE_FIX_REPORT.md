# CSI500 Daily Alpha 信息泄露修复报告

生成日期：2026-05-26  
项目分支：`csi500-daily-alpha`  
主要文件：`csi500_daily_alpha_pipeline.py`

## 1. 背景

同学指出 CSI500 日频策略收益异常，可能存在：

- 回测未来函数
- 因子未来函数
- 买入不可交易股票
- 原始 parquet 数据异常

本轮重点先修复已经确认的回测标签可得性泄露，并重跑完整日频 pipeline。

## 2. 已确认的核心缺陷

日频标签定义位于 `csi500_daily_alpha_pipeline.py`：

```python
g[f"ret_fwd_{horizon}d"] = g["open"].shift(-(horizon + 1)) / g["open"].shift(-1) - 1.0
```

含义是：在信号日 `t` 收盘后选股，假设下一交易日开盘 `open[t+1]` 买入，持有到 `open[t+horizon+1]` 卖出。

因此，对样本日 `t` 来说，`ret_fwd_horizon` 只有在 `t + horizon + 1` 这个交易日开盘之后才可知。原代码训练窗口只要求：

```python
train_date < rebalance_date
```

这会把调仓日前最近 `horizon` 个交易日的未来收益标签放入训练集或 IC 历史。默认 `horizon=5` 时，每次调仓都会混入最近 5 个尚不可知的标签。

## 3. 修复内容

新增统一 helper：

```python
def _known_label_dates(
    dates: list[pd.Timestamp],
    date_pos: int,
    horizon: int,
    lookback_days: int | None = None,
) -> list[pd.Timestamp]:
    """
    Return dates whose forward-return labels are observable by rebalance time.

    ret_fwd_h uses open[t+1] through open[t+h+1]. At the close of rebalance
    date T, labels are known only for sample dates t where t+h+1 <= T.
    """
    end_pos = max(0, date_pos - horizon)
    start_pos = 0 if lookback_days is None else max(0, end_pos - lookback_days)
    return dates[start_pos:end_pos]
```

修复后的逻辑：在调仓日 `T` 做决策时，只允许使用满足：

```text
t + horizon + 1 <= T
```

的历史样本。换成 Python 切片后，最后一个可用训练信号日是：

```text
dates[date_pos - horizon - 1]
```

对应实现是 `dates[start_pos : date_pos - horizon]`。

### 已替换的位置

以下函数已改为使用 `_known_label_dates()`：

- `backtest_ic_weight`
- `backtest_ridge`
- `backtest_ridge_cv`
- `backtest_lightgbm`
- `backtest_model`
- `backtest_lightgbm_optuna`
- `backtest_model_optuna`

IC-weight 原来使用：

```python
hist = ic_indexed.loc[ic_indexed.index < date, FEATURE_COLS].tail(ic_window)
```

现改为：

```python
known_dates = _known_label_dates(dates, date_pos, horizon, ic_window)
hist = ic_indexed.reindex(known_dates)[FEATURE_COLS].dropna(how="all")
```

各 ML 模型原来使用：

```python
train_dates = dates[max(0, date_pos - train_days):date_pos]
```

现改为：

```python
train_dates = _known_label_dates(dates, date_pos, horizon, train_days)
```

## 4. 修复验证

### 编译检查

```powershell
python -m py_compile csi500_daily_alpha_pipeline.py
```

结果：通过。

### 标签可得性检查

使用现有 `data/processed/csi500/daily_alpha/daily_alpha_panel.parquet` 检查默认参数：

- `horizon=5`
- `train_days=504`
- `start_days=756`

检查结果：

```text
rebalance_count = 383
violations = 0
min_gap = 6
```

说明 383 次 ML 调仓中，没有任何训练样本违反 `t + horizon + 1 <= T`。最近训练样本距离调仓日最小为 6 个交易日，符合 `open[t+6] / open[t+1]` 标签定义。

## 5. 原始 parquet 抽查结论

`data/processed/csi500/daily_prices.parquet` 抽查：

- 行数：`1,091,252`
- 股票数：`499`
- 日期范围：`2015-01-05 ~ 2025-12-31`
- `stock_code-date` 重复：`0`
- 非正 open/close：`0`
- OHLC 关系异常：`0`
- `pct_change` 与 close 复算最大误差：`0`

`data/processed/csi500/daily_alpha/daily_alpha_panel.parquet` 抽查：

- 行数：`1,088,258`
- 日期范围：`2015-01-05 ~ 2025-12-23`
- 股票数：`499`
- `ret_fwd_5d` 与原始 open 价复算最大误差：`0`

结论：本轮未发现原始 OHLCV parquet 明显坏表。问题主要来自回测使用未来标签，而不是收益标签计算或原始行情本身错。

## 6. 修复后完整重跑

运行命令等价于 `run_csi500_daily_alpha_pipeline.ps1`：

```powershell
$env:QT_UNIVERSE='csi500'
$env:QT_FACTOR_SET='full'
python -u csi500_daily_alpha_pipeline.py `
  --horizon 5 `
  --top-n 50 `
  --run-all-ml `
  --run-all-ml-optuna `
  --optuna-trials 12 `
  --optuna-val-days 63 `
  --optuna-retune-every 25 `
  --lgbm-max-train-rows 50000
```

输出文件：

- `output/csi500/daily_alpha/performance_summary.csv`
- `output/csi500/daily_alpha/cost_sensitivity.csv`
- 日志：`output/csi500/daily_alpha/run_logs/rerun_no_leak_sync_20260526_195948.log`

## 7. 修复前后绩效变化

| 策略 | 修复前年化 | 修复后年化 | 下降 | 相对下降 |
|---|---:|---:|---:|---:|
| IC-weight | 15.05% | 10.50% | 4.55pp | 30.22% |
| IC-weight size-neutral | 13.75% | 10.53% | 3.22pp | 23.42% |
| Ridge | 36.47% | 29.03% | 7.44pp | 20.39% |
| Ridge size-neutral | 26.87% | 20.79% | 6.08pp | 22.62% |
| RidgeCV | 36.60% | 28.81% | 7.78pp | 21.27% |
| LightGBM | 34.66% | 26.62% | 8.03pp | 23.18% |
| XGBoost | 42.04% | 26.20% | 15.84pp | 37.67% |
| CatBoost | 42.80% | 28.10% | 14.70pp | 34.34% |
| RandomForest | 34.71% | 26.83% | 7.88pp | 22.71% |
| LightGBM Optuna | 38.79% | 22.46% | 16.33pp | 42.10% |
| XGBoost Optuna | 46.68% | 25.74% | 20.94pp | 44.85% |
| CatBoost Optuna | 45.46% | 28.18% | 17.28pp | 38.01% |
| RandomForest Optuna | 39.46% | 26.11% | 13.35pp | 33.82% |

整体结论：修复后年化收益平均下降约 11 个百分点，相对下降约 30%。树模型和 Optuna 模型下降更明显，说明它们此前更容易利用近端泄露标签。

## 8. 修复后统计表

| 策略 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 胜率 | 期数 | 平均换手 | 基准年化 | 超额年化 | IR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IC-weight | 10.50% | 20.05% | 0.524 | -35.71% | 56.14% | 472 | 36.69% | 1.67% | 7.76% | 0.730 |
| IC-weight size-neutral | 10.53% | 19.92% | 0.529 | -33.05% | 56.14% | 472 | 37.11% | 1.67% | 7.79% | 0.749 |
| Ridge | 29.03% | 26.79% | 1.084 | -33.63% | 58.22% | 383 | 35.60% | 2.86% | 25.56% | 2.101 |
| Ridge size-neutral | 20.79% | 25.99% | 0.800 | -35.73% | 55.87% | 383 | 43.82% | 2.86% | 17.36% | 1.472 |
| RidgeCV | 28.81% | 26.80% | 1.075 | -33.90% | 57.96% | 383 | 35.56% | 2.86% | 25.34% | 2.081 |
| LightGBM | 26.62% | 29.36% | 0.907 | -29.81% | 56.14% | 383 | 59.26% | 2.86% | 23.82% | 1.759 |
| XGBoost | 26.20% | 29.10% | 0.900 | -31.39% | 57.44% | 383 | 58.94% | 2.86% | 23.37% | 1.741 |
| CatBoost | 28.10% | 29.06% | 0.967 | -30.38% | 58.22% | 383 | 54.97% | 2.86% | 25.12% | 1.803 |
| RandomForest | 26.83% | 29.23% | 0.918 | -28.84% | 57.44% | 383 | 53.26% | 2.86% | 23.95% | 1.723 |
| LightGBM Optuna | 22.46% | 28.59% | 0.786 | -41.43% | 57.18% | 383 | 60.43% | 2.86% | 19.66% | 1.533 |
| XGBoost Optuna | 25.74% | 28.77% | 0.895 | -32.98% | 56.92% | 383 | 60.89% | 2.86% | 22.92% | 1.821 |
| CatBoost Optuna | 28.18% | 28.38% | 0.993 | -30.02% | 58.22% | 383 | 55.67% | 2.86% | 25.13% | 1.929 |
| RandomForest Optuna | 26.11% | 29.12% | 0.897 | -32.58% | 55.87% | 383 | 53.23% | 2.86% | 23.26% | 1.700 |

## 9. 成本敏感性

当前默认成本是 `30 bps`。脚本也输出了 `60 bps`、`100 bps` 情景：

| 策略 | 30bps 年化 | 60bps 年化 | 100bps 年化 |
|---|---:|---:|---:|
| IC-weight | 10.50% | 4.54% | -2.91% |
| IC-weight size-neutral | 10.53% | 4.50% | -3.04% |
| Ridge | 29.03% | 22.30% | 13.84% |
| Ridge size-neutral | 20.79% | 13.06% | 3.50% |
| RidgeCV | 28.81% | 22.10% | 13.67% |
| LightGBM | 26.62% | 15.81% | 2.78% |
| XGBoost | 26.20% | 15.48% | 2.55% |
| CatBoost | 28.10% | 17.92% | 5.58% |
| RandomForest | 26.83% | 17.05% | 5.15% |
| LightGBM Optuna | 22.46% | 11.80% | -1.02% |
| XGBoost Optuna | 25.74% | 14.72% | 1.47% |
| CatBoost Optuna | 28.18% | 17.87% | 5.38% |
| RandomForest Optuna | 26.11% | 16.39% | 4.56% |

结论：Ridge/RidgeCV 的成本承受能力最好；树模型对成本更敏感；IC-weight 在 100bps 情景下转负。

## 10. 仍未解决的风险

本轮只修复了日频回测的标签可得性泄露。以下问题仍需要下一轮审查：

### 10.1 CSI500 历史成分股/幸存者偏差

`data/raw/csi500/index_constituents.csv` 中 500 只股票的 `date` 全部是 `2026-05-09`，不是历史入选日期。当前数据更像“当前 CSI500 成分股的历史行情”，不是真实历史 CSI500 成分池。

影响：

- 可能包含历史上尚未进入 CSI500 的股票
- 可能遗漏历史上曾经属于 CSI500、但当前已剔除的股票
- 策略收益仍可能带有幸存者偏差或当前成分股偏差

相关代码：

- `data/download.py::_normalize_constituents`
- `data/universe.py`

### 10.2 可交易性过滤不完整

已抽样检查日频持仓：

- 下一交易日零成交：未发现
- 下一交易日开盘涨幅 `>=9.5%`：Ridge 约 0.72% 持仓样本
- 上市不足 120 个交易日：Ridge 约 4.41% 持仓样本

仍缺：

- ST / *ST 过滤
- 涨停买不进、跌停卖不出约束
- 停牌日更严格处理
- 新股上市天数过滤在日频 pipeline 中未统一执行

### 10.3 当前成本模型较粗

当前日频成本为：

```python
net_ret = gross_ret - (cost_bps / 10_000.0) * turnover
```

需要进一步确认：

- `cost_bps` 是否代表单边还是双边
- 与月频回测 `cost = turnover * 2 * transaction_cost` 的口径是否一致
- 高频 5 日调仓是否需要额外滑点或冲击成本

### 10.4 因子是否应整体延迟一天

当前日频因子多数使用当日收盘、最高、最低、成交量等信息，并假设信号在当日收盘后可得，下一交易日开盘买入。这在设定上可以成立。

但下一轮仍建议确认：

- 是否所有字段在收盘后可得
- 是否存在 AKShare 后复权价格在历史时点不可得的问题
- 是否需要用前复权或不复权价格重建交易收益

### 10.5 月频链路还需单独复查

本轮主要修复 `csi500_daily_alpha_pipeline.py`。月频目录仍需独立审查：

- `portfolio/backtest.py`
- `portfolio/combine.py`
- `ml/model_comparison.py`
- `factors/*`
- `data/clean.py`

尤其需要确认月频 `ret_next_month`、财务数据 `NOTICE_DATE`、调仓日与收益归属月份是否严格 point-in-time。

## 11. 建议 Claude Code 下一轮重点

建议按以下顺序继续查：

1. 对 `csi500_daily_alpha_pipeline.py` 再做一次代码审计，确认所有训练、调参、IC 计算都没有使用尚不可知标签。
2. 给 `_known_label_dates()` 加单元测试或最小数据集测试。
3. 给日频 pipeline 加可交易性过滤：
   - 上市满 120 个交易日
   - 下一交易日开盘涨停过滤或无法成交处理
   - ST / *ST 过滤
   - 停牌过滤
4. 重建真实 CSI500 历史成分股池，或在报告中明确当前结果只是“当前 CSI500 成分股历史回测”。
5. 统一日频和月频成本口径，明确单边/双边含义。
6. 单独审查月频回测链路是否有类似标签可得性问题。

