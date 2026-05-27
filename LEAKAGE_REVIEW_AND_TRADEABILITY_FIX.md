# CSI500 Daily Alpha 第二轮复查与可交易性修复报告

生成日期：2026-05-27
项目分支：`csi500-daily-alpha` → 本轮在其上派生修复分支
主要文件：`csi500_daily_alpha_pipeline.py`
关联文档：`NO_LEAKAGE_FIX_REPORT.md`（第一轮，标签可得性泄露修复）

## 1. 本轮目标

第一轮（`NO_LEAKAGE_FIX_REPORT.md`）修复了日频回测的标签可得性泄露。本轮工作分两部分：

1. **复查**第一轮修复是否正确、是否覆盖全部代码路径、是否存在遗漏的泄露点。
2. 在确认无误的基础上，补两项第一轮未处理的问题：**Optuna 调参的 train/val 标签重叠**与**回测缺少可交易性过滤**，并重跑完整 pipeline。

## 2. 对第一轮修复的复查结论

逐项核对，第一轮修复**正确且完整**：

| 复查项 | 结论 |
|---|---|
| 泄露诊断（训练用 `t < T` 而非 `t + h + 1 <= T`） | 诊断正确，确为真实泄露 |
| `_known_label_dates` 切片逻辑 | `end_pos = date_pos - horizon`，最后一个可用信号日的标签卖出日正好落在调仓日 T 的开盘，收盘时已知，既不泄露又最大化用数据 |
| 修复覆盖范围 | 7 个调用点（IC-weight + Ridge + RidgeCV + LightGBM + 通用 model + LightGBM Optuna + 通用 model Optuna）全部使用 `_known_label_dates` |
| 是否有遗漏的旧式写法 | 检索 `:date_pos`、`index <`、`tail(` 等模式，零残留 |
| 特征构造 | 全部 point-in-time（backward 滚动窗口、`shift` 取过去值；横截面 winsorize/zscore 按 `groupby("date")` 计算，仅用当日截面，不泄露） |
| `min_gap = 6 / violations = 0` 自检 | 与独立逻辑推导一致 |

补充确认第一轮报告 10.4 节留作待定的问题：**后复权价格不存在时点泄露**。后复权以 IPO 为锚向后累乘，date t 的复权因子只取决于 t 及之前的分红，未来分红不改变 t 及之前的值，因此用后复权计算收益是 point-in-time 安全的。这也是回测应使用后复权而非前复权的原因。

## 3. 本轮新增修复

### 3.1 Optuna train/val embargo

**问题**：`tune_lightgbm_params` 与 `tune_model_params` 在划分调参用的训练/验证集时，直接 `train_dates = dates[:-val_days]`、`val_dates = dates[-val_days:]`，两段相邻无间隔。训练集尾部 `horizon` 天的前向收益标签（`ret_fwd_h` 使用 `open[t+1]..open[t+h+1]`）会落入验证期，造成 train/val 标签重叠，使验证 IC 偏乐观、调参轻微过拟合。

注意：该问题不会把调仓日 T 之后的信息灌入回测收益（训练与验证都在已知标签窗口内），只影响超参选择质量，属次要瑕疵，但应修正。

**修复**：两个调参函数新增 `horizon` 参数，验证集前丢弃 `horizon` 天训练样本：

```python
val_dates = dates[-val_days:]
train_dates = dates[: -(val_days + horizon)] if horizon > 0 else dates[:-val_days]
```

守卫条件同步改为 `len(dates) <= val_days + horizon + 30`。两个 Optuna 回测函数在调用处传入 `horizon=horizon`。

### 3.2 可交易性过滤

**问题**：原回测每期直接 `nlargest(top_n, "score")`，未判断标的在建仓时点（次日开盘）是否可成交。第一轮报告 10.2 已抽样发现 Ridge 持仓中约 0.72% 在次日开盘接近涨停。

**修复**：在 `build_daily_panel` 中为每条记录新增 `tradeable` 标志，刻画"在 t 日收盘决策、t+1 日开盘建仓"是否可行：

- **次日开盘未涨停**：`open[t+1] / close[t] - 1 < limit`。limit 按板块与日期区分：主板 ±10%；创业板（300）/科创板（688）±20%，创业板自 2020-08-24 起放宽。
- **次日未停牌**：`volume[t+1] > 0`。
- **上市满 120 个交易日**：剔除次新股。

新增 `_select_top_n()` helper，先按 `tradeable` 过滤再取 Top-N，替换全部 7 个选股点。

**过滤强度**（全样本 1,088,258 条）：可交易 94.33%，被过滤 5.67%（其中次日开盘涨停买不进 0.33%，其余为上市不足 120 日 / 停牌 / 末行无次日数据）。比例合理，未过度过滤。

## 4. 编译与一致性验证

- `python -m py_compile csi500_daily_alpha_pipeline.py`：通过。
- `_select_top_n` 单元验证：在含 `tradeable` 标志的样例上正确跳过不可交易标的。
- 复现一致性：在同一份面板上、删除 `tradeable` 列（helper 退回原 `nlargest`）跑出的"过滤前"数字，与第一轮报告修复后的数字完全一致（IC-weight 10.50%、Ridge 29.03%），证明本轮复现忠实。
- RandomForest Optuna 在重跑中顺利完成（此前一次中断系机器休眠所致，非代码死锁）。

## 5. 修复前后绩效对比

下表"修复前"为第一轮（仅修复标签泄露）的年化收益；"修复后"为本轮叠加 embargo + 可交易性过滤后的结果。成本 30 bps，区间 2015–2025，5 日调仓。

| 策略 | 修复前年化 | 修复后年化 | 变化 | 修复后 Sharpe | 修复后 IR |
|---|---:|---:|---:|---:|---:|
| IC-weight | 10.50% | 10.31% | -0.2pp | 0.521 | 0.707 |
| IC-weight size-neutral | 10.53% | 10.03% | -0.5pp | 0.509 | 0.698 |
| Ridge | 29.03% | 21.39% | -7.6pp | 0.801 | 1.555 |
| Ridge size-neutral | 26.87% | 12.81% | -14.1pp | 0.497 | 0.851 |
| RidgeCV | 28.81% | 21.30% | -7.5pp | 0.798 | 1.547 |
| LightGBM | 26.62% | 21.07% | -5.6pp | 0.729 | 1.442 |
| XGBoost | 26.20% | 19.17% | -7.0pp | 0.668 | 1.290 |
| CatBoost | 28.10% | 20.78% | -7.3pp | 0.720 | 1.342 |
| RandomForest | 26.83% | 19.78% | -7.0pp | 0.693 | 1.311 |
| LightGBM Optuna | 22.46% | 17.96% | -4.5pp | 0.640 | 1.287 |
| XGBoost Optuna | 25.74% | 20.22% | -5.5pp | 0.707 | 1.434 |
| CatBoost Optuna | 28.18% | 20.80% | -7.4pp | 0.733 | 1.467 |
| RandomForest Optuna | 26.11% | 19.29% | -6.8pp | 0.680 | 1.295 |

观察：

1. 可交易性过滤普遍砍掉 5–7pp 年化，几乎所有模型都受影响，说明此前确实计入了次日开盘涨停、实际买不进的标的。
2. **IC-weight 几乎不变（-0.2pp）**：平滑线性合成不集中押注极端涨停股，符合预期。
3. **Ridge size-neutral 下降最大（-14.1pp）**：size-neutral 将打分向小盘倾斜后，严重依赖不可成交的小盘涨停股；过滤后收益与 Sharpe 大幅回落，提示该"卖点"此前主要由不可交易标的支撑。
4. 非 Optuna 模型的下降全部来自可交易性过滤；Optuna 模型为 embargo + 可交易性过滤的合并效果。

## 6. 仍未解决的风险

### 6.1 幸存者偏差（当前最大剩余水分）

`data/raw/csi500/index_constituents.csv` 中 500 只成分股的 `date` 全部为 `2026-05-09`，是当前成分股名单，非历史时点成分。当前回测等价于"用当前 CSI500 成分股回溯 2015–2025"，隐含幸存者偏差：

- 作为对照，等权买入全部当前 499 只成分（完全不选股）年化约 16.24%，而真实 CSI500 基准年化仅约 2.86%。两者之差主要来自幸存者偏差与等权小盘暴露。
- 因此修复后表中较高的 IR（如 Ridge 1.55）仍被高估——其分母基准（2.86%）偏低。衡量真实选股 alpha 应以"等权全样本（约 16%）"为基准，据此 Ridge 的边际 alpha 约 5pp，且仍建立在有偏 universe 之上。

根治需要真实历史时点成分股数据（含纳入与剔除记录），当前免费数据源不提供。在获得该数据前，对外应明确标注"结果为当前成分股回测、偏乐观"。

### 6.2 可交易性过滤仍不完整

已覆盖：次日开盘涨停、次日停牌、上市天数。仍缺：ST/*ST 过滤（当前成分股名单无 ST 标记，缺历史 ST 状态）、跌停卖不出约束、停牌期间的更严格处理。

### 6.3 成本口径

日频成本为 `gross_ret - (cost_bps/10000) * turnover`。仍需确认 `cost_bps` 单边/双边口径、与月频 `cost = turnover * 2 * transaction_cost` 是否一致，以及 5 日调仓是否需额外滑点/冲击成本。

## 7. 本轮改动文件清单

- `csi500_daily_alpha_pipeline.py`
  - 新增可交易性常量（`MAIN_BOARD_LIMIT` / `WIDE_BOARD_LIMIT` / `CHINEXT_WIDE_LIMIT_DATE` / `MIN_LIST_DAYS`）
  - `build_daily_panel` 计算并输出 `tradeable` 列
  - 新增 `_select_top_n()` 并替换 7 处选股逻辑
  - `tune_lightgbm_params` / `tune_model_params` 新增 `horizon` 参数与 embargo，两处 Optuna 调用传参
- `output/csi500/daily_alpha/*`：用新代码重跑后的全部官方输出

## 8. 后续建议顺序

1. 重建真实 CSI500 历史时点成分股池，或在所有对外材料中标注当前结果为"当前成分股回测"。
2. 补 ST/*ST 与跌停卖出约束，完善可交易性过滤。
3. 统一日频与月频成本口径，明确单边/双边。
4. 对月频链路（`portfolio/`、`ml/`、`factors/`、`data/clean.py`）做同口径的标签可得性与 point-in-time 复查。
