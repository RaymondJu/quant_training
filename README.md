# CSI500 日频价量 Alpha：可交易约束版

> 基于静态中证 500 股票池的日频价量研究流程。当前版本重点不是追求更高的回测收益，而是把标签可得性、涨跌停、停牌和调仓执行放到同一套可复现口径中。

![最新策略净值](docs/readme_assets/csi500_daily_latest_nav.svg)

## 核心结论

- 当前四个已按最新交易约束重跑的策略中，`Ridge` 表现最稳健：年化收益 **21.14%**、Sharpe **0.793**。
- 加入停牌和跌停卖出约束后，普通 `Ridge` 年化只下降 **0.26 个百分点**；市值中性版本受影响更明显。
- 执行网格识别出 **20,805** 条次日停牌/无报价记录和 **1,664** 条次日跌停不可卖记录。
- 结果仍受到静态成分股幸存者偏差影响；本地也缺少历史 ST / *ST 状态，因此这仍是研究型回测，不是生产级无偏结果。

---

## 最终披露口径

| 项目 | 当前口径 |
|---|---|
| 股票池 | 2026-05-09 静态中证 500 成分股，499 只股票有可用行情 |
| 原始行情区间 | 2015-01-05 至 2025-12-31 |
| 数据频率 | 日频 OHLCV、成交额、换手率和流通股本 |
| 信号时点 | 第 `t` 日收盘后 |
| 建仓时点 | 第 `t+1` 个市场交易日开盘 |
| 预测标签 | `open[t+6] / open[t+1] - 1`，即未来 5 个交易日开盘到开盘收益 |
| 调仓规则 | 每 5 个市场交易日调仓，Top 50 等权 |
| 交易成本 | 按组合换手扣除 30 bps |
| 模型训练 | Walk-forward，只使用调仓时点已经完整实现的历史标签 |
| 当前主结果 | IC-weight、IC-weight size-neutral、Ridge、Ridge size-neutral |

绩效表使用各策略自身的有效 walk-forward 区间：IC 组合 472 个调仓期，Ridge 组合 383 个调仓期。顶部净值图为了横向可比，统一使用 2018-02-05 至 2025-12-22 的共同区间。

---

## 最新结果

以下是当前版本唯一的主结果表。所有数字均已包含：

- 标签可得性修复；
- 次日涨停、停牌和新股买入过滤；
- 次日跌停或停牌时无法卖出的锁仓约束；
- 30 bps 换手成本。

| 策略 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 平均换手 | 锁仓调仓次数 |
|---|---:|---:|---:|---:|---:|---:|
| IC-weight | 10.19% | 19.62% | 0.519 | -34.55% | 36.32% | 153 |
| IC-weight size-neutral | 8.58% | 19.33% | 0.444 | -33.41% | 37.38% | 153 |
| **Ridge** | **21.14%** | 26.67% | **0.793** | **-35.37%** | **35.45%** | 52 |
| Ridge size-neutral | 11.74% | 25.77% | 0.456 | -40.86% | 43.71% | 56 |

![卖出约束影响](docs/readme_assets/csi500_tradeability_impact.svg)

图中锁仓事件按持仓计数，同一次调仓可能有多只股票被锁，因此事件数可以大于受影响的调仓次数。

### 如何理解结果

1. 普通 Ridge 从 21.39% 降至 21.14%，说明它对新增卖出约束相对稳健。
2. 市值中性策略下降更多，说明它们更容易持有停牌或跌停锁仓标的。
3. 锁仓事件以停牌为主，跌停事件数量较少，但跌停约束在 Ridge 持仓中占比更高。
4. 不再把尚未按最新卖出约束重跑的树模型结果放入主表，避免混用不同版本口径。

---

## 研究流程

```mermaid
flowchart LR
    A["日频 OHLCV"] --> B["价量特征"]
    B --> C["横截面缩尾与标准化"]
    C --> D["Point-in-time 标签与训练窗口"]
    D --> E["IC / Ridge 打分"]
    E --> F["次日买入过滤"]
    F --> G["停牌或跌停持仓锁定"]
    G --> H["剩余名额选 Top 50"]
    H --> I["扣除换手成本"]
```

### 日频特征

| 类别 | 特征 |
|---|---|
| 短期反转 | `REV_1D` |
| 动量 | `MOM_5D`, `MOM_20D`, `MOM_60D` |
| 波动率 | `VOL_20D`, `RANGE_20D` |
| 流动性 | `TURN_5D`, `TURN_20D`, `AMIHUD_20D`, `VOLUME_RATIO_5_20` |
| 价格偏离 | `BIAS_20` |
| 风险与规模 | `SIZE`, `BETA_60D` |
| 增量价量信号 | `VOLUME_PRICE_REVERSAL_20D` |

所有特征按每日横截面做 1% / 99% 缩尾和 z-score 标准化。

### 标签可得性

目标收益使用：

```python
ret_fwd_5d = open[t + 6] / open[t + 1] - 1
```

在调仓日 `T` 收盘做决策时，只允许使用满足以下条件的训练样本：

```text
t + horizon + 1 <= T
```

Optuna 内部训练集和验证集之间额外留出 `horizon` 个交易日 embargo，避免两侧标签区间重叠。

### 交易约束

| 场景 | 当前处理 |
|---|---|
| 次日涨停 | 不允许买入；主板按约 10%，创业板/科创板按约 20% 阈值识别 |
| 次日停牌或无报价 | 不允许买入 |
| 上市不足 120 个有效交易日 | 不允许买入 |
| 已持仓股票次日跌停 | 无法卖出，强制继续持有并占用组合名额 |
| 已持仓股票次日停牌 | 无法卖出，强制继续持有并占用组合名额 |
| 停牌期间估值 | 使用最近可用价格盯市 |
| 剩余组合名额 | 从当期可买股票中按分数补足 |

原始数据会省略大部分停牌日，因此流水线先按全市场交易日历补齐每只股票的执行网格，再判断买入、卖出和持有状态。这样不会再由单股票 `shift()` 直接跨过停牌期。

---

## 版本演进

README 只展示当前最终口径。历史数字和修复过程保留在审计报告中，不再混入主结果表。

| 阶段 | 修复内容 | 文档 |
|---|---|---|
| 第一轮 | 修复训练窗口使用尚不可知未来标签的问题 | [`NO_LEAKAGE_FIX_REPORT.md`](NO_LEAKAGE_FIX_REPORT.md) |
| 第二轮 | 增加 Optuna embargo、次日涨停/停牌和新股买入过滤 | [`LEAKAGE_REVIEW_AND_TRADEABILITY_FIX.md`](LEAKAGE_REVIEW_AND_TRADEABILITY_FIX.md) |
| 当前版本 | 按市场日历处理停牌，并增加跌停/停牌无法卖出的锁仓状态 | [`CSI500_DAILY_TRADE_CONSTRAINT_FIX_REPORT.md`](CSI500_DAILY_TRADE_CONSTRAINT_FIX_REPORT.md) |

仓库中旧的树模型、Optuna 和成本敏感性 CSV 仅用于历史复现。由于它们尚未按当前卖出约束完整重跑，不属于本 README 的最终绩效披露。

---

## 重要局限

### 1. 静态股票池

`data/raw/csi500/index_constituents.csv` 是 2026-05-09 的静态名单，不是历史成分股进出记录。当前回测可能：

- 把后来才进入中证 500 的股票放入更早期样本；
- 漏掉历史上曾经属于中证 500、但当前已被剔除的股票；
- 高估长期收益和相对基准表现。

### 2. 缺少历史 ST / *ST 状态

本地行情没有 `is_st`、历史证券简称或证券状态字段。当前静态成分股名称中没有 ST 股票，但不能据此推断 2015-2025 年的历史状态。

因此当前版本没有伪造 ST 过滤。要补齐该约束，需要按交易日提供：

```text
stock_code, date, is_st
```

### 3. 涨跌停是近似识别

本地没有交易所官方涨停价和跌停价。当前按板块、日期和前收盘价近似判断 10% / 20% 涨跌停，无法覆盖所有特殊证券和价格取整细节。

### 4. 交易成本仍然简化

当前只按换手扣除固定 30 bps，没有单独建模冲击成本、盘口深度、佣金最低收费和卖出印花税变化。

---

## 运行方式

准备好 CSI500 原始行情和基准数据后运行：

```powershell
$env:QT_UNIVERSE = "csi500"
$env:QT_FACTOR_SET = "full"

python csi500_daily_alpha_pipeline.py `
  --horizon 5 `
  --top-n 50 `
  --run-all-ml `
  --run-all-ml-optuna `
  --optuna-trials 12 `
  --optuna-val-days 63 `
  --optuna-retune-every 25 `
  --lgbm-max-train-rows 50000
```

运行最小交易约束测试：

```powershell
python -m unittest testing.test_daily_trade_constraints -v
```

重新生成 README 图表：

```powershell
python analysis/plot_csi500_tradeability_readme.py
```

---

## 关键文件

```text
quant_training/
|-- csi500_daily_alpha_pipeline.py
|-- testing/
|   `-- test_daily_trade_constraints.py
|-- analysis/
|   `-- plot_csi500_tradeability_readme.py
|-- docs/readme_assets/
|   |-- csi500_daily_latest_nav.svg
|   `-- csi500_tradeability_impact.svg
|-- output/csi500/daily_alpha/
|   `-- trade_constraint_validation/
|-- NO_LEAKAGE_FIX_REPORT.md
|-- LEAKAGE_REVIEW_AND_TRADEABILITY_FIX.md
`-- CSI500_DAILY_TRADE_CONSTRAINT_FIX_REPORT.md
```

关键输出：

- `output/csi500/daily_alpha/trade_constraint_validation/performance_before_after.csv`
- `output/csi500/daily_alpha/trade_constraint_validation/constraint_coverage.csv`
- `output/csi500/daily_alpha/trade_constraint_validation/*_returns.csv`

---

## 当前定位

这个分支展示的是一套逐步收紧交易假设后的 CSI500 日频研究框架：

- 收益不再建立在近端标签泄漏上；
- 买入端不再假设涨停或停牌股票可以买到；
- 卖出端不再假设跌停或停牌股票可以立即卖出；
- 对仍缺失的历史成分股和 ST 数据明确披露，而不是用不可靠代理掩盖。

当前最可信的结论不是“某个模型能稳定获得 20% 以上收益”，而是：在这份带有静态股票池偏差的数据上，普通 Ridge 对逐步收紧的交易约束相对稳健，但结果仍需在历史动态成分股和 ST 状态补齐后重新验证。
