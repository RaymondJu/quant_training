# 开发记录

## 当前最终状态

本文档记录代码修复和最近一次完整重跑之后，项目当前被接受的正式状态。

### 项目口径

- 股票池：静态沪深 300 成分股名单
- 基准：510300 ETF 累计净值 proxy
- 传统多因子区间：`2015-07 ~ 2025-11`
- 机器学习区间：`2017-10 ~ 2025-11`

### 为什么冻结这个口径

项目曾尝试重建沪深 300 历史成分股进入和剔除路径，但这个流程没有稳定到足够可靠的程度。与其继续展示一个半修复的动态股票池版本，不如保留当前静态股票池版本，并明确承认幸存者偏差。

### 已修复问题

- 动量因子 `MOM_12_1` 的时间定义已修正。
- `standardize()` 已改为使用 `ddof=0`。
- `clean.py` 已说明估算公告日带来的近似 PIT 风险。
- 机器学习换手率计算已与主回测定义统一。
- 基准已改为 510300 累计净值 proxy，不再使用旧版固定股息近似。

### 最新核心结果

| 策略 | 年化收益 | Sharpe | 最大回撤 |
|---|---:|---:|---:|
| Equal-weight | 16.99% | 0.768 | -28.04% |
| IC-weight | 17.10% | 0.785 | -26.69% |
| ICIR-weight | 17.90% | 0.813 | -26.63% |
| XGBoost | 19.94% | 0.942 | -25.53% |
| RandomForest | 20.01% | 0.926 | -22.94% |
| Ridge | 19.89% | 0.994 | -21.64% |
| Benchmark | 3.19% | 0.184 | -33.24% |

### 当前解释

- 传统 baseline 最优方案：`ICIR-weight`
- 机器学习收益最高模型：`RandomForest`
- 机器学习稳定性最优模型：`Ridge`
- TopRisk 主要作为回撤控制层发挥作用。
- TopRisk v2 IC 加权版本弱于 v1 等权版本。

### 当前可信文件

- `README.md`
- `PROJECT_SUMMARY_FOR_INTERVIEW.md`
- `docs/TOP_RISK_FILTER.md`
- `output/analysis/performance_table.csv`
- `output/ml/model_comparison.csv`
- `output/ablation/performance_comparison.csv`
- `output/ablation/v2_performance.csv`

如果其他文档与上述内容矛盾，应优先认为旧文档已经过期。
