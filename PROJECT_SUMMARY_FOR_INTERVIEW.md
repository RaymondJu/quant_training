# 项目总结

## 一句话版本

这是一个以 **静态 CSI300 股票池** 为样本的 A 股月频多因子选股研究项目，覆盖了数据清洗、因子构建、因子检验、传统多因子组合、ML 横向比较、TopRisk 风控过滤和消融实验。

---

## 当前正式口径

- 股票池：**静态 CSI300 成分股名单**
- 频率：月频
- baseline 区间：`2015-07 ~ 2025-11`
- ML 区间：`2017-10 ~ 2025-11`
- benchmark：`510300 ETF` 累计净值 proxy

最重要的披露：

> 这不是一个完全无偏的历史成分股回测。由于历史成分股纳入和剔除路径没有被可靠重建，当前版本保留静态 CSI300 样本池，因此存在幸存者偏差。

这个说法必须始终保持一致，不要再把当前版本表述成“动态 CSI300 / partial fix”。

---

## 你做了什么

### 1. 因子构建

构建了 16 个 alpha 因子，分成 5 类：

- 估值：`EP`, `BP`, `SP`
- 动量/反转：`MOM_12_1`, `REV_1M`
- 质量/成长：`ROE_TTM`, `GPM_change`, `OCF_QUALITY`, `ASSET_GROWTH`
- 波动/风险：`VOL_20D`, `IVOL`, `BETA_60D`
- 流动性/交易：`TURN_1M`, `ABTURN_1M`, `AMIHUD`, `SIZE`

另外构建了 4 个仅用于风控过滤层的风险因子：

- `BIAS_20`
- `UPSHADOW_20`
- `VOL_SPIKE_6M`
- `RET_6M`

### 2. 因子检验

用了三类传统量化检验：

- Rank IC / ICIR
- 分层回测
- Fama-MacBeth 横截面回归

### 3. 组合与 ML

传统多因子组合实现了：

- Equal-weight
- IC-weight
- ICIR-weight

ML 横比实现了：

- LightGBM
- CatBoost
- XGBoost
- RandomForest
- Ridge

并且 ML 部分使用严格 walk-forward：

```text
训练 24 个月 -> 验证 3 个月 -> 预测下一期
```

### 4. 风控层

在 Top-N 选股之后增加了 TopRisk 过滤层：

- 先选 Top 50
- 再剔除风险分最高的 20%
- 剩余股票等权持有

同时做了 v1 等权和 v2 IC 加权两个版本的风险分合成。

---

## 最新结果

数据来源：

- `output/analysis/performance_table.csv`
- `output/ml/model_comparison.csv`
- `output/ablation/performance_comparison.csv`
- `output/ablation/v2_performance.csv`

### 传统多因子

| 策略 | 年化收益 | Sharpe | 最大回撤 | 超额年化 | IR |
|---|---:|---:|---:|---:|---:|
| Equal-weight | 16.99% | 0.768 | -28.04% | 13.70% | 1.129 |
| IC-weight | 17.10% | 0.785 | -26.69% | 13.77% | 1.189 |
| **ICIR-weight** | **17.90%** | **0.813** | **-26.63%** | **14.56%** | **1.223** |
| Benchmark | 3.19% | 0.184 | -33.24% | -- | -- |

结论：

- 传统多因子里，`ICIR-weight` 是当前版本的最优 baseline。

### ML 模型

| 模型 | 年化收益 | Sharpe | 最大回撤 | 超额年化 | IR |
|---|---:|---:|---:|---:|---:|
| LightGBM | 13.22% | 0.621 | -22.94% | 10.14% | 0.920 |
| CatBoost | 15.34% | 0.698 | -28.22% | 12.35% | 1.111 |
| XGBoost | 19.94% | 0.942 | -25.53% | 16.62% | 1.440 |
| **RandomForest** | **20.01%** | 0.926 | -22.94% | **16.68%** | 1.354 |
| **Ridge** | 19.89% | **0.994** | **-21.64%** | 16.49% | **1.657** |

结论：

- 看绝对收益：`RandomForest` 第一
- 看风险调整和稳定性：`Ridge` 第一
- 当前静态 CSI300 版本下，`RandomForest / XGBoost / Ridge` 都已经跑赢传统 baseline 的共同窗口

### TopRisk Ablation

| 策略 | 年化收益 | Sharpe | 最大回撤 | Calmar |
|---|---:|---:|---:|---:|
| ICIR (no filter) | 18.96% | 0.940 | -24.01% | 0.790 |
| ICIR + TopRisk | 18.86% | 0.950 | -23.22% | 0.812 |
| Ridge (no filter) | 19.89% | 0.994 | -21.64% | 0.919 |
| Ridge + TopRisk | 18.15% | 0.905 | -19.81% | 0.917 |
| XGBoost (no filter) | 19.94% | 0.942 | -25.53% | 0.781 |
| XGBoost + TopRisk | 18.18% | 0.879 | -23.90% | 0.761 |

结论：

- TopRisk 现在更像“降回撤层”，不是“提收益层”
- 对 ICIR 有一定风险收益改善
- 对 Ridge / XGBoost 则更多是用收益换回撤

### v1 vs v2 风控加权

| 策略 | 年化收益 | Sharpe | 最大回撤 | Calmar |
|---|---:|---:|---:|---:|
| ICIR + TopRisk v1 equal | 18.86% | 0.950 | -23.22% | 0.812 |
| ICIR + TopRisk v2 IC-weighted | 16.39% | 0.802 | -24.36% | 0.673 |

结论：

- v2 没有优于 v1
- 当前正式版本保留 `v1 equal`

---

## 技术上值得讲的点

### 1. benchmark 修正

项目最早 benchmark 口径偏低。当前已经改成：

- 优先真实全收益指数
- 否则使用 `510300 ETF` 累计净值 proxy
- 最后才 fallback 到价格指数 + 固定股息近似

这比“固定 2% 股息率”更专业，也更适合解释给面试官。

### 2. 这几个问题已经修掉

- `MOM_12_1` 定义修正
- `standardize()` 使用 `ddof=0`
- `clean.py` 明确说明估算 `announce_date` 的局限
- ML 换手率口径和主回测统一
- benchmark 低估问题修正

### 3. 幸存者偏差要主动承认

这是当前版本最重要的边界条件：

> 我没有把这个项目包装成“完全无偏的历史成分股回测”，而是明确保留静态 CSI300 版本，并披露幸存者偏差。

这比装作没有问题更专业。

---

## 面试时的推荐讲法

### 30 秒版

> 我做的是一个基于静态 CSI300 股票池的月频多因子选股研究项目。先构建 16 个 alpha 因子，用 IC、分层回测和 Fama-MacBeth 做因子检验，再构建 Equal、IC、ICIR 三种传统多因子组合。之后我用严格 walk-forward 比较了 LightGBM、CatBoost、XGBoost、RandomForest 和 Ridge，并增加了一个 TopRisk 风控过滤层。当前版本里传统 baseline 已经能明显跑赢 benchmark，而 ML 中 RandomForest 收益最高，Ridge 风险调整后最稳。benchmark 也已经改成 510300 累计净值 proxy。需要诚实说明的是，这一版仍使用静态 CSI300 样本池，因此存在幸存者偏差。 

### 如果被问“为什么 Ridge 这么好”

> 说明这个月频因子问题里的主要有效信号仍然偏线性。Ridge 通过正则化压制了共线性和噪声，所以泛化表现比部分树模型更稳。

### 如果被问“项目最大局限”

> 当前最大局限不是模型，而是股票池口径。因为没有完整重建历史 CSI300 成分股进出路径，所以这一版保留静态成分股样本池，并明确承认幸存者偏差。

---

## 最终定位

这个项目当前最适合的定位是：

> 一个结构完整、结果可解释、文档诚实披露局限的量化研究型敲门砖项目。

它不是生产级无偏回测系统，但已经足够展示你对：

- 因子研究流程
- 回测口径
- benchmark 选择
- walk-forward 验证
- 风控消融实验
- 结果边界披露

这些量化研究基本范式的理解。 
