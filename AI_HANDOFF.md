# AI_HANDOFF — A股多因子选股系统

> 写给下一个 coding agent。项目路径：`E:\大四上课程\毕业设计\数据\output\quant_training\`

---

## 项目概览

**目标**：沪深300多因子量化选股系统（毕业设计），验证多因子 → 组合回测 → ML增强的完整链路。

**数据范围**：2015-07 ~ 2025-12，279只股票（沪深300成分），月频因子面板。

**进度**：**全部主体模块已完成并运行**，产出最终对比结果。

---

## 已完成模块

```
数据层     → factors层    → 因子检验层    → 组合回测层    → ML增强层    → 汇总对比
clean.py     utils.py        ic_analysis      combine.py      lgbm_model    compare_strategies
download.py  value/momentum  quantile_bt      backtest.py     model_comparison
             quality/vol/liq fama_macbeth     performance.py
             preprocess.py
```

### 核心文件

| 文件 | 作用 |
|------|------|
| `data/processed/factor_panel.parquet` | 主数据表：30,249行×16列，11因子+ret_next_month |
| `factors/utils.py` | TTM计算、市值计算、月度对齐 |
| `factors/preprocess.py` | MAD截尾→Z-score→行业中性化 |
| `testing/ic_analysis.py` | Rank IC / ICIR 检验 |
| `testing/quantile_backtest.py` | 分位数回测 |
| `testing/fama_macbeth.py` | Fama-MacBeth两步回归 |
| `portfolio/combine.py` | 因子合成打分（equal/IC/ICIR权重） |
| `portfolio/backtest.py` | 组合回测引擎（Top50等权，单边千三成本） |
| `ml/model_comparison.py` | LightGBM/RF/Ridge 统一Walk-Forward框架 |
| `analysis/compare_strategies.py` | 6策略+基准统一对比 |

---

## 关键实证结论

### 因子质量（IC分析）

| 因子 | IC_mean | ICIR | 显著性 |
|------|---------|------|--------|
| AMIHUD（流动性） | 0.0253 | 0.227 | 最强 ★★★ |
| IVOL | — | — | FM t=2.39** |
| VOL_20D | — | — | FM t=2.33** |
| EP/BP/SP（价值）| 负 | — | A股价值因子失效 |

> A股特征：流动性溢价显著，价值因子反转（小市值炒作行为）。

### 最终策略对比（`output/analysis/performance_table.csv`）

| 策略 | 年化收益 | Sharpe | 最大回撤 | 超额年化 | IR |
|------|---------|--------|---------|---------|-----|
| Equal-weight | 17.74% | 0.809 | -26.36% | 15.49% | 1.282 |
| IC-weight | 18.12% | 0.821 | -26.44% | 15.91% | 1.338 |
| ICIR-weight | 17.63% | 0.801 | -27.68% | 15.46% | 1.332 |
| LightGBM | 16.48% | 0.778 | -29.02% | 14.41% | 1.318 |
| **RandomForest** | **19.44%** | 0.872 | -23.10% | **17.43%** | 1.446 |
| **Ridge** | 19.02% | **0.924** | **-19.89%** | 16.85% | **1.651** |
| HS300基准 | 1.87% | 0.098 | -39.92% | — | — |

> - 回测区间：baseline策略2015-07~2025-11（125月）；ML策略2017-10~2025-11（98月）
> - Walk-forward：24月训练+3月验证，严格无未来数据
> - Ridge表现最好（Sharpe最高、回撤最小），说明月频因子信号以线性为主

---

## 技术约定（重要）

- **市值计算**：`turnover / volume * outstanding_share`（非复权价格代理，规避hfq跨股比较失真）
- **TTM公式**：`Q_current + Annual(Y-1) - Q_corr(Y-1)`；年报 TTM = 值本身
- **财务数据对齐**：使用真实 `NOTICE_DATE`（非报告期），防止未来数据泄漏
- **MAD截尾**：`median ± 3 × 1.4826 × MAD`
- **行业中性化**：OLS回归28个申万一级行业哑变量（6位代码前2位）
- **组合构建**：每月Top 50等权，月末换仓，单边0.3%成本

---

## 输出文件

```
output/
├── ic_analysis/          IC汇总、时序图、衰减图
├── quantile_backtest/    各因子分位数回测图
├── fama_macbeth/         FM回归系数
├── portfolio/            equal/ic/icir三套净值与绩效
├── ml/
│   ├── model_comparison.csv      三模型绩效汇总
│   ├── model_comparison_nav.png  三模型净值曲线
│   ├── nav_{lightgbm/randomforest/ridge}.csv  各模型月度收益
│   └── feature_importance_*.csv/png  各模型特征重要性
└── analysis/
    ├── performance_table.csv     6策略+基准最终对比表
    └── strategy_comparison.png   6策略+基准净值曲线图
```

---

## 可选后续工作

以下均为**可选**，主体实证已完整，可直接撰写论文：

1. **LSTM/Transformer**：`dl/` 目录尚未创建。月频数据量小（~30k行），深度学习很可能欠拟合，性价比低。
2. **因子相关性分析深化**：VOL_20D ↔ IVOL 相关0.90，可做正交化处理后重跑。
3. **调参实验**：Ridge/RF 目前使用默认参数，可系统做 GridSearch（但Walk-forward下耗时高）。
4. **归因分析**：将超额收益归因到行业/风格暴露（Barra风格因子）。

---

## 环境

- Python 环境：`D:\Anaconda`
- 主要依赖：pandas, numpy, lightgbm, sklearn, matplotlib
- 注意：`shap` 不可用（numba与当前numpy版本冲突），已用 `booster_.feature_importance(importance_type="gain")` 替代
- 控制台编码：GBK，print中文可能乱码，不影响文件输出
