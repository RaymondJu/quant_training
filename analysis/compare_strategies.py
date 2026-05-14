# -*- coding: utf-8 -*-
"""
多策略汇总对比

将 equal / ic / icir（线性多因子）和 lgbm（机器学习）四条策略
与当前配置指数基准放在同一张图和统一绩效表中。

输出（output/analysis/）：
  strategy_comparison.png  → 五条净值曲线
  performance_table.csv    → 统一绩效对比表

Usage:
    python analysis/compare_strategies.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INDEX_NAME, OUTPUT_DIR
from portfolio.performance import summarize_returns

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STRATEGIES = {
    "Equal-weight":  ("portfolio/equal/backtest_returns.csv", "#4caf50"),
    "IC-weight":     ("portfolio/ic/backtest_returns.csv",    "#1976d2"),
    "ICIR-weight":   ("portfolio/icir/backtest_returns.csv",  "#7b1fa2"),
    "LightGBM":      ("ml/nav_lightgbm.csv",                  "#d32f2f"),
    "CatBoost":      ("ml/nav_catboost.csv",                  "#6a1b9a"),
    "XGBoost":       ("ml/nav_xgboost.csv",                   "#00838f"),
    "RandomForest":  ("ml/nav_randomforest.csv",              "#e65100"),
    "Ridge":         ("ml/nav_ridge.csv",                     "#9c27b0"),
}


def load_strategy_returns(rel_path: str) -> pd.Series:
    """加载策略月度收益序列，index=PeriodIndex(freq='M')"""
    path = os.path.join(OUTPUT_DIR, rel_path)
    if not os.path.exists(path):
        print(f"  [WARN] 文件不存在，跳过: {path}")
        return pd.Series(dtype=float)

    df = pd.read_csv(path)
    # 识别月份列和收益列
    ym_col = next((c for c in df.columns if "year_month" in c.lower() or c.lower() == "month"), None)
    ret_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["strategy_ret", "net_ret", "strategy"])),
        None,
    )
    if ym_col is None or ret_col is None:
        print(f"  [WARN] 无法识别列名: {list(df.columns)}")
        return pd.Series(dtype=float)

    df["_ym"] = pd.PeriodIndex(df[ym_col].astype(str), freq="M")
    return df.set_index("_ym")[ret_col].dropna().rename(ret_col)


def load_benchmark_returns() -> pd.Series:
    """Benchmark monthly total return, indexed by signal month."""
    from data.benchmark import load_benchmark_returns as _load
    return _load()


def build_performance_table(
    strategy_dict: dict[str, pd.Series],
    benchmark: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, rets in strategy_dict.items():
        if rets.empty:
            continue
        bm = benchmark.reindex(rets.index)
        perf = summarize_returns(rets, benchmark_returns=bm)
        perf.name = name
        rows.append(perf)

    # 基准行
    bm_aligned = benchmark.reindex(
        sorted(set().union(*[s.index for s in strategy_dict.values() if not s.empty]))
    ).dropna()
    bm_perf = summarize_returns(bm_aligned)
    bm_perf.name = f"{INDEX_NAME} (Benchmark)"
    rows.append(bm_perf)

    table = pd.DataFrame(rows)
    # 格式化
    pct_cols = ["Ann_Return", "Ann_Vol", "Max_Drawdown", "Excess_Ann_Return", "Tracking_Error",
                "Benchmark_Ann_Return"]
    for col in pct_cols:
        if col in table.columns:
            table[col] = table[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
    for col in ["Sharpe", "Info_Ratio"]:
        if col in table.columns:
            table[col] = table[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    if "Monthly_WinRate" in table.columns:
        table["Monthly_WinRate"] = table["Monthly_WinRate"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )
    return table


def plot_comparison(
    strategy_dict: dict[str, pd.Series],
    benchmark: pd.Series,
    colors: dict[str, str],
    save_path: str,
):
    """绘制多策略净值曲线对比图"""
    # 以最晚开始的 ML 模型为共同起点
    ml_names = {"LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ridge"}
    ml_starts = [s.index.min() for n, s in strategy_dict.items() if n in ml_names and not s.empty]
    if ml_starts:
        common_start = max(ml_starts)
    else:
        starts = [s.index.min() for s in strategy_dict.values() if not s.empty]
        common_start = max(starts) if starts else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                    gridspec_kw={"height_ratios": [2.5, 1]})

    # 上图：净值曲线
    for name, rets in strategy_dict.items():
        if rets.empty:
            continue
        sliced = rets[rets.index >= common_start] if common_start else rets
        nav = (1 + sliced).cumprod()
        dates = [p.to_timestamp() for p in nav.index]
        ml_names = {"LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ridge"}
        ls = "-" if name in ml_names else "--" if "IC" in name else "-."
        ax1.plot(dates, nav.values, label=name, color=colors[name],
                 linewidth=2, linestyle=ls)

    # 基准
    bm = benchmark[benchmark.index >= common_start] if common_start else benchmark
    bm_nav = (1 + bm.fillna(0)).cumprod()
    bm_dates = [p.to_timestamp() for p in bm_nav.index]
    ax1.plot(bm_dates, bm_nav.values, label=INDEX_NAME, color="gray",
             linewidth=1.5, linestyle=":")
    ax1.set_ylabel("Cumulative NAV")
    ax1.set_title("Strategy NAV Comparison (Common Window)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 下图：各 ML 模型超额收益（相对 benchmark）
    ml_names = {
        "LightGBM": "#d32f2f",
        "CatBoost": "#6a1b9a",
        "XGBoost": "#00838f",
        "RandomForest": "#e65100",
        "Ridge": "#9c27b0",
    }
    plotted_any = False
    for ml_name, ml_color in ml_names.items():
        ml_ret = strategy_dict.get(ml_name, pd.Series(dtype=float))
        if ml_ret.empty:
            continue
        ml_sliced = ml_ret[ml_ret.index >= common_start] if common_start else ml_ret
        bm_aligned = benchmark.reindex(ml_sliced.index).fillna(0)
        excess = ml_sliced - bm_aligned
        excess_cum = (1 + excess).cumprod()
        e_dates = [p.to_timestamp() for p in excess_cum.index]
        ax2.plot(e_dates, excess_cum.values, color=ml_color, linewidth=1.5, label=ml_name)
        plotted_any = True
    if plotted_any:
        ax2.axhline(1, color="black", linestyle="--", alpha=0.4)
        ax2.set_ylabel(f"Excess NAV vs {INDEX_NAME}")
        ax2.set_title(f"ML Model Excess Return vs {INDEX_NAME}")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] 对比图已保存: {save_path}")


def main():
    save_dir = os.path.join(OUTPUT_DIR, "analysis")
    os.makedirs(save_dir, exist_ok=True)

    print("[compare] 加载各策略收益序列...")
    strategy_dict: dict[str, pd.Series] = {}
    colors: dict[str, str] = {}
    for name, (rel_path, color) in STRATEGIES.items():
        rets = load_strategy_returns(rel_path)
        if not rets.empty:
            strategy_dict[name] = rets
            colors[name] = color
            print(f"  {name}: {len(rets)} 个月 ({rets.index.min()} ~ {rets.index.max()})")

    benchmark = load_benchmark_returns()

    # 绩效表
    print("\n[compare] 生成绩效对比表...")
    table = build_performance_table(strategy_dict, benchmark)
    table_path = os.path.join(save_dir, "performance_table.csv")
    table.to_csv(table_path, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("  Strategy Performance Comparison")
    print("=" * 70)
    display_cols = [c for c in ["Ann_Return", "Sharpe", "Max_Drawdown",
                                 "Excess_Ann_Return", "Info_Ratio"] if c in table.columns]
    print(table[display_cols].to_string())

    # 对比图
    plot_comparison(
        strategy_dict, benchmark, colors,
        os.path.join(save_dir, "strategy_comparison.png"),
    )

    print(f"\n[compare] 全部输出已保存至 {save_dir}/")


if __name__ == "__main__":
    main()
