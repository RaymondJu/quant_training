# -*- coding: utf-8 -*-
"""
LightGBM 组合回测 & 与 baseline 对比

读取 lgbm_predictions.parquet，按预测分数月度选股，
计算绩效并与 portfolio/ic baseline 对比。

输出（均在 output/ml/）：
  lgbm_nav.csv          → year_month, strategy_ret, benchmark_ret, nav, excess_nav
  lgbm_performance.csv  → 绩效摘要
  lgbm_vs_baseline.png  → LightGBM vs ic baseline 净值对比图

Usage:
    python ml/evaluate.py
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
from config import INDEX_NAME, OUTPUT_DIR, PROCESSED_DIR, TOP_N_STOCKS, TRANSACTION_COST
from portfolio.performance import summarize_returns

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_lgbm_predictions() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "lgbm_predictions.parquet")
    df = pd.read_parquet(path)
    if not hasattr(df["year_month"].dtype, "freq"):
        df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df


def load_factor_panel_returns() -> pd.DataFrame:
    """加载因子面板，仅取 stock_code, year_month, ret_next_month。"""
    path = os.path.join(PROCESSED_DIR, "factor_panel.parquet")
    df = pd.read_parquet(path)[["stock_code", "year_month", "ret_next_month"]]
    if not hasattr(df["year_month"].dtype, "freq"):
        df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df


def load_benchmark_returns() -> pd.Series:
    """月末基准全收益率，index=year_month（信号月，即持仓月）。"""
    from data.benchmark import load_benchmark_returns as _load
    return _load()


def run_lgbm_backtest(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark: pd.Series,
    top_n: int = TOP_N_STOCKS,
    cost: float = TRANSACTION_COST,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    月度选股回测

    Parameters
    ----------
    predictions : stock_code, year_month, lgbm_score
    returns     : stock_code, year_month, ret_next_month
    benchmark   : year_month → benchmark monthly return

    Returns
    -------
    nav_df  : year_month, strategy_ret, benchmark_ret, nav, excess_nav
    turnover: year_month → turnover rate
    """
    scored = predictions.merge(returns, on=["stock_code", "year_month"], how="inner")
    months = sorted(scored["year_month"].unique())

    rows = []
    prev_holdings: set[str] = set()
    turnover_list = []

    for ym in months:
        cross = scored[scored["year_month"] == ym].dropna(
            subset=["lgbm_score", "ret_next_month"]
        )
        if cross.empty:
            continue

        top = (
            cross.sort_values("lgbm_score", ascending=False)
            .head(top_n)["stock_code"]
            .tolist()
        )
        if not top:
            continue

        # 换手率 & 交易成本
        enter = set(top) - prev_holdings
        exit_ = prev_holdings - set(top)
        to = (len(enter) + len(exit_)) / max(len(prev_holdings | set(top)), 1)
        turnover_list.append(to)
        trade_cost = to * cost  # 单边千三

        # 等权持仓收益
        period_ret = cross[cross["stock_code"].isin(top)]["ret_next_month"].mean()
        net_ret = period_ret - trade_cost

        bm_ret = benchmark.get(ym, np.nan) if ym in benchmark.index else np.nan
        rows.append({"year_month": ym, "strategy_ret": net_ret, "benchmark_ret": bm_ret})
        prev_holdings = set(top)

    nav_df = pd.DataFrame(rows)
    nav_df["nav"] = (1 + nav_df["strategy_ret"]).cumprod()
    nav_df["benchmark_nav"] = (1 + nav_df["benchmark_ret"].fillna(0)).cumprod()
    nav_df["excess_nav"] = nav_df["nav"] / nav_df["benchmark_nav"]

    turnover_s = pd.Series(
        turnover_list,
        index=nav_df["year_month"].values[: len(turnover_list)],
        name="turnover",
    )
    return nav_df, turnover_s


def load_baseline_nav(method: str = "ic") -> pd.Series:
    """加载已有 baseline 的月度收益序列，index=year_month。"""
    path = os.path.join(OUTPUT_DIR, "portfolio", method, "backtest_returns.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    # 找包含月度收益的列（strategy_ret / net_ret 之类）
    ret_col = next(
        (c for c in df.columns if "strategy" in c.lower() or "net" in c.lower()),
        None,
    )
    ym_col = next((c for c in df.columns if "year_month" in c.lower() or "month" in c.lower()), None)
    if ret_col is None or ym_col is None:
        return pd.Series(dtype=float)
    df["_ym"] = pd.PeriodIndex(df[ym_col], freq="M")
    return df.set_index("_ym")[ret_col].dropna()


def plot_comparison(lgbm_nav: pd.DataFrame, baseline_ret: pd.Series, save_dir: str):
    """LightGBM 净值 vs ic baseline vs benchmark."""
    # 共同区间
    lgbm_start = lgbm_nav["year_month"].min()
    if not baseline_ret.empty:
        baseline_start = baseline_ret.index.min()
        common_start = max(lgbm_start, baseline_start)
    else:
        common_start = lgbm_start

    lgbm_slice = lgbm_nav[lgbm_nav["year_month"] >= common_start].copy()
    lgbm_dates = [p.to_timestamp() for p in lgbm_slice["year_month"]]

    fig, ax = plt.subplots(figsize=(12, 6))

    # LightGBM
    ax.plot(lgbm_dates, lgbm_slice["nav"].values, label="LightGBM", color="#d32f2f", linewidth=2)

    # benchmark
    ax.plot(lgbm_dates, lgbm_slice["benchmark_nav"].values,
            label=f"{INDEX_NAME} Benchmark", color="black", linewidth=1.5, linestyle="--")

    # ic baseline（如有）
    if not baseline_ret.empty:
        bl = baseline_ret[baseline_ret.index >= common_start]
        bl_nav = (1 + bl).cumprod()
        bl_dates = [p.to_timestamp() for p in bl_nav.index]
        ax.plot(bl_dates, bl_nav.values, label="Multi-factor (IC)", color="#1976d2",
                linewidth=2, linestyle="-.")

    ax.set_title(f"Strategy NAV Comparison: LightGBM vs Linear Multi-factor vs {INDEX_NAME}")
    ax.set_ylabel("Cumulative NAV")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    path = os.path.join(save_dir, "lgbm_vs_baseline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] 对比图已保存: {path}")


def main():
    save_dir = os.path.join(OUTPUT_DIR, "ml")
    os.makedirs(save_dir, exist_ok=True)

    print("[evaluate] 加载预测结果...")
    predictions = load_lgbm_predictions()
    returns = load_factor_panel_returns()
    benchmark = load_benchmark_returns()

    print(f"[evaluate] 预测月数: {predictions['year_month'].nunique()}, "
          f"股票数/月: {len(predictions)/predictions['year_month'].nunique():.0f}")

    print("[evaluate] 运行回测...")
    nav_df, turnover = run_lgbm_backtest(predictions, returns, benchmark)

    # 绩效统计（以 year_month 为索引对齐）
    nav_indexed = nav_df.set_index("year_month")
    perf = summarize_returns(
        nav_indexed["strategy_ret"],
        benchmark_returns=nav_indexed["benchmark_ret"],
        turnover=turnover,
    )

    print("\n" + "=" * 50)
    print("  LightGBM Portfolio Performance")
    print("=" * 50)
    print(perf.to_string(float_format="{:.4f}".format))

    # 保存
    nav_df.to_csv(os.path.join(save_dir, "lgbm_nav.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(perf).to_csv(
        os.path.join(save_dir, "lgbm_performance.csv"), encoding="utf-8-sig"
    )

    # 对比图
    baseline_ret = load_baseline_nav("ic")
    plot_comparison(nav_df, baseline_ret, save_dir)

    print(f"\n[evaluate] 全部输出已保存至 {save_dir}/")


if __name__ == "__main__":
    main()
