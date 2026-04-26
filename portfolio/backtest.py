# -*- coding: utf-8 -*-
"""Monthly top-N portfolio backtest using combined factor scores."""
from __future__ import annotations

import os
import sys
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INDEX_NAME, OUTPUT_DIR, TOP_N_STOCKS, TRANSACTION_COST
from data.benchmark import load_benchmark_returns_df as _load_benchmark
from portfolio.combine import (
    build_dynamic_factor_weights,
    combine_factor_scores,
    compute_rank_ic_series,
    get_available_factors,
    load_factor_panel,
    save_weight_snapshot,
)
from portfolio.performance import summarize_returns

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_benchmark_returns() -> pd.DataFrame:
    """Load benchmark monthly returns aligned to the signal month."""
    return _load_benchmark()


def _compute_turnover(prev_holdings: dict[str, float], curr_holdings: dict[str, float]) -> float:
    union = set(prev_holdings) | set(curr_holdings)
    return 0.5 * sum(abs(curr_holdings.get(code, 0.0) - prev_holdings.get(code, 0.0)) for code in union)


def select_top_n_portfolio(
    scored_panel: pd.DataFrame,
    top_n: int = TOP_N_STOCKS,
    score_col: str = "composite_score",
) -> pd.DataFrame:
    """Select the top-N stocks by score in each rebalance month."""
    rows = []
    for ym, grp in scored_panel.groupby("year_month"):
        ranked = (
            grp.dropna(subset=[score_col, "ret_next_month"])
            .sort_values(score_col, ascending=False)
            .head(top_n)
        )
        if ranked.empty:
            continue
        weight = 1.0 / len(ranked)
        for _, row in ranked.iterrows():
            rows.append(
                {
                    "year_month": ym,
                    "stock_code": row["stock_code"],
                    "score": row[score_col],
                    "ret_next_month": row["ret_next_month"],
                    "weight": weight,
                }
            )
    return pd.DataFrame(rows)


def plot_backtest_curves(result_df: pd.DataFrame, save_dir: str, method: str) -> None:
    """Plot strategy, benchmark, and excess NAV curves."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    dates = result_df["year_month_ts"]

    axes[0].plot(dates, result_df["strategy_nav"], label=f"{method} strategy", linewidth=1.8)
    axes[0].plot(dates, result_df["benchmark_nav"], label=INDEX_NAME, linewidth=1.6)
    axes[0].set_title(f"Portfolio Backtest - {method}")
    axes[0].set_ylabel("NAV")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(dates, result_df["excess_nav"], color="#d32f2f", linewidth=1.6)
    axes[1].axhline(y=1.0, color="black", linestyle="--", alpha=0.3)
    axes[1].set_title(f"Excess NAV vs {INDEX_NAME}")
    axes[1].set_ylabel("Excess NAV")
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "nav_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_portfolio_backtest(
    panel: pd.DataFrame | None = None,
    factor_cols: Iterable[str] | None = None,
    method: str = "icir",
    top_n: int = TOP_N_STOCKS,
    transaction_cost: float = TRANSACTION_COST,
    save_root: str = "portfolio",
    risk_panel: pd.DataFrame | None = None,
    log_path: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run one monthly top-N backtest for a given factor-combination method."""
    panel = load_factor_panel() if panel is None else panel.copy()
    available = get_available_factors(panel, factor_cols)
    ic_df = compute_rank_ic_series(panel, available)
    weights_df = build_dynamic_factor_weights(ic_df, method=method)
    scored_panel = combine_factor_scores(panel, weights_df, available)
    holdings = select_top_n_portfolio(scored_panel, top_n=top_n)

    # ── 顶部风险过滤层 ──────────────────────────────────────────────────────
    # 仅在 ENABLE_TOP_RISK_FILTER=True 且传入了 risk_panel 时生效。
    # risk_panel=None 时完全跳过，保证 ablation 对照组与旧结果完全一致。
    from config import ENABLE_TOP_RISK_FILTER, TOP_RISK_FILTER_PCT
    from portfolio.risk_filter import apply_top_risk_filter as _apply_risk_filter

    if ENABLE_TOP_RISK_FILTER and risk_panel is not None:
        log_records: list = []
        filtered_rows = []
        for ym, grp in holdings.groupby("year_month"):
            kept = _apply_risk_filter(
                grp["stock_code"].tolist(), ym, risk_panel,
                TOP_RISK_FILTER_PCT, log_records
            )
            kept_grp = grp[grp["stock_code"].isin(kept)].copy()
            if len(kept_grp) > 0:
                kept_grp["weight"] = 1.0 / len(kept_grp)
            filtered_rows.append(kept_grp)
        holdings = pd.concat(filtered_rows, ignore_index=True)
        if log_path and log_records:
            pd.DataFrame(log_records).to_csv(log_path, index=False, encoding="utf-8-sig")
    # ──────────────────────────────────────────────────────────────────────

    benchmark = load_benchmark_returns()
    results = []
    prev_holdings: dict[str, float] = {}

    for ym, grp in holdings.groupby("year_month"):
        curr_holdings = dict(zip(grp["stock_code"], grp["weight"]))
        turnover = _compute_turnover(prev_holdings, curr_holdings) if prev_holdings else 1.0
        gross_ret = (grp["ret_next_month"] * grp["weight"]).sum()
        cost = turnover * 2 * transaction_cost
        net_ret = gross_ret - cost
        results.append(
            {
                "year_month": ym,
                "n_stocks": len(grp),
                "gross_ret": gross_ret,
                "turnover": turnover,
                "cost": cost,
                "net_ret": net_ret,
            }
        )
        prev_holdings = curr_holdings

    result_df = pd.DataFrame(results).sort_values("year_month").reset_index(drop=True)
    result_df = result_df.merge(benchmark, on="year_month", how="left")
    result_df["excess_ret"] = result_df["net_ret"] - result_df["benchmark_monthly_ret"]
    result_df["strategy_nav"] = (1 + result_df["net_ret"]).cumprod()
    result_df["benchmark_nav"] = (1 + result_df["benchmark_monthly_ret"].fillna(0.0)).cumprod()
    result_df["excess_nav"] = (1 + result_df["excess_ret"].fillna(0.0)).cumprod()
    result_df["year_month_ts"] = result_df["year_month"].dt.to_timestamp()

    result_indexed = result_df.set_index("year_month")
    summary = summarize_returns(
        strategy_returns=result_indexed["net_ret"],
        benchmark_returns=result_indexed["benchmark_monthly_ret"],
        turnover=result_indexed["turnover"],
    )

    save_dir = os.path.join(OUTPUT_DIR, save_root, method)
    os.makedirs(save_dir, exist_ok=True)
    result_df.to_csv(os.path.join(save_dir, "backtest_returns.csv"), index=False, encoding="utf-8-sig")
    holdings.to_csv(os.path.join(save_dir, "selected_holdings.csv"), index=False, encoding="utf-8-sig")
    summary.to_frame(name=method).to_csv(
        os.path.join(save_dir, "performance_summary.csv"), encoding="utf-8-sig"
    )
    save_weight_snapshot(weights_df, method, save_root=save_root)
    plot_backtest_curves(result_df, save_dir, method)
    return result_df, summary


def run_all_methods(
    methods: Iterable[str] = ("equal", "ic", "icir"),
    panel_file: str = "factor_panel.parquet",
    save_root: str = "portfolio",
) -> pd.DataFrame:
    """Run all supported combination methods and return the comparison table."""
    panel = load_factor_panel(panel_file=panel_file)
    summaries = {}
    for method in methods:
        _, summary = run_portfolio_backtest(panel=panel, method=method, save_root=save_root)
        summaries[method] = summary

    comparison = pd.DataFrame(summaries).T
    save_dir = os.path.join(OUTPUT_DIR, save_root)
    os.makedirs(save_dir, exist_ok=True)
    comparison.to_csv(os.path.join(save_dir, "method_comparison.csv"), encoding="utf-8-sig")
    return comparison


if __name__ == "__main__":
    comparison = run_all_methods()
    print("\n" + "=" * 70)
    print("  Portfolio Method Comparison")
    print("=" * 70)
    print(comparison.to_string(float_format="{:.4f}".format))
