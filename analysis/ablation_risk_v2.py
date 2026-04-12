# -*- coding: utf-8 -*-
"""Risk-factor correlation audit and v2 weighted TopRisk ablation."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, PROCESSED_DIR, TOP_N_STOCKS, TOP_RISK_FILTER_PCT
from portfolio.backtest import run_portfolio_backtest
from portfolio.combine import (
    build_dynamic_factor_weights,
    combine_factor_scores,
    compute_rank_ic_series,
    get_available_factors,
    load_factor_panel,
)
from portfolio.performance import summarize_returns
from portfolio.backtest import select_top_n_portfolio
from ml.model_comparison import load_benchmark_returns

ABLATION_START = pd.Period("2017-10", freq="M")
ABLATION_END = pd.Period("2025-11", freq="M")
SAVE_DIR = os.path.join(OUTPUT_DIR, "ablation")

RISK_Z_COLS = ["BIAS_20_z", "UPSHADOW_20_z", "VOL_SPIKE_6M_z", "RET_6M_z"]
RISK_FACTORS = [c.replace("_z", "") for c in RISK_Z_COLS]
MIN_HISTORY_OBS = 20


def ensure_period(df: pd.DataFrame, col: str = "year_month") -> pd.DataFrame:
    df = df.copy()
    if not hasattr(df[col].dtype, "freq"):
        df[col] = pd.PeriodIndex(df[col], freq="M")
    return df


def load_risk_detail() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "factor_risk_detail.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing risk detail panel: {path}. Run python factors/risk.py first.")
    df = pd.read_parquet(path)
    df["stock_code"] = df["stock_code"].astype(str)
    return ensure_period(df)


def compute_monthly_average_corr(risk_detail: pd.DataFrame) -> pd.DataFrame:
    window = risk_detail[
        (risk_detail["year_month"] >= ABLATION_START) &
        (risk_detail["year_month"] <= ABLATION_END)
    ].copy()
    corr_sum = pd.DataFrame(0.0, index=RISK_FACTORS, columns=RISK_FACTORS)
    corr_count = pd.DataFrame(0, index=RISK_FACTORS, columns=RISK_FACTORS)

    for _, grp in window.groupby("year_month"):
        cross = grp[RISK_Z_COLS].rename(columns=dict(zip(RISK_Z_COLS, RISK_FACTORS)))
        corr = cross.corr()
        valid = corr.notna()
        corr_sum = corr_sum.add(corr.fillna(0.0), fill_value=0.0)
        corr_count = corr_count.add(valid.astype(int), fill_value=0)

    avg_corr = corr_sum / corr_count.replace(0, np.nan)
    return avg_corr.loc[RISK_FACTORS, RISK_FACTORS]


def plot_corr_heatmap(corr: pd.DataFrame, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticklabels(corr.index)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=9)

    ax.set_title("Risk Factor Cross-Sectional Correlation\n(monthly corr, 2017-10~2025-11 average)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_icir_holdings(panel: pd.DataFrame) -> pd.DataFrame:
    available = get_available_factors(panel)
    ic_df = compute_rank_ic_series(panel, available)
    weights_df = build_dynamic_factor_weights(ic_df, method="icir")
    scored_panel = combine_factor_scores(panel, weights_df, available)
    holdings = select_top_n_portfolio(scored_panel, top_n=TOP_N_STOCKS)
    holdings["stock_code"] = holdings["stock_code"].astype(str)
    return ensure_period(holdings)


def identify_dominant_factor(row: pd.Series) -> str:
    scores = {factor: abs(row[f"{factor}_z"]) for factor in RISK_FACTORS if pd.notna(row[f"{factor}_z"])}
    if not scores:
        return "UNKNOWN"
    return max(scores, key=scores.get)


def build_v1_exclusion_events(holdings: pd.DataFrame, risk_detail: pd.DataFrame) -> pd.DataFrame:
    merged = holdings.merge(
        risk_detail[["stock_code", "year_month"] + RISK_Z_COLS],
        on=["stock_code", "year_month"],
        how="left",
    )
    merged["TOP_RISK_SCORE_V1"] = merged[RISK_Z_COLS].mean(axis=1)
    rows = []

    for ym, grp in merged.groupby("year_month"):
        cross = grp.dropna(subset=["TOP_RISK_SCORE_V1", "ret_next_month"]).copy()
        if cross.empty:
            continue
        n_exclude = max(1, int(len(cross) * TOP_RISK_FILTER_PCT))
        excluded = cross.sort_values("TOP_RISK_SCORE_V1", ascending=False).head(n_exclude).copy()
        excluded["dominant_factor"] = excluded.apply(identify_dominant_factor, axis=1)
        rows.append(excluded[["year_month", "stock_code", "ret_next_month", "dominant_factor"]])

    if not rows:
        raise RuntimeError("No v1 exclusion events generated.")
    return pd.concat(rows, ignore_index=True)


def build_driver_exclusion_events(risk_detail: pd.DataFrame) -> pd.DataFrame:
    """
    Load the v1 equal-weight TopRisk exclusion log used by
    exclusion_driver_analysis.py, then assign each excluded stock to the
    dominant risk factor. This keeps the v2 weighting experiment aligned with
    the documented driver analysis instead of using an ICIR-only subset.
    """
    log_path = os.path.join(SAVE_DIR, "excluded_stocks_log.csv")
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Missing v1 exclusion log: {log_path}. Run python analysis/ablation_top_risk.py first."
        )

    log_df = pd.read_csv(log_path, dtype={"stock_code": str})
    log_df["year_month"] = pd.PeriodIndex(log_df["month"], freq="M")
    log_df = log_df.drop_duplicates(subset=["year_month", "stock_code"]).reset_index(drop=True)

    ret_panel = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "factor_panel.parquet"),
        columns=["stock_code", "year_month", "ret_next_month"],
    )
    ret_panel["stock_code"] = ret_panel["stock_code"].astype(str)
    ret_panel = ensure_period(ret_panel)

    merged = log_df.merge(
        risk_detail[["stock_code", "year_month"] + RISK_Z_COLS],
        on=["stock_code", "year_month"],
        how="left",
    ).merge(
        ret_panel,
        on=["stock_code", "year_month"],
        how="left",
    )
    merged = merged.dropna(subset=["ret_next_month"]).copy()
    merged["dominant_factor"] = merged.apply(identify_dominant_factor, axis=1)
    return merged[["year_month", "stock_code", "ret_next_month", "dominant_factor"]]


def estimate_walk_forward_risk_weights(exclusion_events: pd.DataFrame, months: list[pd.Period]) -> pd.DataFrame:
    weight_rows = []
    for ym in months:
        hist = exclusion_events[exclusion_events["year_month"] < ym]
        raw = {}
        for factor in RISK_FACTORS:
            sub = hist[hist["dominant_factor"] == factor]
            if len(sub) >= MIN_HISTORY_OBS:
                raw[factor] = (sub["ret_next_month"] < 0).mean() - 0.5
            else:
                raw[factor] = np.nan

        weights = pd.Series(raw, dtype=float)
        if weights.notna().sum() == 0:
            weights = pd.Series(1.0 / len(RISK_FACTORS), index=RISK_FACTORS)
            mode = "equal_fallback"
        else:
            weights = weights.fillna(0.0)
            mode = "walk_forward"

        row = {"year_month": ym, "weight_mode": mode}
        row.update({f"{factor}_weight": weights[factor] for factor in RISK_FACTORS})
        weight_rows.append(row)

    return pd.DataFrame(weight_rows)


def build_v2_risk_panel(risk_detail: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    panel = risk_detail[["stock_code", "year_month"] + RISK_Z_COLS].copy()
    panel = panel.merge(weights_df, on="year_month", how="left")
    score = 0.0
    for factor in RISK_FACTORS:
        score = score + panel[f"{factor}_z"].fillna(0.0) * panel[f"{factor}_weight"].fillna(0.0)
    panel["TOP_RISK_SCORE"] = score
    return panel[["stock_code", "year_month", "TOP_RISK_SCORE"]]


def compute_performance(rets: pd.Series, benchmark: pd.Series, turnover: pd.Series | None, label: str) -> dict:
    rets = rets[(rets.index >= ABLATION_START) & (rets.index <= ABLATION_END)]
    bm = benchmark.reindex(rets.index)
    to = turnover.reindex(rets.index) if turnover is not None else None
    perf = summarize_returns(rets, benchmark_returns=bm, turnover=to).to_dict()
    ann_ret = perf.get("Ann_Return", np.nan)
    max_dd = perf.get("Max_Drawdown", np.nan)
    perf["Calmar"] = ann_ret / abs(max_dd) if pd.notna(max_dd) and max_dd != 0 else np.nan
    perf["Label"] = label
    perf["N_months"] = len(rets)
    return perf


def returns_from_backtest(result_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rets = result_df.set_index("year_month")["net_ret"]
    turnover = result_df.set_index("year_month")["turnover"]
    rets.index = pd.PeriodIndex(rets.index, freq="M")
    turnover.index = pd.PeriodIndex(turnover.index, freq="M")
    return rets, turnover


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[risk_v2] window: {ABLATION_START} ~ {ABLATION_END}")

    risk_detail = load_risk_detail()
    factor_panel = load_factor_panel()
    benchmark = load_benchmark_returns()

    corr = compute_monthly_average_corr(risk_detail)
    corr_path = os.path.join(SAVE_DIR, "risk_factor_correlation.csv")
    corr.to_csv(corr_path, encoding="utf-8-sig")
    plot_corr_heatmap(corr, os.path.join(SAVE_DIR, "risk_factor_corr.png"))
    print(f"[risk_v2] correlation outputs saved: {corr_path}")

    holdings = build_icir_holdings(factor_panel)
    icir_exclusion_events = build_v1_exclusion_events(holdings, risk_detail)
    icir_exclusion_events.to_csv(
        os.path.join(SAVE_DIR, "v2_icir_only_weight_training_events.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    exclusion_events = build_driver_exclusion_events(risk_detail)
    exclusion_events.to_csv(
        os.path.join(SAVE_DIR, "v2_weight_training_events.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    months = sorted(risk_detail["year_month"].unique())
    weights_df = estimate_walk_forward_risk_weights(exclusion_events, months)
    weights_df.to_csv(os.path.join(SAVE_DIR, "v2_risk_factor_weights.csv"), index=False, encoding="utf-8-sig")
    v2_risk_panel = build_v2_risk_panel(risk_detail, weights_df)

    import config as cfg
    orig_flag = cfg.ENABLE_TOP_RISK_FILTER
    cfg.ENABLE_TOP_RISK_FILTER = False
    no_filter_result, _ = run_portfolio_backtest(
        method="icir",
        risk_panel=None,
        save_root="ablation/icir_v2_compare_no_filter",
    )
    cfg.ENABLE_TOP_RISK_FILTER = True
    v1_result, _ = run_portfolio_backtest(
        method="icir",
        risk_panel=pd.read_parquet(os.path.join(PROCESSED_DIR, "factor_risk.parquet")),
        save_root="ablation/icir_v2_compare_v1_equal",
    )
    v2_result, _ = run_portfolio_backtest(
        method="icir",
        risk_panel=v2_risk_panel,
        save_root="ablation/icir_v2_compare_v2_weighted",
    )
    cfg.ENABLE_TOP_RISK_FILTER = orig_flag

    rets_no, to_no = returns_from_backtest(no_filter_result)
    rets_v1, to_v1 = returns_from_backtest(v1_result)
    rets_v2, to_v2 = returns_from_backtest(v2_result)

    rows = [
        compute_performance(rets_no, benchmark, to_no, "ICIR (no filter)"),
        compute_performance(rets_v1, benchmark, to_v1, "ICIR + TopRisk v1 equal"),
        compute_performance(rets_v2, benchmark, to_v2, "ICIR + TopRisk v2 IC-weighted"),
    ]
    perf = pd.DataFrame(rows).set_index("Label")
    v2_path = os.path.join(SAVE_DIR, "v2_performance.csv")
    perf.to_csv(v2_path, encoding="utf-8-sig")

    display_cols = ["Ann_Return", "Sharpe", "Max_Drawdown", "Calmar", "Info_Ratio", "Avg_Turnover", "N_months"]
    print("\n[risk_v2] performance")
    print(perf[display_cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n[risk_v2] saved: {v2_path}")


if __name__ == "__main__":
    main()
