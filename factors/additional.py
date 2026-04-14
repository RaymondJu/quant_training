# -*- coding: utf-8 -*-
"""
Additional factor families:
    - SIZE: negative log market cap, higher means smaller size
    - BETA_60D: rolling 60-trading-day market beta
    - ABTURN_1M: abnormal turnover vs prior 12-month average
    - OCF_QUALITY: operating cash flow / net profit quality proxy
    - ASSET_GROWTH: total asset growth

Usage:
    python factors/additional.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RAW_DIR
from factors.utils import compute_monthly_market_cap, load_daily_prices, load_monthly_panel


def build_additional_factors() -> pd.DataFrame:
    print("[additional] loading monthly panel...")
    monthly = load_monthly_panel().copy()
    monthly = monthly.sort_values(["stock_code", "year_month"]).reset_index(drop=True)

    # ---- SIZE ----
    print("[additional] computing SIZE...")
    daily = load_daily_prices()
    market_cap = compute_monthly_market_cap(daily)[["stock_code", "year_month", "ln_market_cap"]].copy()
    market_cap["SIZE"] = -market_cap["ln_market_cap"]

    # ---- BETA_60D ----
    print("[additional] computing BETA_60D...")
    daily = daily.sort_values(["stock_code", "date"]).copy()
    daily["daily_ret"] = daily["pct_change"] / 100.0

    from data.benchmark import load_benchmark_daily_returns
    index_df = load_benchmark_daily_returns()  # [date, index_ret]

    beta_daily = daily.merge(index_df[["date", "index_ret"]], on="date", how="inner")
    beta_daily = beta_daily.dropna(subset=["daily_ret", "index_ret"]).copy()
    beta_daily["year_month"] = beta_daily["date"].dt.to_period("M")

    beta_parts = []
    for stock_code, group in beta_daily.groupby("stock_code"):
        group = group.sort_values("date").copy()
        cov = group["daily_ret"].rolling(window=60, min_periods=40).cov(group["index_ret"])
        var = group["index_ret"].rolling(window=60, min_periods=40).var()
        group["stock_code"] = stock_code
        group["BETA_60D"] = cov / var.replace(0, np.nan)
        beta_parts.append(group[["stock_code", "year_month", "BETA_60D"]])

    beta_monthly = (
        pd.concat(beta_parts, ignore_index=True)
        .groupby(["stock_code", "year_month"])["BETA_60D"]
        .last()
        .reset_index()
    )

    # ---- ABTURN_1M ----
    print("[additional] computing ABTURN_1M...")
    monthly["turnover_mean_12m"] = (
        monthly.groupby("stock_code")["avg_turnover_rate"]
        .transform(lambda x: x.shift(1).rolling(12, min_periods=6).mean())
    )
    monthly["ABTURN_1M"] = monthly["avg_turnover_rate"] - monthly["turnover_mean_12m"]

    # ---- OCF_QUALITY / ASSET_GROWTH ----
    print("[additional] reading financial quality columns from monthly panel...")
    monthly["OCF_QUALITY"] = monthly["经营现金净流量与净利润的比率(%)"] / 100.0
    monthly["ASSET_GROWTH"] = monthly["总资产增长率(%)"] / 100.0

    result = monthly[["stock_code", "year_month", "ABTURN_1M", "OCF_QUALITY", "ASSET_GROWTH"]].copy()
    result = result.merge(
        market_cap[["stock_code", "year_month", "SIZE"]],
        on=["stock_code", "year_month"],
        how="left",
    )
    result = result.merge(
        beta_monthly,
        on=["stock_code", "year_month"],
        how="left",
    )

    output_cols = ["stock_code", "year_month", "SIZE", "BETA_60D", "ABTURN_1M", "OCF_QUALITY", "ASSET_GROWTH"]
    output = result[output_cols]

    print(f"[additional] done: {len(output)} rows")
    for col in output_cols[2:]:
        print(f"  {col}: NaN={output[col].isna().mean():.1%}, median={output[col].median():.6f}")
    return output


if __name__ == "__main__":
    factors = build_additional_factors()
    out_path = os.path.join(PROCESSED_DIR, "factor_additional.parquet")
    factors.to_parquet(out_path, index=False)
    print(f"[additional] saved to {out_path}")
