# -*- coding: utf-8 -*-
"""
Unified benchmark loader.

Priority:
  1. Exact total return index file, if locally provided.
  2. Benchmark ETF cumulative NAV proxy.
  3. Benchmark ETF adjusted close proxy.
  4. Price index + flat 2.0% annual dividend fallback.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BENCHMARK_ETF,
    INDEX_NAME,
    get_benchmark_nav_path,
    get_benchmark_qfq_path,
    get_index_daily_path,
    get_total_return_index_path,
)


_ANNUAL_DIVIDEND_FALLBACK = 0.02


def _standardize_daily_frame(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "收盘": "close",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "净值日期": "date",
        "累计净值": "close",
        "单位净值": "unit_nav",
        "日增长率": "daily_growth_pct",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
    required = {"date", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark file missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    df["_source"] = source
    return df


def _load_index_daily() -> pd.DataFrame:
    tr_path = get_total_return_index_path()
    if os.path.exists(tr_path):
        return _standardize_daily_frame(pd.read_parquet(tr_path), f"{INDEX_NAME}_total_return_index")

    nav_path = get_benchmark_nav_path()
    if os.path.exists(nav_path):
        return _standardize_daily_frame(pd.read_parquet(nav_path), f"etf_{BENCHMARK_ETF}_cumulative_nav")

    qfq_path = get_benchmark_qfq_path()
    if os.path.exists(qfq_path):
        return _standardize_daily_frame(pd.read_parquet(qfq_path), f"etf_{BENCHMARK_ETF}_qfq")

    price_path = get_index_daily_path()
    if not os.path.exists(price_path):
        raise FileNotFoundError(
            f"Benchmark data not found: {price_path}\n"
            "Run python data/download.py first."
        )
    return _standardize_daily_frame(pd.read_parquet(price_path), "price_index_dividend_proxy")


def get_benchmark_source() -> str:
    return str(_load_index_daily()["_source"].iloc[0])


def load_benchmark_returns() -> pd.Series:
    """
    Load monthly benchmark returns aligned to the signal month.
    """
    idx = _load_index_daily()
    source = str(idx["_source"].iloc[0])

    month_end = idx.groupby(idx["date"].dt.to_period("M")).tail(1).copy()
    month_end["ret"] = month_end["close"].pct_change()
    if source == "price_index_dividend_proxy":
        month_end["ret"] = month_end["ret"] + _ANNUAL_DIVIDEND_FALLBACK / 12

    month_end["year_month"] = month_end["date"].dt.to_period("M") - 1
    s = month_end.set_index("year_month")["ret"].dropna()
    return pd.Series(s.values, index=pd.PeriodIndex(s.index, freq="M"), name="benchmark")


def load_benchmark_returns_df() -> pd.DataFrame:
    s = load_benchmark_returns()
    return pd.DataFrame({
        "year_month": s.index,
        "benchmark_monthly_ret": s.values,
    })


def load_benchmark_daily_returns() -> pd.DataFrame:
    """
    Daily index returns for CAPM beta / IVOL estimation.

    Factor regressions continue to use the cash index, not ETF NAV.
    """
    price_path = get_index_daily_path()
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Daily index data not found: {price_path}")
    idx = _standardize_daily_frame(pd.read_parquet(price_path), "price_index")
    idx["index_ret"] = idx["close"].pct_change()
    return idx[["date", "index_ret"]].dropna().reset_index(drop=True)
