# -*- coding: utf-8 -*-
"""
统一 Benchmark 加载模块

提供两个接口:
  load_benchmark_returns()       — 月度 benchmark 收益 (PeriodIndex, signal month 对齐)
  load_benchmark_daily_returns() — 日度 benchmark 收益 (用于 CAPM 回归)

数据来源优先级:
  1. index_hs300_total_return.parquet (全收益指数, 若已下载)
  2. index_hs300_daily.parquet (价格指数) + 年化 2.0% 股息率近似

CSI300 历史平均股息率约 2.0-2.5%，取 2.0% 为保守估计。
近似方法: monthly_ret_total_return = monthly_ret_price + 0.02 / 12
验证: 2015-07 ~ 2025-11 (125月) 基准年化约 3.9%（合理区间 3-5%）
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR

# CSI300 平均年化股息率（保守估计）
_CSI300_ANNUAL_DIV_YIELD = 0.02


def _load_index_daily() -> pd.DataFrame:
    """加载日频指数数据，返回含 [date, close] 的 DataFrame。"""
    # 优先: 全收益指数
    tr_path = os.path.join(RAW_DIR, "index_hs300_total_return.parquet")
    if os.path.exists(tr_path):
        df = pd.read_parquet(tr_path)
        df["date"] = pd.to_datetime(df["date"])
        df["_source"] = "total_return"
        return df.sort_values("date").reset_index(drop=True)

    # Fallback: 价格指数
    price_path = os.path.join(RAW_DIR, "index_hs300_daily.parquet")
    if not os.path.exists(price_path):
        raise FileNotFoundError(
            f"Benchmark 数据不存在: {price_path}\n"
            "请先运行: python data/download.py"
        )
    df = pd.read_parquet(price_path)
    df["date"] = pd.to_datetime(df["date"])
    df["_source"] = "price_index"
    return df.sort_values("date").reset_index(drop=True)


def load_benchmark_returns() -> pd.Series:
    """
    加载月度 benchmark 收益（全收益或近似），返回 pd.Series。

    - index: PeriodIndex(freq='M'), 为 **signal month**（已 shift -1）
    - values: float, 月度收益率

    与 portfolio/backtest.py 的对齐逻辑一致：
    signal_month=t → 月末 t+1 的 close pct_change 作为 t 的 benchmark return。
    """
    idx = _load_index_daily()
    is_price_only = (idx["_source"].iloc[0] == "price_index")

    month_end = idx.groupby(idx["date"].dt.to_period("M")).tail(1).copy()
    month_end["ret"] = month_end["close"].pct_change()

    # 如果是价格指数，叠加股息率近似
    if is_price_only:
        month_end["ret"] = month_end["ret"] + _CSI300_ANNUAL_DIV_YIELD / 12

    # signal month 对齐: year_month = 数据月 - 1
    month_end["year_month"] = month_end["date"].dt.to_period("M") - 1
    s = month_end.set_index("year_month")["ret"].dropna()
    return pd.Series(s.values, index=pd.PeriodIndex(s.index, freq="M"), name="benchmark")


def load_benchmark_returns_df() -> pd.DataFrame:
    """
    返回 DataFrame 格式（兼容 portfolio/backtest.py 的旧接口）。
    列: [year_month (Period), benchmark_monthly_ret (float)]
    """
    s = load_benchmark_returns()
    return pd.DataFrame({
        "year_month": s.index,
        "benchmark_monthly_ret": s.values,
    })


def load_benchmark_daily_returns() -> pd.DataFrame:
    """
    日频 benchmark 收益，用于 CAPM 回归（IVOL / BETA_60D）。
    返回 DataFrame: [date, index_ret]

    注意: 日级别股息率影响极小（~0.008%/天），此处不做近似调整。
    继续使用价格指数日收益，不影响 beta 估计。
    """
    price_path = os.path.join(RAW_DIR, "index_hs300_daily.parquet")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Benchmark 日频数据不存在: {price_path}")
    idx = pd.read_parquet(price_path)
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    idx["index_ret"] = idx["close"].pct_change()
    return idx[["date", "index_ret"]].dropna().reset_index(drop=True)
