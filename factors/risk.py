# -*- coding: utf-8 -*-
"""
顶部风险因子: BIAS_20, UPSHADOW_20, VOL_SPIKE_6M, RET_6M

设计说明:
  - 定位为"风控层"，用于过滤 Top-N 选股中风险最高的部分
  - 不做行业中性化（与 alpha 因子处理方式不同）
  - 所有因子在合成后整体 shift(1)，保证 year_month=t 的风控信号
    使用的是 t-1 月末及之前的数据，严格避免 look-ahead bias

因子定义:
  BIAS_20     : 月末收盘价 / 过去20日均价 - 1（超买程度）
  UPSHADOW_20 : 过去20日日均上影线比率 = (high - max(open,close)) / (high-low+eps)
  VOL_SPIKE_6M: 当月日均成交额 / 过去6月日均成交额均值 - 1（资金涌入程度）
  RET_6M      : 过去6个月累计收益率（近期涨幅）

Usage:
    python factors/risk.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR
from factors.preprocess import winsorize_mad, standardize
from factors.utils import load_daily_prices

RISK_FACTOR_COLS = ["BIAS_20", "UPSHADOW_20", "VOL_SPIKE_6M", "RET_6M"]


def _build_raw_risk_factors() -> pd.DataFrame:
    """从日频价格数据计算 4 个月频原始风控因子（未标准化，自然时间对齐）。"""
    print("[risk] 加载日频数据...")
    daily = load_daily_prices()
    daily = daily.sort_values(["stock_code", "date"]).copy()
    daily["year_month"] = daily["date"].dt.to_period("M")
    daily["daily_ret"] = daily["pct_change"] / 100.0

    # ---- BIAS_20: 月末 close / 20日SMA - 1 ----
    print("[risk] 计算 BIAS_20...")
    daily["sma_20"] = daily.groupby("stock_code")["close"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    daily["bias_20_raw"] = daily["close"] / daily["sma_20"].replace(0, np.nan) - 1
    bias_monthly = (
        daily.groupby(["stock_code", "year_month"])["bias_20_raw"]
        .last()
        .reset_index()
        .rename(columns={"bias_20_raw": "BIAS_20"})
    )

    # ---- UPSHADOW_20: 20日均上影线比率 ----
    print("[risk] 计算 UPSHADOW_20...")
    eps = 1e-8
    body_top = daily[["open", "close"]].max(axis=1)
    daily["upshadow_raw"] = (
        (daily["high"] - body_top) / (daily["high"] - daily["low"] + eps)
    ).clip(lower=0.0)
    daily["upshadow_20"] = daily.groupby("stock_code")["upshadow_raw"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )
    upshadow_monthly = (
        daily.groupby(["stock_code", "year_month"])["upshadow_20"]
        .last()
        .reset_index()
        .rename(columns={"upshadow_20": "UPSHADOW_20"})
    )

    # ---- VOL_SPIKE_6M: 当月日均成交额 / 过去6月日均成交额均值 - 1 ----
    print("[risk] 计算 VOL_SPIKE_6M...")
    monthly_to = (
        daily.groupby(["stock_code", "year_month"])["turnover"]
        .mean()
        .reset_index()
        .rename(columns={"turnover": "avg_turnover_m"})
        .sort_values(["stock_code", "year_month"])
    )
    monthly_to["to_mean_6m"] = monthly_to.groupby("stock_code")["avg_turnover_m"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=3).mean()
    )
    monthly_to["VOL_SPIKE_6M"] = (
        monthly_to["avg_turnover_m"] / monthly_to["to_mean_6m"].replace(0, np.nan) - 1
    )
    vol_spike = monthly_to[["stock_code", "year_month", "VOL_SPIKE_6M"]]

    # ---- RET_6M: 过去6个月累计对数收益率 ----
    print("[risk] 计算 RET_6M...")
    monthly_ret = (
        daily.groupby(["stock_code", "year_month"])["daily_ret"]
        .apply(lambda x: np.log1p(x).sum())  # 月度对数收益
        .reset_index()
        .rename(columns={"daily_ret": "log_ret_m"})
        .sort_values(["stock_code", "year_month"])
    )
    monthly_ret["RET_6M_log"] = monthly_ret.groupby("stock_code")["log_ret_m"].transform(
        lambda x: x.rolling(6, min_periods=5).sum()
    )
    monthly_ret["RET_6M"] = np.expm1(monthly_ret["RET_6M_log"])
    ret_6m = monthly_ret[["stock_code", "year_month", "RET_6M"]]

    # ---- 合并 ----
    result = bias_monthly.merge(upshadow_monthly, on=["stock_code", "year_month"], how="outer")
    result = result.merge(vol_spike, on=["stock_code", "year_month"], how="outer")
    result = result.merge(ret_6m, on=["stock_code", "year_month"], how="outer")
    result = result.sort_values(["stock_code", "year_month"]).reset_index(drop=True)

    return result


def build_top_risk_score(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    对原始风控因子面板执行：MAD截尾 → Z-score → 等权合成 → 整体shift(1)。

    Parameters
    ----------
    raw_df : pd.DataFrame
        含 stock_code, year_month, BIAS_20, UPSHADOW_20, VOL_SPIKE_6M, RET_6M

    Returns
    -------
    pd.DataFrame
        含 stock_code, year_month, TOP_RISK_SCORE
        year_month=t 的风控分来自 t-1 月末数据（shift后）
    """
    df = raw_df.copy()
    periods = sorted(df["year_month"].unique())
    processed_parts = []

    for ym in periods:
        mask = df["year_month"] == ym
        cross = df.loc[mask].copy()

        for col in RISK_FACTOR_COLS:
            if col not in cross.columns:
                continue
            cross[col] = winsorize_mad(cross[col])
            cross[col] = standardize(cross[col])
            cross[col] = cross[col].fillna(0.0)  # 缺失 → 横截面中性（均值=0）

        processed_parts.append(cross)

    result = pd.concat(processed_parts, ignore_index=True)

    # 等权合成
    available = [c for c in RISK_FACTOR_COLS if c in result.columns]
    result["TOP_RISK_SCORE"] = result[available].mean(axis=1)

    # 整体 shift(1)：year_month=t 的风控分 = 用 t-1 月末数据计算的值
    result = result.sort_values(["stock_code", "year_month"])
    result["TOP_RISK_SCORE"] = result.groupby("stock_code")["TOP_RISK_SCORE"].shift(1)

    output = result[["stock_code", "year_month", "TOP_RISK_SCORE"]].copy()
    return output


def build_risk_factors() -> pd.DataFrame:
    """构造月频风控因子面板并返回。"""
    raw_df = _build_raw_risk_factors()

    print("[risk] 原始因子统计:")
    for col in RISK_FACTOR_COLS:
        nan_pct = raw_df[col].isna().mean()
        med = raw_df[col].median()
        print(f"  {col}: NaN={nan_pct:.1%}, median={med:.4f}" if pd.notna(med) else f"  {col}: NaN={nan_pct:.1%}")

    risk_panel = build_top_risk_score(raw_df)

    nan_pct = risk_panel["TOP_RISK_SCORE"].isna().mean()
    print(f"[risk] TOP_RISK_SCORE: NaN={nan_pct:.1%} (首月因shift为NaN，正常)")
    print(f"[risk] 时间范围: {risk_panel['year_month'].min()} ~ {risk_panel['year_month'].max()}")
    print(f"[risk] 总行数: {len(risk_panel)}, 股票数: {risk_panel['stock_code'].nunique()}")

    return risk_panel


def build() -> pd.DataFrame:
    """对外接口，与其他 factors/*.py 模块签名一致。"""
    return build_risk_factors()


if __name__ == "__main__":
    risk_panel = build_risk_factors()
    out_path = os.path.join(PROCESSED_DIR, "factor_risk.parquet")
    risk_panel.to_parquet(out_path, index=False)
    print(f"[risk] 已保存至 {out_path}")
