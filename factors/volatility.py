# -*- coding: utf-8 -*-
"""
波动因子: VOL_20D (20日波动率), IVOL (特质波动率)

数据来源:
    - daily_prices: 日收益率
    - index_hs300_daily: 沪深300日收益率 (CAPM回归用)

Usage:
    python factors/volatility.py
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factors.utils import load_daily_prices
from config import RAW_DIR, PROCESSED_DIR


def _compute_ivol_group(grp):
    """单个 (stock_code, year_month) 组的 IVOL 计算"""
    if len(grp) < 15:
        return np.nan
    y = grp["daily_ret"].values
    x = grp["index_ret"].values
    # 简单 OLS: beta = cov(y,x)/var(x), residuals = y - beta*x
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x == 0:
        return np.nan
    beta = ((x - x_mean) * (y - y_mean)).sum() / var_x
    alpha = y_mean - beta * x_mean
    residuals = y - alpha - beta * x
    return residuals.std(ddof=1)


def build_volatility_factors():
    """构造波动因子 VOL_20D, IVOL"""
    print("[volatility] 加载日频数据...")
    daily = load_daily_prices()
    daily["daily_ret"] = daily["pct_change"] / 100
    daily["year_month"] = daily["date"].dt.to_period("M")

    # ------ VOL_20D: 20日滚动波动率, 取月末值 ------
    print("[volatility] 计算 VOL_20D...")
    daily["vol_20d"] = daily.groupby("stock_code")["daily_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).std()
    )
    vol_monthly = daily.groupby(["stock_code", "year_month"])["vol_20d"].last().reset_index()
    vol_monthly.columns = ["stock_code", "year_month", "VOL_20D"]

    # ------ IVOL: CAPM 残差波动率 ------
    print("[volatility] 加载沪深300指数日收益...")
    from data.benchmark import load_benchmark_daily_returns
    idx_daily = load_benchmark_daily_returns()  # [date, index_ret]

    # 合并股票日收益与指数日收益
    daily_with_idx = daily.merge(
        idx_daily[["date", "index_ret"]], on="date", how="inner"
    )
    daily_with_idx = daily_with_idx.dropna(subset=["daily_ret", "index_ret"])

    print("[volatility] 计算 IVOL (CAPM残差波动率)...")
    ivol = daily_with_idx.groupby(
        ["stock_code", "year_month"]
    ).apply(_compute_ivol_group, include_groups=False).reset_index()
    ivol.columns = ["stock_code", "year_month", "IVOL"]

    # ------ 合并 ------
    output = vol_monthly.merge(ivol, on=["stock_code", "year_month"], how="outer")
    print(f"[volatility] 完成: {len(output)} 行")
    for col in ["VOL_20D", "IVOL"]:
        nan_pct = output[col].isna().mean()
        print(f"  {col}: NaN={nan_pct:.1%}, median={output[col].median():.6f}")

    return output


if __name__ == "__main__":
    factors = build_volatility_factors()
    out_path = os.path.join(PROCESSED_DIR, "factor_volatility.parquet")
    factors.to_parquet(out_path, index=False)
    print(f"[volatility] 已保存至 {out_path}")
