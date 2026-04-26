# -*- coding: utf-8 -*-
"""
因子预处理: 去极值 → 标准化 → 行业中性化 → (可选)市值中性化

每个截面期(year_month)独立处理。

Pipeline:
    1. MAD去极值: median +/- 3 * 1.4826 * MAD
    2. Z-score标准化
    3. 行业中性化: OLS回归申万一级行业哑变量, 取残差
    4. 市值中性化(可选): 加入 ln(市值) 回归
    5. NaN填充: 预处理后剩余NaN填0

Usage:
    python factors/preprocess.py
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factors.utils import load_monthly_panel, load_daily_prices, compute_monthly_market_cap
from config import PROCESSED_DIR, MAD_MULTIPLIER

FACTOR_COLS = ["EP", "BP", "SP", "MOM_12_1", "REV_1M", "ROE_TTM",
               "GPM_change", "VOL_20D", "IVOL", "TURN_1M", "AMIHUD",
               "SIZE", "BETA_60D", "ABTURN_1M", "OCF_QUALITY", "ASSET_GROWTH"]


def load_all_factors():
    """加载所有因子并合并到月度面板"""
    panel = load_monthly_panel()
    base_cols = ["stock_code", "year_month", "ret_next_month", "industry"]
    base = panel[base_cols].copy()

    factor_files = {
        "factor_value.parquet": ["EP", "BP", "SP"],
        "factor_momentum.parquet": ["MOM_12_1", "REV_1M"],
        "factor_quality.parquet": ["ROE_TTM", "GPM_change"],
        "factor_volatility.parquet": ["VOL_20D", "IVOL"],
        "factor_liquidity.parquet": ["TURN_1M", "AMIHUD"],
        "factor_additional.parquet": ["SIZE", "BETA_60D", "ABTURN_1M", "OCF_QUALITY", "ASSET_GROWTH"],
    }

    for fname, cols in factor_files.items():
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            print(f"[WARN] 未找到 {fname}, 跳过")
            continue
        df = pd.read_parquet(path)
        if "year_month" not in df.columns and "year_month_str" in df.columns:
            df["year_month"] = pd.PeriodIndex(df["year_month_str"], freq="M")
        merge_cols = ["stock_code", "year_month"] + [c for c in cols if c in df.columns]
        base = base.merge(df[merge_cols], on=["stock_code", "year_month"], how="left")

    return base


def winsorize_mad(series, n_mad=MAD_MULTIPLIER):
    """MAD去极值: median +/- n_mad * 1.4826 * MAD"""
    median = series.median()
    mad = (series - median).abs().median()
    scale = n_mad * 1.4826 * mad
    return series.clip(lower=median - scale, upper=median + scale)


def standardize(series):
    """Z-score标准化"""
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return series * 0
    return (series - mean) / std


def neutralize(df, factor_col, industry_col="industry_l1", mktcap_col=None):
    """
    行业中性化 (+ 可选市值中性化)

    使用 OLS 回归行业哑变量(和ln市值), 取残差
    """
    valid_mask = df[factor_col].notna() & df[industry_col].notna()
    if mktcap_col:
        valid_mask = valid_mask & df[mktcap_col].notna()

    if valid_mask.sum() < 10:
        return df[factor_col]

    y = df.loc[valid_mask, factor_col].values

    # 行业哑变量
    dummies = pd.get_dummies(df.loc[valid_mask, industry_col], dtype=float)
    X = dummies.values

    if mktcap_col:
        lnmc = df.loc[valid_mask, mktcap_col].values.reshape(-1, 1)
        X = np.hstack([X, lnmc])

    # 加截距
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # OLS: beta = (X'X)^-1 X'y, residuals = y - X*beta
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
    except np.linalg.LinAlgError:
        residuals = y - y.mean()

    result = df[factor_col].copy()
    result.loc[valid_mask] = residuals
    return result


def map_industry_level1(industry_series):
    """
    将申万行业代码映射到一级行业

    industry_classification.csv 中的 industry 列可能是:
    - 6位数字代码 (如 "490101") → 取前2位
    - 中文行业名 → 直接使用
    """
    def _map(val):
        if pd.isna(val):
            return "unknown"
        val = str(val).strip()
        if val.isdigit() and len(val) >= 2:
            return val[:2]
        return val

    return industry_series.apply(_map)


def preprocess_factors(neutralize_mktcap=False):
    """
    执行完整的因子预处理 Pipeline

    Parameters
    ----------
    neutralize_mktcap : bool
        是否同时做市值中性化
    """
    print("[preprocess] 加载所有因子...")
    df = load_all_factors()

    # 行业一级分类
    df["industry_l1"] = map_industry_level1(df["industry"])
    n_industries = df["industry_l1"].nunique()
    print(f"[preprocess] 行业一级分类: {n_industries} 个")
    print(f"[preprocess] 行业分布:\n{df['industry_l1'].value_counts().head(10)}")

    # 市值 (如果需要市值中性化)
    mktcap_col = None
    if neutralize_mktcap:
        print("[preprocess] 计算月末市值 (用于市值中性化)...")
        daily = load_daily_prices()
        mkt_cap = compute_monthly_market_cap(daily)
        df = df.merge(mkt_cap[["stock_code", "year_month", "ln_market_cap"]],
                      on=["stock_code", "year_month"], how="left")
        mktcap_col = "ln_market_cap"

    # 确定实际可用的因子列
    available_factors = [c for c in FACTOR_COLS if c in df.columns]
    print(f"[preprocess] 因子列: {available_factors}")

    # 逐截面期处理
    print("[preprocess] 逐截面期执行: 去极值 → 标准化 → 中性化...")
    periods = sorted(df["year_month"].unique())
    processed_parts = []

    for ym in periods:
        mask = df["year_month"] == ym
        cross = df.loc[mask].copy()

        for col in available_factors:
            # 1. 去极值
            cross[col] = winsorize_mad(cross[col])
            # 2. 标准化
            cross[col] = standardize(cross[col])
            # 3. 中性化
            cross[col] = neutralize(cross, col, "industry_l1", mktcap_col)
            # 4. 再标准化 (中性化后重新归一)
            cross[col] = standardize(cross[col])

        processed_parts.append(cross)

    result = pd.concat(processed_parts, ignore_index=True)

    # 5. NaN填充
    for col in available_factors:
        nan_before = result[col].isna().mean()
        result[col] = result[col].fillna(0)
        if nan_before > 0.01:
            print(f"  {col}: 填充 {nan_before:.1%} NaN → 0")

    # 输出列
    output_cols = (["stock_code", "year_month"] + available_factors +
                   ["ret_next_month", "industry", "industry_l1"])
    if mktcap_col and mktcap_col in result.columns:
        output_cols.append(mktcap_col)
    result = result[[c for c in output_cols if c in result.columns]]

    print(f"\n[preprocess] 最终面板: {len(result)} 行, {len(result.columns)} 列")
    print(f"[preprocess] 月份范围: {result['year_month'].min()} ~ {result['year_month'].max()}")

    return result


if __name__ == "__main__":
    factor_panel = preprocess_factors(neutralize_mktcap=False)
    out_path = os.path.join(PROCESSED_DIR, "factor_panel.parquet")
    factor_panel.to_parquet(out_path, index=False)
    print(f"\n[preprocess] 已保存至 {out_path}")
