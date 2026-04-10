# -*- coding: utf-8 -*-
"""
价值因子: EP (Earnings/Price), BP (Book/Price), SP (Sales/Price)

数据来源:
    - profit_sheet: PARENT_NETPROFIT (归母净利润), OPERATE_INCOME (营业收入)
    - balance_sheet: TOTAL_PARENT_EQUITY (归母净资产)
    - daily_prices: 计算月末市值 (turnover/volume * outstanding_share)

Usage:
    python factors/value.py
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factors.utils import (
    load_financial_statements, load_daily_prices, load_monthly_panel,
    compute_ttm, align_to_monthly, compute_monthly_market_cap,
)
from config import PROCESSED_DIR


def build_value_factors():
    """构造价值因子 EP, BP, SP"""
    print("[value] 加载月度面板网格...")
    panel = load_monthly_panel()
    monthly_grid = panel[["stock_code", "year_month"]].drop_duplicates()

    # ------ 利润表: TTM(归母净利润), TTM(营业收入) ------
    print("[value] 加载利润表并计算 TTM...")
    profit = load_financial_statements(
        "profit_sheet", fields=["PARENT_NETPROFIT", "OPERATE_INCOME"]
    )
    profit_ttm = compute_ttm(profit, ["PARENT_NETPROFIT", "OPERATE_INCOME"])
    profit_aligned = align_to_monthly(
        profit_ttm, monthly_grid, ["PARENT_NETPROFIT", "OPERATE_INCOME"]
    )

    # ------ 资产负债表: 归母净资产 (时点值, 不需要TTM) ------
    print("[value] 加载资产负债表...")
    balance = load_financial_statements(
        "balance_sheet", fields=["TOTAL_PARENT_EQUITY"]
    )
    balance_aligned = align_to_monthly(
        balance, monthly_grid, ["TOTAL_PARENT_EQUITY"]
    )

    # ------ 市值 ------
    print("[value] 计算月末市值...")
    daily = load_daily_prices()
    mkt_cap = compute_monthly_market_cap(daily)

    # ------ 合并 & 计算因子 ------
    print("[value] 合并计算 EP, BP, SP...")
    result = monthly_grid.copy()
    result = result.merge(
        profit_aligned[["stock_code", "year_month", "PARENT_NETPROFIT", "OPERATE_INCOME"]],
        on=["stock_code", "year_month"], how="left"
    )
    result = result.merge(
        balance_aligned[["stock_code", "year_month", "TOTAL_PARENT_EQUITY"]],
        on=["stock_code", "year_month"], how="left"
    )
    result = result.merge(
        mkt_cap[["stock_code", "year_month", "market_cap"]],
        on=["stock_code", "year_month"], how="left"
    )

    # 价值因子 (分母为市值, 需 > 0)
    valid_cap = result["market_cap"] > 0
    result["EP"] = np.where(valid_cap, result["PARENT_NETPROFIT"] / result["market_cap"], np.nan)
    result["BP"] = np.where(valid_cap, result["TOTAL_PARENT_EQUITY"] / result["market_cap"], np.nan)
    result["SP"] = np.where(valid_cap, result["OPERATE_INCOME"] / result["market_cap"], np.nan)

    output = result[["stock_code", "year_month", "EP", "BP", "SP"]]
    print(f"[value] 完成: {len(output)} 行")
    for col in ["EP", "BP", "SP"]:
        nan_pct = output[col].isna().mean()
        print(f"  {col}: NaN={nan_pct:.1%}, median={output[col].median():.4f}")

    return output


if __name__ == "__main__":
    factors = build_value_factors()
    out_path = os.path.join(PROCESSED_DIR, "factor_value.parquet")
    factors.to_parquet(out_path, index=False)
    print(f"[value] 已保存至 {out_path}")
