# -*- coding: utf-8 -*-
"""
质量因子: ROE_TTM, GPM_change (毛利率同比变化)

数据来源:
    - profit_sheet: PARENT_NETPROFIT, OPERATE_INCOME, OPERATE_COST
    - balance_sheet: TOTAL_PARENT_EQUITY

注意: 金融类股票(~48只) 缺少 OPERATE_COST, GPM_change 为 NaN

Usage:
    python factors/quality.py
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factors.utils import (
    load_financial_statements, load_monthly_panel,
    compute_ttm, align_to_monthly,
)
from config import PROCESSED_DIR


def build_quality_factors():
    """构造质量因子 ROE_TTM, GPM_change"""
    print("[quality] 加载月度面板网格...")
    panel = load_monthly_panel()
    monthly_grid = panel[["stock_code", "year_month"]].drop_duplicates()

    # ------ 利润表 TTM ------
    print("[quality] 加载利润表并计算 TTM...")
    profit = load_financial_statements(
        "profit_sheet", fields=["PARENT_NETPROFIT", "OPERATE_INCOME", "OPERATE_COST"]
    )
    profit_ttm = compute_ttm(profit, ["PARENT_NETPROFIT", "OPERATE_INCOME", "OPERATE_COST"])
    profit_aligned = align_to_monthly(
        profit_ttm, monthly_grid, ["PARENT_NETPROFIT", "OPERATE_INCOME", "OPERATE_COST"]
    )

    # ------ 资产负债表 ------
    print("[quality] 加载资产负债表...")
    balance = load_financial_statements(
        "balance_sheet", fields=["TOTAL_PARENT_EQUITY"]
    )
    balance_aligned = align_to_monthly(
        balance, monthly_grid, ["TOTAL_PARENT_EQUITY"]
    )

    # ------ 合并 ------
    result = monthly_grid.copy()
    result = result.merge(
        profit_aligned[["stock_code", "year_month", "PARENT_NETPROFIT", "OPERATE_INCOME", "OPERATE_COST"]],
        on=["stock_code", "year_month"], how="left"
    )
    result = result.merge(
        balance_aligned[["stock_code", "year_month", "TOTAL_PARENT_EQUITY"]],
        on=["stock_code", "year_month"], how="left"
    )

    # ------ ROE_TTM ------
    valid_equity = result["TOTAL_PARENT_EQUITY"].abs() > 0
    result["ROE_TTM"] = np.where(
        valid_equity,
        result["PARENT_NETPROFIT"] / result["TOTAL_PARENT_EQUITY"],
        np.nan,
    )

    # ------ GPM (毛利率 TTM) ------
    valid_income = result["OPERATE_INCOME"].abs() > 0
    result["GPM_TTM"] = np.where(
        valid_income,
        (result["OPERATE_INCOME"] - result["OPERATE_COST"]) / result["OPERATE_INCOME"],
        np.nan,
    )

    # GPM_change: 毛利率同比变化 (当月 GPM_TTM - 12个月前 GPM_TTM)
    result = result.sort_values(["stock_code", "year_month"])
    result["GPM_TTM_lag12"] = result.groupby("stock_code")["GPM_TTM"].shift(12)
    result["GPM_change"] = result["GPM_TTM"] - result["GPM_TTM_lag12"]

    output = result[["stock_code", "year_month", "ROE_TTM", "GPM_change"]]
    print(f"[quality] 完成: {len(output)} 行")
    for col in ["ROE_TTM", "GPM_change"]:
        nan_pct = output[col].isna().mean()
        med = output[col].median()
        print(f"  {col}: NaN={nan_pct:.1%}, median={med:.4f}" if pd.notna(med) else f"  {col}: NaN={nan_pct:.1%}")

    return output


if __name__ == "__main__":
    factors = build_quality_factors()
    out_path = os.path.join(PROCESSED_DIR, "factor_quality.parquet")
    factors.to_parquet(out_path, index=False)
    print(f"[quality] 已保存至 {out_path}")
