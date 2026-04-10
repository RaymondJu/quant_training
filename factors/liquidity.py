# -*- coding: utf-8 -*-
"""
流动性因子: TURN_1M (月均换手率), AMIHUD (Amihud非流动性)

数据来源:
    - daily_prices: turnover_rate, turnover, pct_change

Usage:
    python factors/liquidity.py
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factors.utils import load_daily_prices
from config import PROCESSED_DIR


def build_liquidity_factors():
    """构造流动性因子 TURN_1M, AMIHUD"""
    print("[liquidity] 加载日频数据...")
    daily = load_daily_prices()
    daily["year_month"] = daily["date"].dt.to_period("M")
    daily["daily_ret"] = daily["pct_change"] / 100

    # ------ TURN_1M: 月均换手率 ------
    print("[liquidity] 计算 TURN_1M...")
    turn = daily.groupby(["stock_code", "year_month"])["turnover_rate"].mean().reset_index()
    turn.columns = ["stock_code", "year_month", "TURN_1M"]

    # ------ AMIHUD: mean(|daily_ret| / turnover) ------
    print("[liquidity] 计算 AMIHUD...")
    # 排除零成交日 (停牌等)
    valid = daily[daily["turnover"] > 0].copy()
    valid["amihud_daily"] = valid["daily_ret"].abs() / valid["turnover"]

    amihud = valid.groupby(
        ["stock_code", "year_month"]
    )["amihud_daily"].mean().reset_index()
    amihud.columns = ["stock_code", "year_month", "AMIHUD"]

    # ------ 合并 ------
    output = turn.merge(amihud, on=["stock_code", "year_month"], how="outer")
    print(f"[liquidity] 完成: {len(output)} 行")
    for col in ["TURN_1M", "AMIHUD"]:
        nan_pct = output[col].isna().mean()
        print(f"  {col}: NaN={nan_pct:.1%}, median={output[col].median():.6e}")

    return output


if __name__ == "__main__":
    factors = build_liquidity_factors()
    out_path = os.path.join(PROCESSED_DIR, "factor_liquidity.parquet")
    factors.to_parquet(out_path, index=False)
    print(f"[liquidity] 已保存至 {out_path}")
