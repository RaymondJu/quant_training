# -*- coding: utf-8 -*-
"""
动量因子: MOM_12_1 (12减1月动量), REV_1M (1个月反转)

数据来源:
    - monthly_panel: ret_monthly (月度收益率)

Usage:
    python factors/momentum.py
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factors.utils import load_monthly_panel
from config import PROCESSED_DIR


def build_momentum_factors():
    """构造动量因子 MOM_12_1, REV_1M"""
    print("[momentum] 加载月度面板...")
    panel = load_monthly_panel()
    df = panel[["stock_code", "year_month", "ret_monthly"]].copy()
    df = df.sort_values(["stock_code", "year_month"]).reset_index(drop=True)

    # 对数收益率 (避免复利偏差)
    df["log_ret"] = np.log1p(df["ret_monthly"])

    # MOM_12_1: 过去12个月累计收益, 剔除最近1个月
    # = sum(log_ret from t-12 to t-2) = rolling_12_sum - current_log_ret
    df["rolling_12_log"] = df.groupby("stock_code")["log_ret"].transform(
        lambda x: x.rolling(12, min_periods=12).sum()
    )
    df["MOM_12_1"] = np.exp(df["rolling_12_log"] - df["log_ret"]) - 1

    # REV_1M: 当月收益率 (短期反转效应)
    df["REV_1M"] = df["ret_monthly"]

    output = df[["stock_code", "year_month", "MOM_12_1", "REV_1M"]]
    print(f"[momentum] 完成: {len(output)} 行")
    for col in ["MOM_12_1", "REV_1M"]:
        nan_pct = output[col].isna().mean()
        print(f"  {col}: NaN={nan_pct:.1%}, median={output[col].median():.4f}")

    return output


if __name__ == "__main__":
    factors = build_momentum_factors()
    out_path = os.path.join(PROCESSED_DIR, "factor_momentum.parquet")
    factors.to_parquet(out_path, index=False)
    print(f"[momentum] 已保存至 {out_path}")
