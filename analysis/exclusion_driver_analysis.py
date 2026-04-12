# -*- coding: utf-8 -*-
"""
剔除驱动因子分析

目的：
  按主导风控因子（BIAS_20、UPSHADOW_20、VOL_SPIKE_6M、RET_6M）分组，
  统计被剔除股票后续 1 个月和 3 个月平均收益，判断各因子的拦截有效性。

  若某类主导因子对应的后续收益为负，说明该因子确实识别了高风险票（拦截有效）；
  若后续收益为正（特别是 VOL_SPIKE_6M），说明该因子可能过度拦截了上升趋势中的票。

输入：
  output/ablation/excluded_stocks_log.csv   — 被剔除股票记录
  data/processed/factor_risk_detail.parquet — 4个子因子 z-score（需先运行 factors/risk.py）
  data/processed/factor_panel.parquet       — ret_next_month（用于计算后续收益）

输出：
  output/ablation/exclusion_driver_analysis.csv
    列：dominant_factor, count, mean_ret_1m, mean_ret_3m,
        pct_negative_1m, pct_negative_3m

Usage:
    python analysis/exclusion_driver_analysis.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, PROCESSED_DIR

SAVE_DIR = os.path.join(OUTPUT_DIR, "ablation")
Z_COLS   = ["BIAS_20_z", "UPSHADOW_20_z", "VOL_SPIKE_6M_z", "RET_6M_z"]


def load_excluded_log() -> pd.DataFrame:
    path = os.path.join(SAVE_DIR, "excluded_stocks_log.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"剔除记录不存在: {path}\n请先运行: python analysis/ablation_top_risk.py"
        )
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    # 去重：同一 (month, stock_code) 可能因多策略同时剔除而重复
    df = df.drop_duplicates(subset=["month", "stock_code"]).reset_index(drop=True)
    print(f"[driver] 剔除记录加载：{len(df)} 条（去重后）")
    return df


def load_risk_detail() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "factor_risk_detail.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"子因子详情不存在: {path}\n请先运行: python factors/risk.py"
        )
    df = pd.read_parquet(path)
    df["stock_code"] = df["stock_code"].astype(str)
    if not hasattr(df["year_month"].dtype, "freq"):
        df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df[["stock_code", "year_month"] + Z_COLS]


def load_ret_panel() -> dict[tuple, float]:
    """加载 factor_panel 的 ret_next_month，构建 (stock_code, year_month) → ret 的字典。"""
    path = os.path.join(PROCESSED_DIR, "factor_panel.parquet")
    df = pd.read_parquet(path, columns=["stock_code", "year_month", "ret_next_month"])
    df["stock_code"] = df["stock_code"].astype(str)
    if not hasattr(df["year_month"].dtype, "freq"):
        df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    df = df.dropna(subset=["ret_next_month"])
    return dict(zip(zip(df["stock_code"], df["year_month"]), df["ret_next_month"]))


def identify_dominant_factor(row: pd.Series) -> str:
    """返回 z-score 绝对值最大的因子名（去掉 _z 后缀）。"""
    scores = {col: abs(row[col]) for col in Z_COLS if pd.notna(row[col])}
    if not scores:
        return "UNKNOWN"
    dominant_col = max(scores, key=scores.get)
    return dominant_col.replace("_z", "")


def compute_ret_3m(sc: str, month: pd.Period, ret_map: dict) -> float:
    """复利计算 month+1 至 month+3 的 3 个月累计收益（任一缺失则返回 NaN）。"""
    r1 = ret_map.get((sc, month), np.nan)
    r2 = ret_map.get((sc, month + 1), np.nan)
    r3 = ret_map.get((sc, month + 2), np.nan)
    if any(np.isnan(x) for x in [r1, r2, r3]):
        return np.nan
    return (1 + r1) * (1 + r2) * (1 + r3) - 1


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    excluded = load_excluded_log()
    detail   = load_risk_detail()
    ret_map  = load_ret_panel()

    # Join z-scores
    merged = excluded.merge(
        detail.rename(columns={"year_month": "month"}),
        on=["stock_code", "month"],
        how="left",
    )

    # 识别主导因子
    merged["dominant_factor"] = merged.apply(identify_dominant_factor, axis=1)

    # 计算后续收益
    merged["ret_1m"] = merged.apply(
        lambda r: ret_map.get((r["stock_code"], r["month"]), np.nan), axis=1
    )
    merged["ret_3m"] = merged.apply(
        lambda r: compute_ret_3m(r["stock_code"], r["month"], ret_map), axis=1
    )

    print(f"\n[driver] 主导因子分布:")
    print(merged["dominant_factor"].value_counts().to_string())

    # 分组统计
    def group_stats(df: pd.DataFrame, label: str) -> dict:
        return {
            "dominant_factor": label,
            "count":           len(df),
            "mean_ret_1m":     np.nanmean(df["ret_1m"]) if len(df) > 0 else np.nan,
            "mean_ret_3m":     np.nanmean(df["ret_3m"]) if len(df) > 0 else np.nan,
            "pct_negative_1m": (df["ret_1m"].dropna() < 0).mean() if df["ret_1m"].notna().any() else np.nan,
            "pct_negative_3m": (df["ret_3m"].dropna() < 0).mean() if df["ret_3m"].notna().any() else np.nan,
        }

    rows = []
    for factor in ["BIAS_20", "UPSHADOW_20", "VOL_SPIKE_6M", "RET_6M", "UNKNOWN"]:
        sub = merged[merged["dominant_factor"] == factor]
        if len(sub) > 0:
            rows.append(group_stats(sub, factor))

    # ALL 行（全量，含 UNKNOWN）
    rows.append(group_stats(merged, "ALL"))

    result = pd.DataFrame(rows)

    # 格式化输出
    print("\n" + "=" * 72)
    print("  剔除驱动因子分析 — 按主导因子分组的后续平均收益")
    print("  （ret>0 表示被剔除后股价继续上涨，即该因子可能过度拦截）")
    print("=" * 72)
    for _, row in result.iterrows():
        flag = " ← 注意" if row["dominant_factor"] == "VOL_SPIKE_6M" else ""
        print(
            f"  {row['dominant_factor']:<18}  n={int(row['count']):>4}"
            f"  ret_1m={row['mean_ret_1m']:+.2%}  ret_3m={row['mean_ret_3m']:+.2%}"
            f"  neg_1m={row['pct_negative_1m']:.1%}"
            f"{flag}"
        )

    out_path = os.path.join(SAVE_DIR, "exclusion_driver_analysis.csv")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[driver] 已保存: {out_path}")


if __name__ == "__main__":
    main()
