# -*- coding: utf-8 -*-
"""
因子构造共享工具
- 财务报表加载
- TTM计算（利润表累计值→滚动12个月）
- Point-in-time 对齐（按 NOTICE_DATE 避免 look-ahead bias）
"""
import os
import sys
import numpy as np
import pandas as pd
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, PROCESSED_DIR


# =====================================================
#  数据加载
# =====================================================

def load_financial_statements(sheet_type, fields=None):
    """
    批量读取原始财务报表 parquet 文件

    Parameters
    ----------
    sheet_type : str
        "balance_sheet" 或 "profit_sheet"
    fields : list[str], optional
        需要保留的字段名。始终会保留 REPORT_DATE, NOTICE_DATE, stock_code。

    Returns
    -------
    pd.DataFrame
    """
    data_dir = os.path.join(RAW_DIR, sheet_type)
    files = glob(os.path.join(data_dir, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 {sheet_type} 数据，请先运行 data/download.py")

    all_dfs = []
    for f in files:
        df = pd.read_parquet(f)
        code = os.path.basename(f).replace(".parquet", "")
        df["stock_code"] = code
        all_dfs.append(df)

    result = pd.concat(all_dfs, ignore_index=True)
    result["REPORT_DATE"] = pd.to_datetime(result["REPORT_DATE"], errors="coerce")
    result["NOTICE_DATE"] = pd.to_datetime(result["NOTICE_DATE"], errors="coerce")

    # 保留指定字段
    must_keep = ["stock_code", "REPORT_DATE", "NOTICE_DATE"]
    if fields is not None:
        keep = must_keep + [c for c in fields if c in result.columns and c not in must_keep]
        result = result[keep]

    result = result.dropna(subset=["REPORT_DATE"]).sort_values(
        ["stock_code", "REPORT_DATE"]
    ).reset_index(drop=True)
    return result


def load_daily_prices():
    """加载 processed/daily_prices.parquet"""
    path = os.path.join(PROCESSED_DIR, "daily_prices.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_monthly_panel():
    """加载 processed/monthly_panel.parquet"""
    path = os.path.join(PROCESSED_DIR, "monthly_panel.parquet")
    df = pd.read_parquet(path)
    if "year_month" not in df.columns and "year_month_str" in df.columns:
        df["year_month"] = pd.PeriodIndex(df["year_month_str"], freq="M")
    return df


# =====================================================
#  TTM 计算（利润表累计值 → 滚动12个月）
# =====================================================

def compute_ttm(df, value_cols):
    """
    将利润表的累计值(YTD)转换为 TTM (Trailing Twelve Months)

    利润表数据是从年初累计到报告期的值：
    - Q1报告(3月底): 1-3月累计
    - Q2报告(6月底): 1-6月累计
    - Q3报告(9月底): 1-9月累计
    - 年报(12月底): 1-12月累计 = 全年值

    TTM公式:
    - 年报:   TTM = 年报值
    - Q1/Q2/Q3: TTM = 当期累计 + 上年年报 - 上年同期累计

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 stock_code, REPORT_DATE 和 value_cols 中的列
    value_cols : list[str]
        需要计算 TTM 的列名

    Returns
    -------
    pd.DataFrame
        原始列被替换为 TTM 值，新增 report_quarter 列
    """
    df = df.copy()
    df["report_year"] = df["REPORT_DATE"].dt.year
    df["report_month"] = df["REPORT_DATE"].dt.month

    # 标准报告期: 3, 6, 9, 12月
    df = df[df["report_month"].isin([3, 6, 9, 12])].copy()
    df["report_quarter"] = df["report_month"] // 3  # 1, 2, 3, 4

    results = []
    for code, grp in df.groupby("stock_code"):
        grp = grp.sort_values("REPORT_DATE").copy()
        # 构建 (year, quarter) → value 的映射
        for col in value_cols:
            lookup = grp.set_index(["report_year", "report_quarter"])[col].to_dict()
            ttm_vals = []
            for _, row in grp.iterrows():
                y, q = int(row["report_year"]), int(row["report_quarter"])
                if q == 4:  # 年报
                    ttm_vals.append(row[col])
                else:
                    annual_prev = lookup.get((y - 1, 4), np.nan)
                    same_q_prev = lookup.get((y - 1, q), np.nan)
                    current = row[col]
                    if pd.notna(annual_prev) and pd.notna(same_q_prev) and pd.notna(current):
                        ttm_vals.append(current + annual_prev - same_q_prev)
                    else:
                        ttm_vals.append(np.nan)
            grp[col] = ttm_vals
        results.append(grp)

    if not results:
        return df.iloc[:0]
    return pd.concat(results, ignore_index=True)


# =====================================================
#  Point-in-Time 对齐
# =====================================================

def align_to_monthly(fin_df, monthly_dates, value_cols):
    """
    按 NOTICE_DATE 将财务数据对齐到月度网格（point-in-time）

    对每只股票的每个月，取 NOTICE_DATE <= 该月月末 的最新一条财务数据。

    Parameters
    ----------
    fin_df : pd.DataFrame
        包含 stock_code, NOTICE_DATE 和 value_cols
    monthly_dates : pd.DataFrame
        包含 stock_code, year_month 的月度网格
    value_cols : list[str]
        需要对齐的字段

    Returns
    -------
    pd.DataFrame
        以 (stock_code, year_month) 为索引的对齐后数据
    """
    fin = fin_df.dropna(subset=["NOTICE_DATE"]).copy()
    fin["year_month"] = fin["NOTICE_DATE"].dt.to_period("M")

    # 同一个 (stock_code, year_month) 取 REPORT_DATE 最新的
    fin = fin.sort_values(["stock_code", "year_month", "REPORT_DATE"])
    fin = fin.groupby(["stock_code", "year_month"]).last().reset_index()

    keep = ["stock_code", "year_month"] + [c for c in value_cols if c in fin.columns]
    fin = fin[keep]

    # 合并到月度网格并 forward fill
    merged = monthly_dates.merge(fin, on=["stock_code", "year_month"], how="left")
    merged = merged.sort_values(["stock_code", "year_month"])
    for col in value_cols:
        if col in merged.columns:
            merged[col] = merged.groupby("stock_code")[col].ffill()

    return merged


# =====================================================
#  市值计算
# =====================================================

def compute_monthly_market_cap(daily_df):
    """
    计算月末市值

    daily_prices 的 close 是后复权价，不能跨股票比较。
    使用 turnover / volume 作为非复权 VWAP 近似，乘以 outstanding_share 得到市值。

    Returns
    -------
    pd.DataFrame
        columns: stock_code, year_month, market_cap, ln_market_cap
    """
    df = daily_df.copy()
    df["year_month"] = df["date"].dt.to_period("M")

    # 非复权价格近似
    df["unadj_price"] = np.where(df["volume"] > 0, df["turnover"] / df["volume"], np.nan)
    # forward fill 零成交日
    df["unadj_price"] = df.groupby("stock_code")["unadj_price"].ffill()
    df["market_cap"] = df["unadj_price"] * df["outstanding_share"]

    # 取月末最后一个交易日
    month_end = df.groupby(["stock_code", "year_month"]).last().reset_index()
    result = month_end[["stock_code", "year_month", "market_cap"]].copy()
    result["ln_market_cap"] = np.log(result["market_cap"].clip(lower=1))

    return result
