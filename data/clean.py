# -*- coding: utf-8 -*-
"""
数据清洗 & 面板构建 Pipeline
将下载的原始数据处理为标准面板格式

Pipeline:
    1. 合并日行情 → 月频面板
    2. 剔除次新股 (上市<120交易日)
    3. 计算月度收益率 (后复权)
    4. 合并财务数据 (按公告日对齐, 避免 look-ahead bias)
    5. 合并行业分类
    6. 输出 processed/ 下的 Parquet 面板

Usage:
    python data/clean.py
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, PROCESSED_DIR, MIN_LIST_DAYS, MISSING_THRESHOLD


# =====================================================
#  Step 1: 合并日行情并生成月频面板
# =====================================================
def build_monthly_panel():
    """将个股日行情合并为月频面板"""
    price_dir = os.path.join(RAW_DIR, "daily_prices")
    files = glob(os.path.join(price_dir, "*.parquet"))
    
    if not files:
        raise FileNotFoundError(f"未找到日行情数据, 请先运行 data/download.py")
    
    print(f"[INFO] 正在合并 {len(files)} 只股票的日行情数据...")
    
    all_daily = []
    for f in tqdm(files, desc="读取日行情"):
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"])
        all_daily.append(df)
    
    daily = pd.concat(all_daily, ignore_index=True)
    daily = daily.sort_values(["stock_code", "date"]).reset_index(drop=True)
    
    # 保存合并后的日频数据 (后续因子构造会用)
    daily_out = os.path.join(PROCESSED_DIR, "daily_prices.parquet")
    daily.to_parquet(daily_out, index=False)
    print(f"[INFO] 日频行情合并完成: {len(daily)} 条, 保存至 {daily_out}")
    
    # ------ 生成月频面板 ------
    # 每月最后一个交易日
    daily["year_month"] = daily["date"].dt.to_period("M")
    
    # 月末数据: 取每月最后一个交易日
    month_end = daily.groupby(["stock_code", "year_month"]).last().reset_index()
    month_end = month_end.rename(columns={"date": "trade_date"})
    
    # 月初数据: 取每月第一个交易日
    month_start = daily.groupby(["stock_code", "year_month"]).first().reset_index()
    month_start = month_start[["stock_code", "year_month", "open"]].rename(
        columns={"open": "month_open"}
    )
    
    # 合并
    monthly = month_end.merge(month_start, on=["stock_code", "year_month"], how="left")
    
    # 计算月度收益率 (后复权收盘价)
    monthly = monthly.sort_values(["stock_code", "year_month"]).reset_index(drop=True)
    monthly["ret_monthly"] = monthly.groupby("stock_code")["close"].pct_change()
    
    # 计算下月收益率 (因子检验用)
    monthly["ret_next_month"] = monthly.groupby("stock_code")["ret_monthly"].shift(-1)
    
    # 月均换手率
    monthly_turnover = daily.groupby(
        ["stock_code", "year_month"]
    )["turnover_rate"].mean().reset_index()
    monthly_turnover.columns = ["stock_code", "year_month", "avg_turnover_rate"]
    monthly = monthly.merge(monthly_turnover, on=["stock_code", "year_month"], how="left")
    
    # 月成交量
    monthly_vol = daily.groupby(
        ["stock_code", "year_month"]
    )["volume"].sum().reset_index()
    monthly_vol.columns = ["stock_code", "year_month", "monthly_volume"]
    monthly = monthly.merge(monthly_vol, on=["stock_code", "year_month"], how="left")
    
    print(f"[INFO] 月频面板: {len(monthly)} 条 (股票×月)")
    return daily, monthly


# =====================================================
#  Step 2: 计算每只股票的上市天数, 剔除次新股
# =====================================================
def filter_new_stocks(daily, monthly):
    """剔除上市不满 MIN_LIST_DAYS 交易日的股票-月记录"""
    # 计算每只股票截至该月有多少交易日
    stock_first_date = daily.groupby("stock_code")["date"].min().reset_index()
    stock_first_date.columns = ["stock_code", "first_trade_date"]
    
    monthly = monthly.merge(stock_first_date, on="stock_code", how="left")
    
    # 计算交易天数
    trading_days = daily.groupby(
        ["stock_code", "year_month"]
    )["date"].count().reset_index()
    trading_days.columns = ["stock_code", "year_month", "n_trade_days"]
    
    # 累计交易天数
    trading_days = trading_days.sort_values(["stock_code", "year_month"])
    trading_days["cum_trade_days"] = trading_days.groupby("stock_code")["n_trade_days"].cumsum()
    
    monthly = monthly.merge(
        trading_days[["stock_code", "year_month", "cum_trade_days"]],
        on=["stock_code", "year_month"], how="left"
    )
    
    before = len(monthly)
    monthly = monthly[monthly["cum_trade_days"] >= MIN_LIST_DAYS].reset_index(drop=True)
    after = len(monthly)
    print(f"[INFO] 次新股过滤: {before} → {after} (剔除 {before - after} 条)")
    
    return monthly


# =====================================================
#  Step 3: 合并财务数据 (来自 stock_financial_analysis_indicator)
# =====================================================
def merge_financial_data(monthly):
    """
    合并财务分析指标, 按公告日对齐以避免 look-ahead bias
    注意: stock_financial_analysis_indicator 的第一列为报告期日期
    """
    fin_dir = os.path.join(RAW_DIR, "financial")
    files = glob(os.path.join(fin_dir, "*.parquet"))
    
    if not files:
        print("[WARN] 未找到财务数据, 跳过合并")
        return monthly
    
    print(f"[INFO] 正在合并 {len(files)} 只股票的财务指标...")
    
    all_fin = []
    for f in tqdm(files, desc="读取财务指标"):
        df = pd.read_parquet(f)
        code = os.path.basename(f).replace(".parquet", "")
        
        if df.empty:
            continue
        
        # 第一列为报告期
        report_col = df.columns[0]
        df = df.rename(columns={report_col: "report_date"})
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df["stock_code"] = code
        
        # 简化: 假设公告日 = 报告期 + 1个季度 (保守估计)
        # Q1(03-31)→04-30, Q2(06-30)→08-31, Q3(09-30)→10-31, 年报(12-31)→04-30
        df["announce_date"] = df["report_date"].apply(_estimate_announce_date)
        df["year_month"] = df["announce_date"].dt.to_period("M")
        
        all_fin.append(df)
    
    if not all_fin:
        return monthly
    
    fin_all = pd.concat(all_fin, ignore_index=True)
    
    # 选择关键财务列 (中文列名)
    # 通过列索引来识别, 因为 AKShare 返回的中文列名在不同编码下可能不一致
    # 我们保留所有列, 后续在因子构造时按需选取
    fin_all = fin_all.sort_values(["stock_code", "report_date"]).reset_index(drop=True)
    
    # 使用 forward-fill: 在公告日之后, 该财务数据才可用
    # 将财务数据挂到月频面板上
    # 方法: 对每只股票, 在每个月取最近的已公告的财务数据
    fin_latest = _get_latest_financials(fin_all, monthly)
    
    if fin_latest is not None:
        monthly = monthly.merge(fin_latest, on=["stock_code", "year_month"], how="left")
        print(f"[INFO] 财务数据合并完成")
    
    return monthly


def _estimate_announce_date(report_date):
    """估算公告日 (保守)"""
    if pd.isna(report_date):
        return pd.NaT
    month = report_date.month
    year = report_date.year
    if month == 3:      # Q1 → 4月30日
        return pd.Timestamp(year, 4, 30)
    elif month == 6:    # Q2 → 8月31日
        return pd.Timestamp(year, 8, 31)
    elif month == 9:    # Q3 → 10月31日
        return pd.Timestamp(year, 10, 31)
    elif month == 12:   # 年报 → 次年4月30日
        return pd.Timestamp(year + 1, 4, 30)
    else:
        return report_date + pd.DateOffset(months=3)


def _get_latest_financials(fin_all, monthly):
    """
    对每只股票每个月, 取最新的已公告财务数据
    """
    # 去重: 同一个 stock_code + year_month 的财务数据, 取最新的一条
    fin_dedup = fin_all.sort_values(
        ["stock_code", "year_month", "report_date"]
    ).groupby(["stock_code", "year_month"]).last().reset_index()
    
    # 只保留数值列 + stock_code + year_month
    numeric_cols = fin_dedup.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = ["stock_code", "year_month"] + numeric_cols
    fin_dedup = fin_dedup[[c for c in keep_cols if c in fin_dedup.columns]]
    
    # 为每个 (stock_code, year_month) 做 forward fill
    # 创建完整的月度网格
    all_months = monthly[["stock_code", "year_month"]].drop_duplicates()
    fin_merged = all_months.merge(fin_dedup, on=["stock_code", "year_month"], how="left")
    
    # 按股票 forward fill
    fin_merged = fin_merged.sort_values(["stock_code", "year_month"])
    for col in numeric_cols:
        if col in fin_merged.columns:
            fin_merged[col] = fin_merged.groupby("stock_code")[col].ffill()
    
    return fin_merged


# =====================================================
#  Step 4: 合并行业分类
# =====================================================
def merge_industry(monthly):
    """合并行业分类信息"""
    ind_file = os.path.join(RAW_DIR, "industry_classification.csv")
    
    if not os.path.exists(ind_file):
        print("[WARN] 未找到行业分类数据, 跳过")
        return monthly
    
    ind = pd.read_csv(ind_file, dtype=str)
    # 一只股票可能属于多个行业板块, 只取第一个
    ind_unique = ind.groupby("stock_code").first().reset_index()
    ind_unique = ind_unique[["stock_code", "industry"]]
    
    monthly = monthly.merge(ind_unique, on="stock_code", how="left")
    n_missing = monthly["industry"].isna().sum()
    print(f"[INFO] 行业分类合并完成, 缺失行业: {n_missing} 条")
    
    return monthly


# =====================================================
#  Step 5: 保存清洗后的面板数据
# =====================================================
def save_panel(monthly):
    """保存最终面板数据"""
    out_file = os.path.join(PROCESSED_DIR, "monthly_panel.parquet")
    
    # 转换 year_month 为字符串以兼容 parquet
    monthly["year_month_str"] = monthly["year_month"].astype(str)
    
    # 报告数据质量
    print("\n" + "=" * 50)
    print("  最终面板数据概览")
    print("=" * 50)
    print(f"  总记录数: {len(monthly)}")
    print(f"  股票数: {monthly['stock_code'].nunique()}")
    print(f"  月份范围: {monthly['year_month'].min()} ~ {monthly['year_month'].max()}")
    print(f"  列数: {len(monthly.columns)}")
    
    # 缺失值统计
    missing_pct = monthly.isnull().mean()
    high_missing = missing_pct[missing_pct > MISSING_THRESHOLD]
    if len(high_missing) > 0:
        print(f"\n  [WARN] 以下列缺失率 > {MISSING_THRESHOLD*100}%:")
        for col, pct in high_missing.items():
            print(f"    {col}: {pct:.1%}")
    
    monthly.to_parquet(out_file, index=False)
    print(f"\n  已保存至: {out_file}")
    print("=" * 50)
    
    # 同时保存股票基本信息表
    stock_info = monthly.groupby("stock_code").agg({
        "first_trade_date": "first",
        "industry": "first",
    }).reset_index()
    stock_info_file = os.path.join(PROCESSED_DIR, "stock_info.parquet")
    stock_info.to_parquet(stock_info_file, index=False)
    print(f"[INFO] 股票信息表已保存: {stock_info_file}")


# =====================================================
#  Main
# =====================================================
def main():
    print("=" * 60)
    print("  A股多因子选股项目 — 数据清洗 & 面板构建")
    print("=" * 60)
    
    # Step 1: 合并日行情 → 月频
    print("\n[Step 1] 合并日行情 → 月频面板")
    daily, monthly = build_monthly_panel()
    
    # Step 2: 剔除次新股
    print("\n[Step 2] 剔除次新股")
    monthly = filter_new_stocks(daily, monthly)
    
    # Step 3: 合并财务数据
    print("\n[Step 3] 合并财务指标")
    monthly = merge_financial_data(monthly)
    
    # Step 4: 合并行业分类
    print("\n[Step 4] 合并行业分类")
    monthly = merge_industry(monthly)
    
    # Step 5: 标记时变 universe（基于成分股纳入日期）
    print("\n[Step 5] 标记时变 CSI300 成分 (in_universe)")
    from data.universe import get_universe_at_month
    unique_months = sorted(monthly["year_month"].unique())
    universe_cache = {ym: get_universe_at_month(ym) for ym in unique_months}
    monthly["in_universe"] = monthly.apply(
        lambda r: r["stock_code"] in universe_cache[r["year_month"]], axis=1
    )
    n_in = monthly["in_universe"].sum()
    print(f"[INFO] in_universe=True: {n_in}/{len(monthly)} ({n_in/len(monthly):.1%})")

    # Step 6: 保存
    print("\n[Step 6] 保存面板数据")
    save_panel(monthly)
    
    print("\n数据清洗 Pipeline 完成!")


if __name__ == "__main__":
    main()
