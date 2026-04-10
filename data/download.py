# -*- coding: utf-8 -*-
"""
数据下载脚本
从 AKShare 下载沪深300成分股的行情、财务、行业分类等数据
支持断点续传，已下载的数据会跳过

Usage:
    python data/download.py
"""
import os
import sys
import random

# ===== 绕过代理：VPN开着也能直连国内数据接口 =====
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
            'ALL_PROXY', 'all_proxy']:
    os.environ.pop(key, None)
os.environ['NO_PROXY'] = '*'

# Monkey-patch requests，强制所有请求不走代理
import requests as _requests
_original_get = _requests.get
_original_post = _requests.post

def _get_no_proxy(*args, **kwargs):
    kwargs.setdefault('proxies', {'http': '', 'https': ''})
    kwargs.setdefault('timeout', 30)
    return _original_get(*args, **kwargs)

def _post_no_proxy(*args, **kwargs):
    kwargs.setdefault('proxies', {'http': '', 'https': ''})
    kwargs.setdefault('timeout', 30)
    return _original_post(*args, **kwargs)

_requests.get = _get_no_proxy
_requests.post = _post_no_proxy
# ================================================

import time
import traceback
import pandas as pd
import akshare as ak
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DIR, START_DATE, END_DATE, INDEX_CODE,
    DOWNLOAD_SLEEP, MAX_RETRY
)


def sleep_with_jitter(base_seconds):
    """带随机抖动的等待，避免固定间隔被限流"""
    jitter = random.uniform(0.5, 1.5)
    time.sleep(base_seconds * jitter)


def retry_backoff_sleep(retry_num):
    """指数退避等待：3s → 6s → 12s → 24s → ... (最大60s)"""
    wait = min(3 * (2 ** retry_num), 60) + random.uniform(0, 2)
    print(f"  [RETRY] 等待 {wait:.1f}s 后重试 (第{retry_num+1}次)...")
    time.sleep(wait)


def get_hs300_stocks():
    """获取沪深300成分股列表"""
    cache_file = os.path.join(RAW_DIR, "hs300_stocks.csv")
    
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, dtype=str)
        print(f"[INFO] 从缓存加载沪深300成分股: {len(df)} 只")
        return df
    
    print("[INFO] 正在从 AKShare 获取沪深300成分股列表...")
    df = ak.index_stock_cons(symbol=INDEX_CODE)
    # 统一列名
    df.columns = ["stock_code", "stock_name", "date"]
    # 补全6位代码
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    print(f"[INFO] 获取沪深300成分股 {len(df)} 只, 已保存至 {cache_file}")
    return df


def download_daily_prices(stock_list):
    """
    下载日频行情数据（后复权）— 使用网易163数据源
    每只股票存为一个 Parquet 文件, 支持断点续传
    """
    price_dir = os.path.join(RAW_DIR, "daily_prices")
    os.makedirs(price_dir, exist_ok=True)
    
    success, skip, fail = 0, 0, 0
    failed_stocks = []
    
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="下载日行情"):
        code = row["stock_code"]
        out_file = os.path.join(price_dir, f"{code}.parquet")
        
        # 断点续传: 已下载则跳过
        if os.path.exists(out_file):
            skip += 1
            continue
        
        # stock_zh_a_daily 需要 sz/sh 前缀
        if code.startswith("6"):
            symbol = f"sh{code}"
        else:
            symbol = f"sz{code}"
        
        for retry in range(MAX_RETRY):
            try:
                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    adjust="hfq"  # 后复权, 用于计算收益率
                )
                if df is not None and len(df) > 0:
                    # 标准化列名 (163源列: date,open,high,low,close,volume,amount,outstanding_share,turnover)
                    df = df.reset_index() if 'date' not in df.columns else df
                    df = df.rename(columns={
                        'amount': 'turnover',
                        'turnover': 'turnover_rate',
                    })
                    df["date"] = pd.to_datetime(df["date"])
                    df["stock_code"] = code
                    # 计算涨跌幅
                    if "pct_change" not in df.columns:
                        df["pct_change"] = df["close"].pct_change() * 100
                    df.to_parquet(out_file, index=False)
                    success += 1
                else:
                    print(f"\n[WARN] {code} 无数据")
                    fail += 1
                break
            except Exception as e:
                if retry < MAX_RETRY - 1:
                    retry_backoff_sleep(retry)
                else:
                    print(f"\n[ERROR] {code} 下载失败: {e}")
                    failed_stocks.append(code)
                    fail += 1
        
        sleep_with_jitter(DOWNLOAD_SLEEP)
    
    print(f"\n[INFO] 日行情下载完成: 成功={success}, 跳过={skip}, 失败={fail}")
    if failed_stocks:
        print(f"[WARN] 失败股票: {failed_stocks}")
    return failed_stocks


def download_financial_data(stock_list):
    """
    下载财务分析指标数据（新浪源）
    包含: EPS, BPS, ROE, 净利润率, 毛利率等
    """
    fin_dir = os.path.join(RAW_DIR, "financial")
    os.makedirs(fin_dir, exist_ok=True)
    
    success, skip, fail = 0, 0, 0
    failed_stocks = []
    
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="下载财务指标"):
        code = row["stock_code"]
        out_file = os.path.join(fin_dir, f"{code}.parquet")
        
        if os.path.exists(out_file):
            skip += 1
            continue
        
        for retry in range(MAX_RETRY):
            try:
                df = ak.stock_financial_analysis_indicator(
                    symbol=code, start_year="2014"
                )
                if df is not None and len(df) > 0:
                    df.to_parquet(out_file, index=False)
                    success += 1
                else:
                    fail += 1
                break
            except Exception as e:
                if retry < MAX_RETRY - 1:
                    retry_backoff_sleep(retry)
                else:
                    print(f"\n[ERROR] {code} 财务数据下载失败: {e}")
                    failed_stocks.append(code)
                    fail += 1
        
        sleep_with_jitter(DOWNLOAD_SLEEP)
    
    print(f"\n[INFO] 财务指标下载完成: 成功={success}, 跳过={skip}, 失败={fail}")
    return failed_stocks


def download_balance_sheet(stock_list):
    """
    下载资产负债表（东财源）
    用于获取: 净资产(所有者权益), 总资产等
    """
    bs_dir = os.path.join(RAW_DIR, "balance_sheet")
    os.makedirs(bs_dir, exist_ok=True)
    
    success, skip, fail = 0, 0, 0
    
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="下载资产负债表"):
        code = row["stock_code"]
        out_file = os.path.join(bs_dir, f"{code}.parquet")
        
        if os.path.exists(out_file):
            skip += 1
            continue
        
        # 东财接口需要 SH/SZ 前缀
        if code.startswith("6"):
            symbol = f"SH{code}"
        else:
            symbol = f"SZ{code}"
        
        for retry in range(MAX_RETRY):
            try:
                df = ak.stock_balance_sheet_by_report_em(symbol=symbol)
                if df is not None and len(df) > 0:
                    # 只保留关键列以节省空间
                    key_cols = [c for c in df.columns if c in [
                        "SECUCODE", "SECURITY_CODE", "REPORT_DATE_NAME",
                        "REPORT_DATE", "NOTICE_DATE",
                        "TOTAL_ASSETS",          # 总资产
                        "TOTAL_LIABILITIES",     # 总负债
                        "TOTAL_EQUITY",          # 所有者权益合计
                        "TOTAL_PARENT_EQUITY",   # 归属母公司所有者权益
                        "MONETARYFUNDS",         # 货币资金
                        "TOTAL_CURRENT_ASSETS",  # 流动资产合计
                        "TOTAL_NONCURRENT_ASSETS",  # 非流动资产合计
                        "TOTAL_CURRENT_LIAB",    # 流动负债合计
                        "TOTAL_NONCURRENT_LIAB", # 非流动负债合计
                        "INVENTORY",             # 存货
                    ]]
                    if key_cols:
                        df = df[key_cols]
                    df.to_parquet(out_file, index=False)
                    success += 1
                else:
                    fail += 1
                break
            except Exception as e:
                if retry < MAX_RETRY - 1:
                    retry_backoff_sleep(retry)
                else:
                    print(f"\n[ERROR] {code} 资产负债表下载失败: {e}")
                    fail += 1
        
        sleep_with_jitter(DOWNLOAD_SLEEP)
    
    print(f"\n[INFO] 资产负债表下载完成: 成功={success}, 跳过={skip}, 失败={fail}")


def download_profit_sheet(stock_list):
    """
    下载利润表（东财源）
    用于获取: 营业收入, 归母净利润, 毛利等
    """
    ps_dir = os.path.join(RAW_DIR, "profit_sheet")
    os.makedirs(ps_dir, exist_ok=True)
    
    success, skip, fail = 0, 0, 0
    
    for _, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="下载利润表"):
        code = row["stock_code"]
        out_file = os.path.join(ps_dir, f"{code}.parquet")
        
        if os.path.exists(out_file):
            skip += 1
            continue
        
        if code.startswith("6"):
            symbol = f"SH{code}"
        else:
            symbol = f"SZ{code}"
        
        for retry in range(MAX_RETRY):
            try:
                df = ak.stock_profit_sheet_by_report_em(symbol=symbol)
                if df is not None and len(df) > 0:
                    key_cols = [c for c in df.columns if c in [
                        "SECUCODE", "SECURITY_CODE", "REPORT_DATE_NAME",
                        "REPORT_DATE", "NOTICE_DATE",
                        "TOTAL_OPERATE_INCOME",   # 营业总收入
                        "OPERATE_INCOME",          # 营业收入
                        "OPERATE_COST",            # 营业成本
                        "OPERATE_PROFIT",          # 营业利润
                        "TOTAL_PROFIT",            # 利润总额
                        "NETPROFIT",               # 净利润
                        "PARENT_NETPROFIT",        # 归属母公司净利润
                        "OPERATE_TAX_ADD",         # 税金及附加
                        "SALE_EXPENSE",            # 销售费用
                        "MANAGE_EXPENSE",          # 管理费用
                        "FINANCE_EXPENSE",         # 财务费用
                        "RESEARCH_EXPENSE",        # 研发费用
                    ]]
                    if key_cols:
                        df = df[key_cols]
                    df.to_parquet(out_file, index=False)
                    success += 1
                else:
                    fail += 1
                break
            except Exception as e:
                if retry < MAX_RETRY - 1:
                    retry_backoff_sleep(retry)
                else:
                    print(f"\n[ERROR] {code} 利润表下载失败: {e}")
                    fail += 1
        
        sleep_with_jitter(DOWNLOAD_SLEEP)
    
    print(f"\n[INFO] 利润表下载完成: 成功={success}, 跳过={skip}, 失败={fail}")


def download_industry_classification(stock_list):
    """
    下载申万行业分类（通过 sw_index + stock_industry_clf_hist_sw）
    用于行业中性化
    """
    cache_file = os.path.join(RAW_DIR, "industry_classification.csv")
    
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, dtype=str)
        print(f"[INFO] 从缓存加载行业分类: {len(df)} 条记录")
        return df
    
    print("[INFO] 正在下载申万行业分类...")
    
    # Step 1: 获取申万一级行业名称映射 (行业代码 → 行业名称)
    try:
        sw_info = ak.sw_index_first_info()
        # 行业代码格式: 801010.SI → 取前6位 8010
        code_to_name = {}
        for _, r in sw_info.iterrows():
            full_code = str(r.iloc[0]).replace(".SI", "")  # 801010
            name = str(r.iloc[1])
            # 申万一级代码前4位用于匹配 stock_industry_clf_hist_sw 的6位行业代码
            code_to_name[full_code] = name
        print(f"[INFO] 获取到 {len(code_to_name)} 个申万一级行业")
    except Exception as e:
        print(f"[ERROR] 获取申万行业列表失败: {e}")
        code_to_name = {}
    
    # Step 2: 获取全部股票的行业变动历史
    try:
        clf_df = ak.stock_industry_clf_hist_sw()
        # 列: symbol, start_date, industry_code, update_time
        # 每只股票取最新的行业分类
        clf_df["start_date"] = pd.to_datetime(clf_df["start_date"])
        latest = clf_df.sort_values("start_date").groupby("symbol").last().reset_index()
        print(f"[INFO] 获取到 {len(latest)} 只股票的行业分类")
    except Exception as e:
        print(f"[ERROR] 获取行业分类历史失败: {e}")
        latest = pd.DataFrame()
    
    # Step 3: 匹配沪深300成分股
    hs300_codes = set(stock_list["stock_code"].tolist())
    stock_names = dict(zip(stock_list["stock_code"], stock_list.get("stock_name", [""] * len(stock_list))))
    
    all_records = []
    for _, row in stock_list.iterrows():
        code = row["stock_code"]
        name = row.get("stock_name", "")
        industry = "未知"
        
        if len(latest) > 0:
            match = latest[latest["symbol"] == code]
            if len(match) > 0:
                ind_code = str(match.iloc[0]["industry_code"])
                # 行业代码6位(如310201)，前6位对应申万一级代码(如801010)
                # 实际映射: 前4位 → 8010xx
                sw_first = f"80{ind_code[:2]}0"
                industry = code_to_name.get(sw_first, ind_code)
                # Fallback: try matching with full code patterns
                if industry == ind_code:
                    for k, v in code_to_name.items():
                        if ind_code[:2] == k[2:4]:
                            industry = v
                            break
        
        all_records.append({
            "industry": str(industry),
            "stock_code": code,
            "stock_name": name,
        })
    
    df = pd.DataFrame(all_records)
    df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    n_classified = len(df[df["industry"] != "未知"])
    print(f"[INFO] 行业分类完成: {n_classified}/{len(df)} 只股票已分类, 已保存至 {cache_file}")
    print(f"[INFO] 行业分类下载完成: {len(df)} 条记录, 已保存至 {cache_file}")
    return df


def download_index_daily():
    """下载沪深300指数日行情 (用于计算CAPM beta / 特质波动率)"""
    cache_file = os.path.join(RAW_DIR, "index_hs300_daily.parquet")
    
    if os.path.exists(cache_file):
        print("[INFO] 沪深300指数日行情已存在, 跳过")
        return
    
    print("[INFO] 正在下载沪深300指数日行情...")
    df = ak.stock_zh_index_daily(symbol="sh000300")
    if df is not None and len(df) > 0:
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(cache_file, index=False)
        print(f"[INFO] 沪深300指数日行情已保存: {len(df)} 条")
    else:
        print("[ERROR] 沪深300指数日行情下载失败")


def main():
    print("=" * 60)
    print("  A股多因子选股项目 — 数据下载")
    print(f"  范围: 沪深300 | 时间: {START_DATE}–{END_DATE}")
    print(f"  AKShare 版本: {ak.__version__}")
    print("=" * 60)
    
    # Step 1: 获取沪深300成分股
    stock_list = get_hs300_stocks()
    
    # Step 2: 下载日行情 (后复权)
    print("\n" + "=" * 40)
    print("  Step 2: 下载日频行情数据 (后复权)")
    print("=" * 40)
    download_daily_prices(stock_list)
    
    # Step 3: 下载财务分析指标
    print("\n" + "=" * 40)
    print("  Step 3: 下载财务分析指标")
    print("=" * 40)
    download_financial_data(stock_list)
    
    # Step 4: 下载资产负债表
    print("\n" + "=" * 40)
    print("  Step 4: 下载资产负债表")
    print("=" * 40)
    download_balance_sheet(stock_list)
    
    # Step 5: 下载利润表
    print("\n" + "=" * 40)
    print("  Step 5: 下载利润表")
    print("=" * 40)
    download_profit_sheet(stock_list)
    
    # Step 6: 下载行业分类
    print("\n" + "=" * 40)
    print("  Step 6: 下载行业分类")
    print("=" * 40)
    download_industry_classification(stock_list)
    
    # Step 7: 下载沪深300指数行情
    print("\n" + "=" * 40)
    print("  Step 7: 下载沪深300指数日行情")
    print("=" * 40)
    download_index_daily()
    
    print("\n" + "=" * 60)
    print("  数据下载全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
