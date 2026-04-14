# -*- coding: utf-8 -*-
"""
时变 Universe 模块: 基于成分股纳入日期过滤

CSI300 每年 6 月和 12 月调整成分股。理想做法是用完整的历史成分变更记录
（纳入+剔除），但 AKShare 仅提供当前成分及其纳入日期，无法获取已被剔除的股票。

当前实现 (partial fix):
  - 只纳入 entry_date <= 月末 的股票（排除尚未被纳入指数的股票）
  - 无法排除已被剔除但当前仍在列表中的股票（因数据源限制）
  - 这是一个保守的 partial fix: 消除了"未来纳入"偏差，但保留了"已剔除不排除"偏差

影响评估:
  - 280 只当前成分股中，181 只在 2015-07 之后纳入
  - 回测早期（2015-2017）的有效 universe 会显著缩小
  - 这比完全不处理更诚实，但需在文档中说明局限性

Usage:
    from data.universe import get_universe_at_month
    stocks = get_universe_at_month(pd.Period("2018-06", freq="M"))
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR


@lru_cache(maxsize=1)
def _load_constituents() -> pd.DataFrame:
    """加载成分股列表（含纳入日期），缓存避免重复 IO。"""
    path = os.path.join(RAW_DIR, "hs300_stocks.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"成分股文件不存在: {path}\n请先运行: python data/download.py"
        )
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].str.zfill(6)
    df["entry_date"] = pd.to_datetime(df["date"])
    # 转为 entry_month (Period)，使用月初对齐
    df["entry_month"] = df["entry_date"].dt.to_period("M")
    return df[["stock_code", "stock_name", "entry_date", "entry_month"]]


def get_universe_at_month(year_month: pd.Period) -> set[str]:
    """
    返回指定月份的可投资 universe（成分股集合）。

    逻辑: 只包含 entry_date <= year_month 月末 的股票。
    即: 该股票在 year_month 或之前已被纳入 CSI300。

    Parameters
    ----------
    year_month : pd.Period (freq='M')

    Returns
    -------
    set[str] : 6 位股票代码集合
    """
    constituents = _load_constituents()
    mask = constituents["entry_month"] <= year_month
    return set(constituents.loc[mask, "stock_code"])


def get_all_stock_codes() -> set[str]:
    """返回当前全部成分股代码（不做时间过滤）。"""
    return set(_load_constituents()["stock_code"])


if __name__ == "__main__":
    # 验证: 各年份 universe 大小
    import numpy as np
    test_months = [pd.Period(f"{y}-06", freq="M") for y in range(2015, 2026)]
    print("Universe size by year:")
    for ym in test_months:
        u = get_universe_at_month(ym)
        print(f"  {ym}: {len(u)} stocks")
