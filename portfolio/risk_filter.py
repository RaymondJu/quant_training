# -*- coding: utf-8 -*-
"""
顶部风险过滤层公共函数

供 portfolio/backtest.py 和 ml/model_comparison.py 共用，
避免重复实现同一逻辑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_top_risk_filter(
    selected_stocks: list[str],
    month,                          # pd.Period
    top_risk_panel: pd.DataFrame,   # cols: stock_code, year_month, TOP_RISK_SCORE
    filter_pct: float,
    log_records: list | None = None,
) -> list[str]:
    """
    从 selected_stocks 中剔除风险得分最高的 filter_pct 比例。

    Parameters
    ----------
    selected_stocks : list[str]
        当月 Top-N 选出的股票代码列表
    month : pd.Period
        当前调仓月（year_month，与 top_risk_panel 的 year_month 对应）
    top_risk_panel : pd.DataFrame
        月频风控面板，已在 factors/risk.py 中 shift(1)，
        year_month=t 的风控分基于 t-1 月末数据
    filter_pct : float
        剔除比例，如 0.20 表示剔除风控分最高的 20%
    log_records : list | None
        若提供，被剔除的股票记录追加到此列表中

    Returns
    -------
    list[str]
        过滤后的股票代码列表。若过滤后为空（极端情况），返回原列表以避免空仓。
    """
    if not selected_stocks or filter_pct <= 0:
        return selected_stocks

    # 取当月风控分
    month_risk = top_risk_panel[top_risk_panel["year_month"] == month][
        ["stock_code", "TOP_RISK_SCORE"]
    ].copy()
    risk_map = dict(zip(month_risk["stock_code"], month_risk["TOP_RISK_SCORE"]))

    # 构建得分 DataFrame
    score_df = pd.DataFrame(
        {"stock_code": selected_stocks,
         "TOP_RISK_SCORE": [risk_map.get(c, np.nan) for c in selected_stocks]}
    )

    # 缺失风控分 → 填横截面中位数（风险中性处理，不直接剔除）
    median_score = score_df["TOP_RISK_SCORE"].median()
    if pd.isna(median_score):
        median_score = 0.0
    score_df["TOP_RISK_SCORE"] = score_df["TOP_RISK_SCORE"].fillna(median_score)

    # 按风控分降序排列，剔除前 filter_pct 比例
    score_df = score_df.sort_values("TOP_RISK_SCORE", ascending=False).reset_index(drop=True)
    n_total = len(score_df)
    n_exclude = max(1, int(n_total * filter_pct))

    excluded = score_df.head(n_exclude)
    kept_df = score_df.iloc[n_exclude:]
    kept = kept_df["stock_code"].tolist()

    # 记录被剔除股票（用于 excluded_stocks_log.csv）
    if log_records is not None:
        for rank, row in enumerate(excluded.itertuples(index=False), start=1):
            log_records.append({
                "month": str(month),
                "stock_code": row.stock_code,
                "top_risk_score": row.TOP_RISK_SCORE,
                "rank_in_topn": rank,
            })

    # 保险机制：若过滤后为空，返回原列表
    return kept if kept else selected_stocks
