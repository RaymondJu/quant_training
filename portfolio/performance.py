# -*- coding: utf-8 -*-
"""Performance metrics for monthly portfolio returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return (1 + returns).prod() ** (12 / len(returns)) - 1


def _annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    return returns.std() * np.sqrt(12)


def _max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    nav = (1 + returns).cumprod()
    return (nav / nav.cummax() - 1).min()


def summarize_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
) -> pd.Series:
    """Summarize the main risk and return metrics of a monthly strategy."""
    strategy_returns = strategy_returns.dropna()
    ann_ret = _annualized_return(strategy_returns)
    ann_vol = _annualized_volatility(strategy_returns)
    sharpe = ann_ret / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan
    max_dd = _max_drawdown(strategy_returns)

    summary = {
        "Ann_Return": ann_ret,
        "Ann_Vol": ann_vol,
        "Sharpe": sharpe,
        "Max_Drawdown": max_dd,
        "Monthly_WinRate": (strategy_returns > 0).mean() if not strategy_returns.empty else np.nan,
        "Months": len(strategy_returns),
    }

    if turnover is not None:
        summary["Avg_Turnover"] = turnover.dropna().mean()

    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.reindex(strategy_returns.index).dropna()
        aligned_strategy = strategy_returns.reindex(benchmark_returns.index).dropna()
        benchmark_returns = benchmark_returns.reindex(aligned_strategy.index)
        excess = aligned_strategy - benchmark_returns

        summary["Benchmark_Ann_Return"] = _annualized_return(benchmark_returns)
        summary["Excess_Ann_Return"] = _annualized_return(excess)
        tracking_error = _annualized_volatility(excess)
        summary["Tracking_Error"] = tracking_error
        summary["Info_Ratio"] = (
            summary["Excess_Ann_Return"] / tracking_error
            if pd.notna(tracking_error) and tracking_error > 0
            else np.nan
        )

    return pd.Series(summary)
