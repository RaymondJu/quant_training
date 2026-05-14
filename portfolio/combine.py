# -*- coding: utf-8 -*-
"""Factor combination utilities for monthly stock selection."""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from config import ACTIVE_FACTOR_COLS, IC_ROLLING_WINDOW, OUTPUT_DIR, PROCESSED_DIR

DEFAULT_FACTOR_COLS = ACTIVE_FACTOR_COLS


def load_factor_panel(panel_file: str = "factor_panel.parquet") -> pd.DataFrame:
    """Load the processed factor panel used across portfolio modules."""
    path = os.path.join(PROCESSED_DIR, panel_file)
    panel = pd.read_parquet(path).copy()
    panel["year_month"] = pd.PeriodIndex(panel["year_month"], freq="M")
    return panel.sort_values(["year_month", "stock_code"]).reset_index(drop=True)


def get_available_factors(
    panel: pd.DataFrame,
    factor_cols: Iterable[str] | None = None,
) -> list[str]:
    cols = DEFAULT_FACTOR_COLS if factor_cols is None else list(factor_cols)
    return [col for col in cols if col in panel.columns]


def compute_rank_ic_series(
    panel: pd.DataFrame,
    factor_cols: Iterable[str] | None = None,
    ret_col: str = "ret_next_month",
    min_obs: int = 20,
) -> pd.DataFrame:
    """Compute monthly cross-sectional rank IC series for each factor."""
    available = get_available_factors(panel, factor_cols)
    results: dict[str, pd.Series] = {}

    for factor in available:
        values = {}
        for ym, grp in panel.groupby("year_month"):
            valid = grp[[factor, ret_col]].dropna()
            if len(valid) < min_obs:
                continue
            if valid[factor].nunique() <= 1 or valid[ret_col].nunique() <= 1:
                continue
            ic, _ = stats.spearmanr(valid[factor], valid[ret_col])
            if pd.notna(ic):
                values[ym] = ic
        results[factor] = pd.Series(values, name=factor)

    return pd.DataFrame(results).sort_index()


def _normalize_weights(raw_weights: pd.Series, factor_cols: list[str]) -> pd.Series:
    weights = raw_weights.reindex(factor_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = weights.abs().sum()
    if denom == 0:
        return pd.Series(1.0 / len(factor_cols), index=factor_cols)
    return weights / denom


def build_dynamic_factor_weights(
    ic_df: pd.DataFrame,
    method: str = "equal",
    window: int = IC_ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Build factor weights for each rebalance month using only past IC history.

    Methods:
        equal: equal absolute weights, sign decided by trailing mean IC
        ic:    weights proportional to trailing mean IC
        icir:  weights proportional to trailing mean IC / trailing std IC
    """
    factor_cols = list(ic_df.columns)
    month_weights = {}

    for ym in ic_df.index:
        hist = ic_df.loc[ic_df.index < ym].tail(window)
        if hist.empty:
            raw = pd.Series(1.0, index=factor_cols)
        else:
            mean_ic = hist.mean()
            if method == "equal":
                raw = mean_ic.apply(lambda x: 1.0 if pd.isna(x) or x >= 0 else -1.0)
            elif method == "ic":
                raw = mean_ic
            elif method == "icir":
                raw = mean_ic / hist.std().replace(0, np.nan)
            else:
                raise ValueError(f"Unsupported combination method: {method}")
        month_weights[ym] = _normalize_weights(raw, factor_cols)

    return pd.DataFrame(month_weights).T.sort_index()


def combine_factor_scores(
    panel: pd.DataFrame,
    weights_df: pd.DataFrame,
    factor_cols: Iterable[str] | None = None,
    score_col: str = "composite_score",
) -> pd.DataFrame:
    """Apply factor weights to the panel and return one composite score per stock-month."""
    available = get_available_factors(panel, factor_cols)
    if not available:
        raise ValueError("No factor columns are available for combination.")

    factor_data = panel[["stock_code", "year_month", "ret_next_month"] + available].copy()
    aligned_weights = weights_df.reindex(columns=available).fillna(0.0)
    factor_data = factor_data.merge(
        aligned_weights.reset_index().rename(columns={"index": "year_month"}),
        on="year_month",
        how="left",
        suffixes=("", "_weight"),
    )

    weighted_parts = [
        factor_data[factor].fillna(0.0) * factor_data[f"{factor}_weight"].fillna(0.0)
        for factor in available
    ]
    factor_data[score_col] = np.sum(weighted_parts, axis=0)
    return factor_data[["stock_code", "year_month", "ret_next_month", score_col]]


def save_weight_snapshot(weights_df: pd.DataFrame, method: str, save_root: str = "portfolio") -> str:
    """Persist monthly factor weights for audit."""
    save_dir = os.path.join(OUTPUT_DIR, save_root)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"factor_weights_{method}.csv")
    weights_df.to_csv(path, encoding="utf-8-sig")
    return path
