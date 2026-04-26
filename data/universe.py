# -*- coding: utf-8 -*-
"""
Time-varying universe module based on constituent entry dates.

AKShare only provides the current constituent list plus entry dates, so the
current implementation is still a partial survivorship-bias fix:
  - It excludes stocks that had not yet entered the index by the rebalance date.
  - It cannot remove stocks that once belonged to the index but were later
    deleted from today's constituent list.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INDEX_NAME, UNIVERSE_ID, get_constituents_path


@lru_cache(maxsize=4)
def _load_constituents(universe_id: str = UNIVERSE_ID) -> pd.DataFrame:
    """Load constituent metadata with entry dates."""
    path = get_constituents_path(universe_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Constituent file not found: {path}\n"
            "Run python data/download.py first."
        )
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].str.zfill(6)
    df["entry_date"] = pd.to_datetime(df["date"])
    df["entry_month"] = df["entry_date"].dt.to_period("M")
    return df[["stock_code", "stock_name", "entry_date", "entry_month"]]


def get_universe_at_month(year_month: pd.Period, universe_id: str = UNIVERSE_ID) -> set[str]:
    """
    Return investable constituents for the specified month.

    Logic: only keep stocks whose entry month is no later than the rebalance
    month. This removes "future entry" leakage.
    """
    constituents = _load_constituents(universe_id)
    mask = constituents["entry_month"] <= year_month
    return set(constituents.loc[mask, "stock_code"])


def get_all_stock_codes(universe_id: str = UNIVERSE_ID) -> set[str]:
    """Return the current constituent codes without time filtering."""
    return set(_load_constituents(universe_id)["stock_code"])


if __name__ == "__main__":
    test_months = [pd.Period(f"{y}-06", freq="M") for y in range(2015, 2026)]
    print(f"Universe size by year for {INDEX_NAME} ({UNIVERSE_ID}):")
    for ym in test_months:
        u = get_universe_at_month(ym)
        print(f"  {ym}: {len(u)} stocks")
