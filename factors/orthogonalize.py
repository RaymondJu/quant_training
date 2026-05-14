# -*- coding: utf-8 -*-
"""Cross-sectional factor orthogonalization utilities."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIVE_FACTOR_COLS, OUTPUT_DIR, PROCESSED_DIR

DEFAULT_FACTOR_COLS = ACTIVE_FACTOR_COLS


@dataclass(frozen=True)
class OrthogonalizationSpec:
    target: str
    bases: tuple[str, ...]


ORTHOGONALIZATION_SPECS = (
    OrthogonalizationSpec(target="BP", bases=("EP",)),
    OrthogonalizationSpec(target="SP", bases=("EP", "BP")),
    OrthogonalizationSpec(target="IVOL", bases=("VOL_20D",)),
    OrthogonalizationSpec(target="TURN_1M", bases=("VOL_20D", "IVOL")),
)


def load_factor_panel(panel_file: str = "factor_panel.parquet") -> pd.DataFrame:
    """Load the processed factor panel and normalize year_month dtype."""
    path = os.path.join(PROCESSED_DIR, panel_file)
    panel = pd.read_parquet(path).copy()
    panel["year_month"] = pd.PeriodIndex(panel["year_month"], freq="M")
    return panel.sort_values(["year_month", "stock_code"]).reset_index(drop=True)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def orthogonalize_monthly_factor(
    panel: pd.DataFrame,
    target: str,
    bases: list[str],
) -> pd.Series:
    """
    Residualize one factor against base factors month by month.

    The residual is z-scored cross-sectionally each month so the factor scale remains
    comparable to the original preprocessed panel.
    """
    result = pd.Series(index=panel.index, dtype=float)
    cols = [target] + bases

    for _, grp in panel.groupby("year_month"):
        data = grp[cols].copy()
        valid = data.dropna()
        if len(valid) < max(20, len(bases) + 5):
            result.loc[grp.index] = grp[target]
            continue

        x = valid[bases].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        y = valid[target].to_numpy(dtype=float)

        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        resid = _zscore(pd.Series(resid, index=valid.index))

        out = grp[target].copy()
        out.loc[valid.index] = resid
        out = out.fillna(0.0)
        result.loc[grp.index] = out

    return result


def orthogonalize_factor_panel(
    panel: pd.DataFrame,
    specs: tuple[OrthogonalizationSpec, ...] = ORTHOGONALIZATION_SPECS,
) -> pd.DataFrame:
    """Apply sequential orthogonalization specs to a factor panel."""
    orth_panel = panel.copy()
    for spec in specs:
        orth_panel[spec.target] = orthogonalize_monthly_factor(
            orth_panel,
            target=spec.target,
            bases=list(spec.bases),
        )
    return orth_panel


def compute_factor_correlation(panel: pd.DataFrame, factor_cols: list[str] | None = None) -> pd.DataFrame:
    """Compute Spearman correlation matrix for factor columns."""
    factor_cols = DEFAULT_FACTOR_COLS if factor_cols is None else factor_cols
    available = [col for col in factor_cols if col in panel.columns]
    return panel[available].corr(method="spearman")


def summarize_target_pairs(
    before: pd.DataFrame,
    after: pd.DataFrame,
    specs: tuple[OrthogonalizationSpec, ...] = ORTHOGONALIZATION_SPECS,
) -> pd.DataFrame:
    """Summarize before/after correlation changes for targeted factor pairs."""
    rows = []
    for spec in specs:
        for base in spec.bases:
            rows.append(
                {
                    "Target": spec.target,
                    "Base": base,
                    "Before_Corr": before.loc[spec.target, base],
                    "After_Corr": after.loc[spec.target, base],
                }
            )
    summary = pd.DataFrame(rows)
    summary["Corr_Reduction"] = summary["Before_Corr"].abs() - summary["After_Corr"].abs()
    return summary.sort_values("Corr_Reduction", ascending=False)


def save_orthogonalization_outputs(
    orth_panel: pd.DataFrame,
    before_corr: pd.DataFrame,
    after_corr: pd.DataFrame,
    pair_summary: pd.DataFrame,
    panel_file: str = "factor_panel_orthogonal.parquet",
) -> tuple[str, str]:
    """Persist orthogonalized panel and audit artifacts."""
    panel_path = os.path.join(PROCESSED_DIR, panel_file)
    orth_panel.to_parquet(panel_path, index=False)

    save_dir = os.path.join(OUTPUT_DIR, "orthogonalization")
    os.makedirs(save_dir, exist_ok=True)
    before_corr.to_csv(os.path.join(save_dir, "factor_correlation_before.csv"), encoding="utf-8-sig")
    after_corr.to_csv(os.path.join(save_dir, "factor_correlation_after.csv"), encoding="utf-8-sig")
    pair_summary.to_csv(os.path.join(save_dir, "target_pair_summary.csv"), index=False, encoding="utf-8-sig")
    return panel_path, save_dir


def main() -> None:
    panel = load_factor_panel()
    before_corr = compute_factor_correlation(panel)
    orth_panel = orthogonalize_factor_panel(panel)
    after_corr = compute_factor_correlation(orth_panel)
    pair_summary = summarize_target_pairs(before_corr, after_corr)
    panel_path, save_dir = save_orthogonalization_outputs(orth_panel, before_corr, after_corr, pair_summary)

    print(f"[orthogonalize] Saved orthogonalized panel: {panel_path}")
    print(f"[orthogonalize] Saved audit outputs to: {save_dir}")
    print(pair_summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == "__main__":
    main()
