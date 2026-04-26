# -*- coding: utf-8 -*-
"""Run orthogonalized-factor portfolio and ML experiments."""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from typing import Any, Callable

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INDEX_NAME, LGBM_TRAIN_WINDOW, OUTPUT_DIR, PROCESSED_DIR, RAW_DIR, TOP_N_STOCKS, TRANSACTION_COST
from factors.orthogonalize import (
    ORTHOGONALIZATION_SPECS,
    compute_factor_correlation,
    load_factor_panel,
    orthogonalize_factor_panel,
    save_orthogonalization_outputs,
    summarize_target_pairs,
)
from ml.model_comparison import (
    FACTOR_COLS,
    VAL_WINDOW,
    backtest_from_scores,
    get_model_family,
    plot_feature_importance,
    walk_forward_single_model,
)
from portfolio.backtest import run_all_methods
from portfolio.performance import summarize_returns

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

ORTH_PANEL_FILE = "factor_panel_orthogonal.parquet"
PORTFOLIO_SAVE_ROOT = "portfolio_orth"
ML_DEFAULT_SAVE_ROOT = "ml_orth_default"
ML_TUNED_SAVE_ROOT = "ml_orth_tuned"
ANALYSIS_SAVE_ROOT = "analysis_orth"
TUNING_PRED_MONTHS = 24


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def get_benchmark_returns() -> pd.Series:
    from data.benchmark import load_benchmark_returns
    return load_benchmark_returns()


def get_default_model_factories() -> dict[str, Callable[[], Any]]:
    return {
        "LightGBM-default": lambda: lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "RandomForest-default": lambda: RandomForestRegressor(random_state=42, n_jobs=1),
        "Ridge-default": lambda: Ridge(alpha=1.0),
    }


def get_tuning_candidates() -> dict[str, list[tuple[str, dict[str, Any], Callable[[], Any]]]]:
    return {
        "LightGBM": [
            (
                "lgbm_shallow",
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 7,
                 "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8},
                lambda: lgb.LGBMRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=3, num_leaves=7,
                    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbose=-1,
                ),
            ),
            (
                "lgbm_balanced",
                {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15,
                 "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
                 "reg_alpha": 0.1, "reg_lambda": 0.1},
                lambda: lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1,
                ),
            ),
            (
                "lgbm_deeper",
                {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 5, "num_leaves": 31,
                 "min_child_samples": 15, "subsample": 0.9, "colsample_bytree": 0.8},
                lambda: lgb.LGBMRegressor(
                    n_estimators=500, learning_rate=0.03, max_depth=5, num_leaves=31,
                    min_child_samples=15, subsample=0.9, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbose=-1,
                ),
            ),
        ],
        "RandomForest": [
            (
                "rf_shallow",
                {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 20, "max_features": 0.7},
                lambda: RandomForestRegressor(
                    n_estimators=200, max_depth=4, min_samples_leaf=20,
                    max_features=0.7, random_state=42, n_jobs=1,
                ),
            ),
            (
                "rf_balanced",
                {"n_estimators": 400, "max_depth": 6, "min_samples_leaf": 10, "max_features": "sqrt"},
                lambda: RandomForestRegressor(
                    n_estimators=400, max_depth=6, min_samples_leaf=10,
                    max_features="sqrt", random_state=42, n_jobs=1,
                ),
            ),
            (
                "rf_regularized",
                {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 15, "max_features": 0.5},
                lambda: RandomForestRegressor(
                    n_estimators=300, max_depth=5, min_samples_leaf=15,
                    max_features=0.5, random_state=42, n_jobs=1,
                ),
            ),
        ],
        "Ridge": [
            (
                f"ridge_alpha_{alpha:g}",
                {"alpha": alpha},
                lambda alpha=alpha: Ridge(alpha=alpha),
            )
            for alpha in [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        ],
    }


def orthogonalize_and_save() -> pd.DataFrame:
    panel = load_factor_panel()
    before_corr = compute_factor_correlation(panel)
    orth_panel = orthogonalize_factor_panel(panel, specs=ORTHOGONALIZATION_SPECS)
    after_corr = compute_factor_correlation(orth_panel)
    pair_summary = summarize_target_pairs(before_corr, after_corr, specs=ORTHOGONALIZATION_SPECS)
    panel_path, save_dir = save_orthogonalization_outputs(
        orth_panel, before_corr, after_corr, pair_summary, panel_file=ORTH_PANEL_FILE
    )
    print(f"[orth] orthogonalized panel saved: {panel_path}")
    print(f"[orth] audit outputs saved: {save_dir}")
    print(pair_summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return orth_panel


def run_portfolio_suite() -> pd.DataFrame:
    comparison = run_all_methods(panel_file=ORTH_PANEL_FILE, save_root=PORTFOLIO_SAVE_ROOT)
    print("\n[portfolio_orth] comparison")
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    return comparison


def get_tuning_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Period]:
    all_months = sorted(panel["year_month"].unique())
    end_idx = min(len(all_months) - 1, LGBM_TRAIN_WINDOW + VAL_WINDOW + TUNING_PRED_MONTHS - 1)
    end_month = all_months[end_idx]
    return panel[panel["year_month"] <= end_month].copy(), end_month


def tune_models(panel: pd.DataFrame, features: list[str], benchmark: pd.Series) -> tuple[dict[str, Callable[[], Any]], pd.DataFrame]:
    tuning_panel, tune_end = get_tuning_panel(panel)
    print(f"[tune] tuning panel end month: {tune_end}")
    rows = []
    selected: dict[str, Callable[[], Any]] = {}

    for family, candidates in get_tuning_candidates().items():
        print(f"\n[tune] {family}")
        best_key: tuple[float, float, float] | None = None
        best_factory: Callable[[], Any] | None = None
        for candidate_name, params, factory in candidates:
            predictions, monthly_ic, _ = walk_forward_single_model(
                tuning_panel,
                family,
                factory(),
                features,
            )
            if predictions.empty:
                continue
            strategy_rets, turnover = backtest_from_scores(predictions, tuning_panel, benchmark)
            perf = summarize_returns(
                strategy_rets,
                benchmark_returns=benchmark.reindex(strategy_rets.index),
                turnover=turnover,
            )
            key = (
                perf.get("Info_Ratio", float("-inf")) if pd.notna(perf.get("Info_Ratio")) else float("-inf"),
                perf.get("Sharpe", float("-inf")) if pd.notna(perf.get("Sharpe")) else float("-inf"),
                perf.get("Ann_Return", float("-inf")) if pd.notna(perf.get("Ann_Return")) else float("-inf"),
            )
            rows.append(
                {
                    "Family": family,
                    "Candidate": candidate_name,
                    "Params": json.dumps(params, ensure_ascii=False),
                    "Ann_Return": perf.get("Ann_Return"),
                    "Sharpe": perf.get("Sharpe"),
                    "Info_Ratio": perf.get("Info_Ratio"),
                    "Avg_Turnover": perf.get("Avg_Turnover"),
                    "RankIC_Mean": pd.Series(monthly_ic).mean() if monthly_ic else None,
                    "Months": len(strategy_rets),
                    "Tuning_End": str(tune_end),
                }
            )
            print(
                f"  {candidate_name}: Sharpe={perf.get('Sharpe', float('nan')):.3f}, "
                f"IR={perf.get('Info_Ratio', float('nan')):.3f}, "
                f"Ann={perf.get('Ann_Return', float('nan')):.2%}"
            )
            if best_key is None or key > best_key:
                best_key = key
                best_factory = factory

        if best_factory is None:
            raise RuntimeError(f"No valid tuning candidate for {family}")
        selected[f"{family}-tuned"] = best_factory

    tuning_df = pd.DataFrame(rows).sort_values(["Family", "Info_Ratio", "Sharpe"], ascending=[True, False, False])
    return selected, tuning_df


def run_model_suite(
    panel: pd.DataFrame,
    benchmark: pd.Series,
    model_factories: dict[str, Callable[[], Any]],
    save_root: str,
) -> pd.DataFrame:
    save_dir = os.path.join(OUTPUT_DIR, save_root)
    os.makedirs(save_dir, exist_ok=True)
    features = [col for col in FACTOR_COLS if col in panel.columns]
    perf_rows = []
    strategy_series: dict[str, pd.Series] = {}

    for model_label, factory in model_factories.items():
        print(f"\n[{save_root}] {model_label}")
        model = factory()
        predictions, monthly_ic, mean_imp = walk_forward_single_model(
            panel,
            model_label,
            model,
            features,
        )
        if predictions.empty:
            print("  [WARN] no predictions, skipped")
            continue

        slug = slugify(model_label)
        predictions.to_parquet(os.path.join(save_dir, f"predictions_{slug}.parquet"), index=False)
        if mean_imp is not None:
            plot_feature_importance(mean_imp, features, model_label, save_dir)

        strategy_rets, turnover = backtest_from_scores(predictions, panel, benchmark)
        strategy_series[model_label] = strategy_rets

        nav_df = pd.DataFrame({
            "year_month": strategy_rets.index.astype(str),
            "strategy_ret": strategy_rets.values,
            "benchmark_ret": benchmark.reindex(strategy_rets.index).values,
        })
        nav_df["nav"] = (1 + nav_df["strategy_ret"]).cumprod()
        nav_df["benchmark_nav"] = (1 + nav_df["benchmark_ret"].fillna(0.0)).cumprod()
        nav_df["excess_nav"] = nav_df["nav"] / nav_df["benchmark_nav"]
        nav_df.to_csv(os.path.join(save_dir, f"nav_{slug}.csv"), index=False, encoding="utf-8-sig")

        perf = summarize_returns(
            strategy_rets,
            benchmark_returns=benchmark.reindex(strategy_rets.index),
            turnover=turnover,
        )
        perf["Model"] = model_label
        perf["Model_Family"] = get_model_family(model_label)
        perf["RankIC_Mean"] = pd.Series(monthly_ic).mean() if monthly_ic else None
        perf_rows.append(perf)

        print(
            f"  RankIC={perf['RankIC_Mean']:.4f} "
            f"Ann={perf['Ann_Return']:.2%} Sharpe={perf['Sharpe']:.3f} "
            f"IR={perf.get('Info_Ratio', float('nan')):.3f}"
        )

    perf_df = pd.DataFrame(perf_rows).set_index("Model")
    perf_df.to_csv(os.path.join(save_dir, "model_comparison.csv"), encoding="utf-8-sig")
    plot_strategy_lines(strategy_series, benchmark, os.path.join(save_dir, "model_comparison_nav.png"), "Model NAV Comparison")
    return perf_df


def load_strategy_returns(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    ym_col = next((c for c in df.columns if "year_month" in c.lower() or c.lower() == "month"), None)
    ret_col = next((c for c in df.columns if any(k in c.lower() for k in ["strategy_ret", "net_ret", "strategy"])), None)
    if ym_col is None or ret_col is None:
        raise ValueError(f"Cannot infer columns from {csv_path}")
    idx = pd.PeriodIndex(df[ym_col].astype(str), freq="M")
    return pd.Series(df[ret_col].values, index=idx, name=os.path.basename(csv_path))


def build_strategy_map() -> dict[str, tuple[str, str]]:
    return {
        "Original IC-weight": (os.path.join(OUTPUT_DIR, "portfolio", "ic", "backtest_returns.csv"), "#1976d2"),
        "Orth Equal-weight": (os.path.join(OUTPUT_DIR, PORTFOLIO_SAVE_ROOT, "equal", "backtest_returns.csv"), "#4caf50"),
        "Orth IC-weight": (os.path.join(OUTPUT_DIR, PORTFOLIO_SAVE_ROOT, "ic", "backtest_returns.csv"), "#1565c0"),
        "Orth ICIR-weight": (os.path.join(OUTPUT_DIR, PORTFOLIO_SAVE_ROOT, "icir", "backtest_returns.csv"), "#7b1fa2"),
        "Orth LightGBM-default": (os.path.join(OUTPUT_DIR, ML_DEFAULT_SAVE_ROOT, "nav_lightgbm_default.csv"), "#c62828"),
        "Orth RandomForest-default": (os.path.join(OUTPUT_DIR, ML_DEFAULT_SAVE_ROOT, "nav_randomforest_default.csv"), "#ef6c00"),
        "Orth Ridge-default": (os.path.join(OUTPUT_DIR, ML_DEFAULT_SAVE_ROOT, "nav_ridge_default.csv"), "#6a1b9a"),
        "Orth LightGBM-tuned": (os.path.join(OUTPUT_DIR, ML_TUNED_SAVE_ROOT, "nav_lightgbm_tuned.csv"), "#ef5350"),
        "Orth RandomForest-tuned": (os.path.join(OUTPUT_DIR, ML_TUNED_SAVE_ROOT, "nav_randomforest_tuned.csv"), "#fb8c00"),
        "Orth Ridge-tuned": (os.path.join(OUTPUT_DIR, ML_TUNED_SAVE_ROOT, "nav_ridge_tuned.csv"), "#ab47bc"),
    }


def plot_strategy_lines(strategy_dict: dict[str, pd.Series], benchmark: pd.Series, save_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {
        name: color for name, (_, color) in build_strategy_map().items()
    }
    starts = [series.index.min() for series in strategy_dict.values() if not series.empty]
    common_start = max(starts) if starts else None

    for name, rets in strategy_dict.items():
        sliced = rets[rets.index >= common_start] if common_start else rets
        nav = (1 + sliced).cumprod()
        ax.plot(
            [p.to_timestamp() for p in nav.index],
            nav.values,
            label=name,
            linewidth=2 if "tuned" in name.lower() else 1.6,
            linestyle="-" if "tuned" in name.lower() else "--" if "default" in name.lower() else "-.",
            color=colors.get(name, None),
        )

    bm = benchmark[benchmark.index >= common_start] if common_start else benchmark
    bm_nav = (1 + bm.fillna(0)).cumprod()
    ax.plot([p.to_timestamp() for p in bm_nav.index], bm_nav.values, label=INDEX_NAME, color="black", linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Cumulative NAV")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_combined_analysis(benchmark: pd.Series) -> pd.DataFrame:
    save_dir = os.path.join(OUTPUT_DIR, ANALYSIS_SAVE_ROOT)
    os.makedirs(save_dir, exist_ok=True)

    strategy_map = build_strategy_map()
    strategy_dict: dict[str, pd.Series] = {}
    rows = []
    for name, (path, _) in strategy_map.items():
        if not os.path.exists(path):
            continue
        rets = load_strategy_returns(path)
        strategy_dict[name] = rets
        perf = summarize_returns(rets, benchmark_returns=benchmark.reindex(rets.index))
        perf.name = name
        rows.append(perf)

    union_index = sorted(set().union(*[series.index for series in strategy_dict.values() if not series.empty]))
    benchmark_perf = summarize_returns(benchmark.reindex(union_index).dropna())
    benchmark_perf.name = f"{INDEX_NAME} (Benchmark)"
    rows.append(benchmark_perf)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(save_dir, "performance_table_orth.csv"), encoding="utf-8-sig")
    formatted = table.copy()
    pct_cols = ["Ann_Return", "Ann_Vol", "Max_Drawdown", "Excess_Ann_Return", "Tracking_Error", "Benchmark_Ann_Return"]
    for col in pct_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "NA")
    for col in ["Sharpe", "Info_Ratio"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    if "Monthly_WinRate" in formatted.columns:
        formatted["Monthly_WinRate"] = formatted["Monthly_WinRate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "NA")
    formatted.to_csv(os.path.join(save_dir, "performance_table_orth_formatted.csv"), encoding="utf-8-sig")

    plot_strategy_lines(
        strategy_dict,
        benchmark,
        os.path.join(save_dir, "strategy_comparison_orth.png"),
        "Orthogonalized Strategy Comparison",
    )
    return table


def main() -> None:
    print("[pipeline] step 1/5 orthogonalize factor panel")
    orth_panel = orthogonalize_and_save()

    print("\n[pipeline] step 2/5 rerun portfolio baseline on orthogonalized factors")
    run_portfolio_suite()

    benchmark = get_benchmark_returns()
    features = [col for col in FACTOR_COLS if col in orth_panel.columns]

    print("\n[pipeline] step 3/5 run default ML suite on orthogonalized factors")
    default_perf = run_model_suite(
        orth_panel,
        benchmark,
        get_default_model_factories(),
        ML_DEFAULT_SAVE_ROOT,
    )

    print("\n[pipeline] step 4/5 tune models and rerun tuned ML suite")
    tuned_factories, tuning_df = tune_models(orth_panel, features, benchmark)
    tuning_dir = os.path.join(OUTPUT_DIR, ML_TUNED_SAVE_ROOT)
    os.makedirs(tuning_dir, exist_ok=True)
    tuning_df.to_csv(os.path.join(tuning_dir, "tuning_summary.csv"), index=False, encoding="utf-8-sig")
    selected_df = pd.DataFrame(
        [{"Model": model_name, "Params": tuning_df[tuning_df["Family"] == get_model_family(model_name)].iloc[0]["Params"]}
         for model_name in tuned_factories]
    )
    selected_df.to_csv(os.path.join(tuning_dir, "selected_params.csv"), index=False, encoding="utf-8-sig")
    tuned_perf = run_model_suite(
        orth_panel,
        benchmark,
        tuned_factories,
        ML_TUNED_SAVE_ROOT,
    )

    print("\n[pipeline] step 5/5 build comparison analysis tables")
    analysis_table = build_combined_analysis(benchmark)

    print("\n[pipeline] completed")
    print("\n[default ML]")
    print(default_perf.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n[tuned ML]")
    print(tuned_perf.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n[analysis table]")
    print(analysis_table.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
