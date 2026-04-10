# -*- coding: utf-8 -*-
"""Tune ML models with Optuna and compare default vs tuned variants."""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from typing import Any, Callable

from catboost import CatBoostRegressor
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LGBM_TRAIN_WINDOW, OUTPUT_DIR
from ml.model_comparison import (
    FACTOR_COLS,
    VAL_WINDOW,
    backtest_from_scores,
    load_benchmark_returns,
    load_factor_panel,
    plot_feature_importance,
    walk_forward_single_model,
)
from portfolio.performance import summarize_returns

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ML_TUNED_SAVE_ROOT = "ml_tuned"
ANALYSIS_SAVE_ROOT = "analysis_tuned"
TUNING_PRED_MONTHS = 24
TPE_SEED = 42
N_TRIALS = {
    "LightGBM": 8,
    "CatBoost": 6,
    "XGBoost": 8,
    "RandomForest": 8,
    "Ridge": 10,
}
FAMILY_COLORS = {
    "LightGBM": "#d32f2f",
    "CatBoost": "#6a1b9a",
    "XGBoost": "#00838f",
    "RandomForest": "#e65100",
    "Ridge": "#7b1fa2",
    "IC-weight": "#1976d2",
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def get_default_factories() -> dict[str, Callable[[], Any]]:
    return {
        "LightGBM-default": lambda: lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost-default": lambda: CatBoostRegressor(
            loss_function="RMSE",
            iterations=300,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=42,
            thread_count=1,
            verbose=False,
        ),
        "XGBoost-default": lambda: XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            eval_metric="rmse",
            verbosity=0,
        ),
        "RandomForest-default": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=20,
            max_features=0.7,
            random_state=42,
            n_jobs=1,
        ),
        "Ridge-default": lambda: Ridge(alpha=1.0),
    }


def get_tuning_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Period]:
    all_months = sorted(panel["year_month"].unique())
    end_idx = min(len(all_months) - 1, LGBM_TRAIN_WINDOW + VAL_WINDOW + TUNING_PRED_MONTHS - 1)
    end_month = all_months[end_idx]
    return panel[panel["year_month"] <= end_month].copy(), end_month


def score_from_perf(perf: pd.Series) -> float:
    info_ratio = perf.get("Info_Ratio")
    sharpe = perf.get("Sharpe")
    turnover = perf.get("Avg_Turnover")
    info_ratio = float(info_ratio) if pd.notna(info_ratio) else -5.0
    sharpe = float(sharpe) if pd.notna(sharpe) else -5.0
    turnover = float(turnover) if pd.notna(turnover) else 1.0
    return info_ratio + 0.3 * sharpe - 0.05 * turnover


def build_model(family: str, params: dict[str, Any]) -> Any:
    if family == "LightGBM":
        return lgb.LGBMRegressor(
            objective="regression",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    if family == "CatBoost":
        return CatBoostRegressor(
            loss_function="RMSE",
            random_seed=42,
            thread_count=1,
            verbose=False,
            **params,
        )
    if family == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            eval_metric="rmse",
            verbosity=0,
            **params,
        )
    if family == "RandomForest":
        return RandomForestRegressor(random_state=42, n_jobs=1, **params)
    if family == "Ridge":
        return Ridge(**params)
    raise ValueError(f"Unknown family: {family}")


def sample_params(trial: optuna.Trial, family: str) -> dict[str, Any]:
    if family == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63, step=4),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40, step=5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        }
    if family == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 150, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        }
    if family == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 40.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        }
    if family == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30, step=5),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 60, step=10),
            "max_features": trial.suggest_categorical("max_features", [0.5, 0.7, "sqrt", "log2"]),
        }
    if family == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-2, 1e2, log=True)}
    raise ValueError(f"Unknown family: {family}")


def get_default_search_params(family: str) -> dict[str, Any]:
    defaults = {
        "LightGBM": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        "CatBoost": {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 4,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "bagging_temperature": 0.0,
        },
        "XGBoost": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 20.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        "RandomForest": {
            "n_estimators": 200,
            "max_depth": 4,
            "min_samples_leaf": 20,
            "min_samples_split": 20,
            "max_features": 0.7,
        },
        "Ridge": {"alpha": 1.0},
    }
    return defaults[family]


def evaluate_params(
    panel: pd.DataFrame,
    benchmark: pd.Series,
    family: str,
    params: dict[str, Any],
    features: list[str],
) -> tuple[float, pd.Series, float]:
    model = build_model(family, params)
    predictions, monthly_ic, _ = walk_forward_single_model(panel, family, model, features)
    if predictions.empty:
        raise RuntimeError(f"{family} returned no predictions")
    strategy_rets, turnover = backtest_from_scores(predictions, panel, benchmark)
    perf = summarize_returns(
        strategy_rets,
        benchmark_returns=benchmark.reindex(strategy_rets.index),
        turnover=turnover,
    )
    perf["RankIC_Mean"] = pd.Series(monthly_ic).mean() if monthly_ic else np.nan
    return score_from_perf(perf), perf, len(strategy_rets)


def tune_family(
    family: str,
    panel: pd.DataFrame,
    benchmark: pd.Series,
    features: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, family)
        try:
            score, perf, months = evaluate_params(panel, benchmark, family, params, features)
        except Exception:
            return -1e9
        rows.append(
            {
                "Family": family,
                "Trial": trial.number,
                "Score": score,
                "Params": json.dumps(params, ensure_ascii=False),
                "Ann_Return": perf.get("Ann_Return"),
                "Sharpe": perf.get("Sharpe"),
                "Info_Ratio": perf.get("Info_Ratio"),
                "Avg_Turnover": perf.get("Avg_Turnover"),
                "RankIC_Mean": perf.get("RankIC_Mean"),
                "Months": months,
            }
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=TPE_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(get_default_search_params(family))
    study.optimize(objective, n_trials=N_TRIALS[family], show_progress_bar=False)
    trials_df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    return study.best_trial.params, trials_df


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
        predictions, monthly_ic, mean_imp = walk_forward_single_model(
            panel,
            model_label,
            factory(),
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

        pd.DataFrame(
            {"year_month": strategy_rets.index.astype(str), "strategy_ret": strategy_rets.values}
        ).to_csv(
            os.path.join(save_dir, f"nav_{slug}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        perf = summarize_returns(
            strategy_rets,
            benchmark_returns=benchmark.reindex(strategy_rets.index),
            turnover=turnover,
        )
        perf["Model"] = model_label
        perf["RankIC_Mean"] = pd.Series(monthly_ic).mean() if monthly_ic else np.nan
        perf_rows.append(perf)
        print(
            f"  RankIC={perf['RankIC_Mean']:.4f} "
            f"Ann={perf['Ann_Return']:.2%} Sharpe={perf['Sharpe']:.3f} "
            f"IR={perf.get('Info_Ratio', float('nan')):.3f}"
        )

    perf_df = pd.DataFrame(perf_rows).set_index("Model")
    perf_df.to_csv(os.path.join(save_dir, "model_comparison.csv"), encoding="utf-8-sig")
    return perf_df


def load_strategy_returns(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    ym_col = next((c for c in df.columns if "year_month" in c.lower() or c.lower() == "month"), None)
    ret_col = next((c for c in df.columns if any(k in c.lower() for k in ["strategy_ret", "net_ret", "strategy"])), None)
    if ym_col is None or ret_col is None:
        raise ValueError(f"Cannot infer columns from {csv_path}")
    idx = pd.PeriodIndex(df[ym_col].astype(str), freq="M")
    return pd.Series(df[ret_col].values, index=idx, name=os.path.basename(csv_path))


def build_comparison_table(strategy_dict: dict[str, pd.Series], benchmark: pd.Series) -> pd.DataFrame:
    rows = []
    for name, rets in strategy_dict.items():
        bm = benchmark.reindex(rets.index)
        perf = summarize_returns(rets, benchmark_returns=bm)
        perf.name = name
        rows.append(perf)
    return pd.DataFrame(rows)


def plot_default_vs_tuned(strategy_dict: dict[str, pd.Series], benchmark: pd.Series, save_path: str) -> None:
    ml_starts = [s.index.min() for name, s in strategy_dict.items() if name != "IC-weight" and not s.empty]
    common_start = max(ml_starts) if ml_starts else None
    fig, ax = plt.subplots(figsize=(14, 7))

    baseline = strategy_dict.get("IC-weight", pd.Series(dtype=float))
    if not baseline.empty:
        sliced = baseline[baseline.index >= common_start] if common_start else baseline
        nav = (1 + sliced).cumprod()
        ax.plot([p.to_timestamp() for p in nav.index], nav.values, color=FAMILY_COLORS["IC-weight"], linewidth=2.4, linestyle="--", label="IC-weight")

    for family in ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ridge"]:
        for variant, linestyle in [("default", ":"), ("tuned", "-")]:
            name = f"{family}-{variant}"
            rets = strategy_dict.get(name, pd.Series(dtype=float))
            if rets.empty:
                continue
            sliced = rets[rets.index >= common_start] if common_start else rets
            nav = (1 + sliced).cumprod()
            ax.plot(
                [p.to_timestamp() for p in nav.index],
                nav.values,
                color=FAMILY_COLORS[family],
                linewidth=2.2 if variant == "tuned" else 1.5,
                linestyle=linestyle,
                label=name,
            )

    bm = benchmark[benchmark.index >= common_start] if common_start else benchmark
    bm_nav = (1 + bm.fillna(0)).cumprod()
    ax.plot([p.to_timestamp() for p in bm_nav.index], bm_nav.values, color="gray", linewidth=1.5, linestyle="-.", label="HS300")

    ax.set_title("Default vs Tuned ML Models")
    ax.set_ylabel("Cumulative NAV")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("[tuning] load panel and benchmark")
    panel = load_factor_panel()
    benchmark = load_benchmark_returns()
    features = [col for col in FACTOR_COLS if col in panel.columns]
    tuning_panel, tune_end = get_tuning_panel(panel)
    print(f"[tuning] tuning panel end month: {tune_end}")

    tuned_save_dir = os.path.join(OUTPUT_DIR, ML_TUNED_SAVE_ROOT)
    analysis_save_dir = os.path.join(OUTPUT_DIR, ANALYSIS_SAVE_ROOT)
    os.makedirs(tuned_save_dir, exist_ok=True)
    os.makedirs(analysis_save_dir, exist_ok=True)

    trial_tables = []
    tuned_factories: dict[str, Callable[[], Any]] = {}
    selected_rows = []

    for family in ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ridge"]:
        print(f"\n[tuning] {family}")
        best_params, trials_df = tune_family(family, tuning_panel, benchmark, features)
        trial_tables.append(trials_df)
        tuned_factories[f"{family}-tuned"] = (
            lambda family=family, best_params=best_params: build_model(family, best_params)
        )
        selected_rows.append(
            {"Model": f"{family}-tuned", "Params": json.dumps(best_params, ensure_ascii=False), "Tuning_End": str(tune_end)}
        )
        best_row = trials_df.iloc[0]
        print(
            f"  selected: Score={best_row['Score']:.3f} "
            f"Sharpe={best_row['Sharpe']:.3f} IR={best_row['Info_Ratio']:.3f}"
        )

    tuning_summary = pd.concat(trial_tables, ignore_index=True)
    tuning_summary.to_csv(os.path.join(tuned_save_dir, "tuning_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(
        os.path.join(tuned_save_dir, "selected_params.csv"), index=False, encoding="utf-8-sig"
    )

    print("\n[tuning] run full-sample tuned model suite")
    tuned_perf = run_model_suite(panel, benchmark, tuned_factories, ML_TUNED_SAVE_ROOT)

    strategy_map = {
        "IC-weight": os.path.join(OUTPUT_DIR, "portfolio", "ic", "backtest_returns.csv"),
    }
    for family in ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ridge"]:
        strategy_map[f"{family}-default"] = os.path.join(OUTPUT_DIR, "ml", f"nav_{family.lower()}.csv")
        strategy_map[f"{family}-tuned"] = os.path.join(OUTPUT_DIR, ML_TUNED_SAVE_ROOT, f"nav_{slugify(family + '-tuned')}.csv")

    strategy_dict = {
        name: load_strategy_returns(path)
        for name, path in strategy_map.items()
        if os.path.exists(path)
    }
    comparison = build_comparison_table(strategy_dict, benchmark)
    comparison.to_csv(
        os.path.join(analysis_save_dir, "default_vs_tuned_performance.csv"),
        encoding="utf-8-sig",
    )
    plot_default_vs_tuned(
        strategy_dict,
        benchmark,
        os.path.join(analysis_save_dir, "default_vs_tuned_nav.png"),
    )

    print("\n[tuning] top rows")
    print(
        comparison[
            [c for c in ["Ann_Return", "Sharpe", "Max_Drawdown", "Excess_Ann_Return", "Info_Ratio"] if c in comparison.columns]
        ].sort_values("Sharpe", ascending=False).to_string(float_format=lambda x: f"{x:.4f}")
    )
    print(f"\n[tuning] saved to {tuned_save_dir} and {analysis_save_dir}")


if __name__ == "__main__":
    main()
