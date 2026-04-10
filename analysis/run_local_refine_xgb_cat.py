# -*- coding: utf-8 -*-
"""Local second-stage tuning for XGBoost and CatBoost around default params."""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import optuna
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ml_tuning as base

LOCAL_TUNING_PRED_MONTHS = 36
LOCAL_TPE_SEED = 123
LOCAL_TRIALS = {
    "CatBoost": 12,
    "XGBoost": 12,
}
ML_LOCAL_SAVE_ROOT = "ml_tuned_local"
ANALYSIS_LOCAL_SAVE_ROOT = "analysis_tuned_local"


def get_local_tuning_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Period]:
    all_months = sorted(panel["year_month"].unique())
    end_idx = min(
        len(all_months) - 1,
        base.LGBM_TRAIN_WINDOW + base.VAL_WINDOW + LOCAL_TUNING_PRED_MONTHS - 1,
    )
    end_month = all_months[end_idx]
    return panel[panel["year_month"] <= end_month].copy(), end_month


def get_local_default_params(family: str) -> dict[str, Any]:
    defaults = {
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
    }
    return defaults[family]


def sample_local_params(trial: optuna.Trial, family: str) -> dict[str, Any]:
    if family == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 250, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.035, 0.10, log=True),
            "depth": trial.suggest_int("depth", 3, 5),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.5, 6.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.4, 1.6),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 0.6),
        }
    if family == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.035, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "min_child_weight": trial.suggest_float("min_child_weight", 10.0, 25.0),
            "subsample": trial.suggest_float("subsample", 0.75, 0.90),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 0.90),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.03, 0.30, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.03, 0.30, log=True),
        }
    raise ValueError(f"Unsupported family: {family}")


def tune_local_family(
    family: str,
    panel: pd.DataFrame,
    benchmark: pd.Series,
    features: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = sample_local_params(trial, family)
        try:
            score, perf, months = base.evaluate_params(panel, benchmark, family, params, features)
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

    sampler = optuna.samplers.TPESampler(seed=LOCAL_TPE_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(get_local_default_params(family))
    study.optimize(objective, n_trials=LOCAL_TRIALS[family], show_progress_bar=False)
    trials_df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    return study.best_trial.params, trials_df


def load_series_map(paths: dict[str, str]) -> dict[str, pd.Series]:
    return {
        name: base.load_strategy_returns(path)
        for name, path in paths.items()
        if os.path.exists(path)
    }


def main() -> None:
    print("[local_refine] load panel and benchmark")
    panel = base.load_factor_panel()
    benchmark = base.load_benchmark_returns()
    features = [col for col in base.FACTOR_COLS if col in panel.columns]
    tuning_panel, tune_end = get_local_tuning_panel(panel)
    print(f"[local_refine] tuning panel end month: {tune_end}")

    local_save_dir = os.path.join(base.OUTPUT_DIR, ML_LOCAL_SAVE_ROOT)
    analysis_save_dir = os.path.join(base.OUTPUT_DIR, ANALYSIS_LOCAL_SAVE_ROOT)
    os.makedirs(local_save_dir, exist_ok=True)
    os.makedirs(analysis_save_dir, exist_ok=True)

    selected_rows = []
    all_trials = []
    tuned_factories = {}

    for family in ["CatBoost", "XGBoost"]:
        print(f"\n[local_refine] {family}")
        best_params, trials_df = tune_local_family(family, tuning_panel, benchmark, features)
        all_trials.append(trials_df)
        tuned_factories[f"{family}-local"] = (
            lambda family=family, best_params=best_params: base.build_model(family, best_params)
        )
        best_row = trials_df.iloc[0]
        selected_rows.append(
            {
                "Model": f"{family}-local",
                "Params": json.dumps(best_params, ensure_ascii=False),
                "Tuning_End": str(tune_end),
                "Best_Score": best_row["Score"],
                "Best_Sharpe": best_row["Sharpe"],
                "Best_Info_Ratio": best_row["Info_Ratio"],
            }
        )
        print(
            f"  selected: Score={best_row['Score']:.3f} "
            f"Sharpe={best_row['Sharpe']:.3f} IR={best_row['Info_Ratio']:.3f}"
        )

    pd.concat(all_trials, ignore_index=True).to_csv(
        os.path.join(local_save_dir, "local_tuning_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(selected_rows).to_csv(
        os.path.join(local_save_dir, "selected_local_params.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[local_refine] run full-sample local tuned suites")
    base.run_model_suite(panel, benchmark, tuned_factories, ML_LOCAL_SAVE_ROOT)

    comparison_paths = {
        "CatBoost-default": os.path.join(base.OUTPUT_DIR, "ml", "nav_catboost.csv"),
        "CatBoost-tuned": os.path.join(base.OUTPUT_DIR, "ml_tuned", "nav_catboost_tuned.csv"),
        "CatBoost-local": os.path.join(base.OUTPUT_DIR, ML_LOCAL_SAVE_ROOT, "nav_catboost_local.csv"),
        "XGBoost-default": os.path.join(base.OUTPUT_DIR, "ml", "nav_xgboost.csv"),
        "XGBoost-tuned": os.path.join(base.OUTPUT_DIR, "ml_tuned", "nav_xgboost_tuned.csv"),
        "XGBoost-local": os.path.join(base.OUTPUT_DIR, ML_LOCAL_SAVE_ROOT, "nav_xgboost_local.csv"),
        "IC-weight": os.path.join(base.OUTPUT_DIR, "portfolio", "ic", "backtest_returns.csv"),
    }
    strategy_dict = load_series_map(comparison_paths)
    comparison = base.build_comparison_table(strategy_dict, benchmark)
    comparison.to_csv(
        os.path.join(analysis_save_dir, "xgb_cat_local_comparison.csv"),
        encoding="utf-8-sig",
    )
    base.plot_default_vs_tuned(
        strategy_dict,
        benchmark,
        os.path.join(analysis_save_dir, "xgb_cat_local_comparison.png"),
    )

    display_cols = [c for c in ["Ann_Return", "Sharpe", "Max_Drawdown", "Excess_Ann_Return", "Info_Ratio"] if c in comparison.columns]
    print("\n[local_refine] comparison")
    print(comparison[display_cols].sort_values("Sharpe", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n[local_refine] saved to {local_save_dir} and {analysis_save_dir}")


if __name__ == "__main__":
    main()
