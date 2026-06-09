from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_UNIVERSE", "csi500")

from csi500_daily_alpha_pipeline import (
    _forward_compound_return,
    backtest_lightgbm,
    backtest_model,
    backtest_model_optuna,
    backtest_ridge_cv,
    build_cost_sensitivity,
    summarize_period_returns,
)
from data.benchmark import load_benchmark_daily_returns


PROCESSED_DIR = ROOT / "data" / "processed" / "csi500" / "daily_alpha"
OUTPUT_DIR = ROOT / "output" / "csi500" / "daily_alpha" / "trade_constraint_validation"
PANEL_PATH = PROCESSED_DIR / "daily_alpha_panel_trade_constraints.parquet"
OLD_SUMMARY_PATH = ROOT / "output" / "csi500" / "daily_alpha" / "performance_summary.csv"

BASELINE_RETURN_FILES = {
    "IC-weight": "ic_weight_returns.csv",
    "IC-weight size-neutral": "ic_weight_size_neutral_returns.csv",
    "Ridge": "ridge_returns.csv",
    "Ridge size-neutral": "ridge_size_neutral_returns.csv",
}

MODEL_RETURN_FILES = {
    "RidgeCV": "ridge_cv_returns.csv",
    "LightGBM": "lightgbm_returns.csv",
    "XGBoost": "xgboost_returns.csv",
    "CatBoost": "catboost_returns.csv",
    "RandomForest": "random_forest_returns.csv",
    "LightGBM Optuna": "lightgbm_optuna_returns.csv",
    "XGBoost Optuna": "xgboost_optuna_returns.csv",
    "CatBoost Optuna": "catboost_optuna_returns.csv",
    "RandomForest Optuna": "random_forest_optuna_returns.csv",
}

PARAM_FILES = {
    "RidgeCV": "ridge_cv_params.csv",
    "LightGBM Optuna": "lightgbm_optuna_params.csv",
    "XGBoost Optuna": "xgboost_optuna_params.csv",
    "CatBoost Optuna": "catboost_optuna_params.csv",
    "RandomForest Optuna": "random_forest_optuna_params.csv",
}

DEFAULT_MODELS = list(MODEL_RETURN_FILES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume CSI500 ML backtests on the latest trade-constraint panel."
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model labels, or 'all'. Existing outputs are skipped unless --force is used.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--start-days", type=int, default=756)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--cost-scenarios-bps", default="30,60,100")
    parser.add_argument("--max-train-rows", type=int, default=50_000)
    parser.add_argument("--optuna-trials", type=int, default=12)
    parser.add_argument("--optuna-val-days", type=int, default=63)
    parser.add_argument("--optuna-retune-every", type=int, default=25)
    return parser.parse_args()


def selected_models(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return DEFAULT_MODELS.copy()
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in requested if item not in MODEL_RETURN_FILES]
    if unknown:
        raise ValueError(f"Unknown model labels: {unknown}. Valid labels: {DEFAULT_MODELS}")
    return requested


def validate_panel(panel: pd.DataFrame, horizon: int) -> str:
    target_col = f"ret_fwd_{horizon}d"
    required = {
        "stock_code",
        "date",
        target_col,
        "execution_ret",
        "tradeable",
        "sellable",
        "suspended_next_open",
        "limit_down_next_open",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"{PANEL_PATH} is not the latest trade-constraint panel; missing {missing}")
    return target_col


def save_returns(label: str, returns: pd.DataFrame) -> None:
    returns.to_csv(OUTPUT_DIR / MODEL_RETURN_FILES[label], index=False, encoding="utf-8-sig")


def run_model(
    label: str,
    panel: pd.DataFrame,
    target_col: str,
    args: argparse.Namespace,
) -> None:
    common = (
        panel,
        target_col,
        args.horizon,
        args.top_n,
        args.train_days,
        args.start_days,
        args.cost_bps,
    )
    if label == "RidgeCV":
        returns, params = backtest_ridge_cv(*common)
        params.to_csv(OUTPUT_DIR / PARAM_FILES[label], index=False, encoding="utf-8-sig")
    elif label == "LightGBM":
        returns = backtest_lightgbm(*common, max_train_rows=args.max_train_rows)
    elif label in {"XGBoost", "CatBoost", "RandomForest"}:
        returns = backtest_model(
            label,
            *common,
            max_train_rows=args.max_train_rows,
        )
    else:
        model_name = label.removesuffix(" Optuna")
        returns, params = backtest_model_optuna(
            model_name,
            *common,
            n_trials=args.optuna_trials,
            val_days=args.optuna_val_days,
            retune_every=args.optuna_retune_every,
            max_train_rows=args.max_train_rows,
        )
        params.to_csv(OUTPUT_DIR / PARAM_FILES[label], index=False, encoding="utf-8-sig")
    save_returns(label, returns)


def build_summary(horizon: int) -> pd.DataFrame:
    benchmark = load_benchmark_daily_returns().copy()
    benchmark[f"benchmark_fwd_{horizon}d"] = _forward_compound_return(
        benchmark["index_ret"], horizon
    )
    rows = []
    for label, filename in {**BASELINE_RETURN_FILES, **MODEL_RETURN_FILES}.items():
        path = OUTPUT_DIR / filename
        if not path.exists():
            continue
        returns = pd.read_csv(path, parse_dates=["date"])
        rows.append(summarize_period_returns(returns, benchmark, horizon, label))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "performance_all_models.csv", index=False, encoding="utf-8-sig")
    return summary


def load_return_sets() -> dict[str, pd.DataFrame]:
    return_sets = {}
    for label, filename in {**BASELINE_RETURN_FILES, **MODEL_RETURN_FILES}.items():
        path = OUTPUT_DIR / filename
        if path.exists():
            return_sets[label] = pd.read_csv(path, parse_dates=["date"])
    return return_sets


def build_supporting_tables(horizon: int, cost_bps_list: list[float]) -> None:
    benchmark = load_benchmark_daily_returns().copy()
    benchmark[f"benchmark_fwd_{horizon}d"] = _forward_compound_return(
        benchmark["index_ret"], horizon
    )
    return_sets = load_return_sets()
    cost_summary = build_cost_sensitivity(
        return_sets,
        benchmark,
        horizon,
        cost_bps_list,
    )
    cost_summary.to_csv(
        OUTPUT_DIR / "cost_sensitivity_all_models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if OLD_SUMMARY_PATH.exists():
        old = pd.read_csv(OLD_SUMMARY_PATH)
        new = pd.read_csv(OUTPUT_DIR / "performance_all_models.csv")
        compare_cols = ["Label", "Ann_Return", "Sharpe", "Max_Drawdown", "Avg_Turnover"]
        comparison = old[compare_cols].merge(
            new[
                compare_cols
                + [
                    "Locked_Rebalances",
                    "Limit_Down_Locks",
                    "Suspension_Locks",
                ]
            ],
            on="Label",
            suffixes=("_Before", "_After"),
        )
        comparison["Ann_Return_Change_pp"] = (
            comparison["Ann_Return_After"] - comparison["Ann_Return_Before"]
        ) * 100
        comparison["Sharpe_Change"] = (
            comparison["Sharpe_After"] - comparison["Sharpe_Before"]
        )
        comparison.to_csv(
            OUTPUT_DIR / "performance_before_after_all_models.csv",
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    target_col = validate_panel(panel, args.horizon)
    models = selected_models(args.models)

    print(
        f"[trade-constraint-ml] panel={PANEL_PATH} rows={len(panel):,} "
        f"dates={panel['date'].nunique():,} stocks={panel['stock_code'].nunique():,}"
    )
    for label in models:
        output_path = OUTPUT_DIR / MODEL_RETURN_FILES[label]
        if output_path.exists() and not args.force:
            print(f"[trade-constraint-ml] skip existing {label}: {output_path.name}")
            continue
        print(f"[trade-constraint-ml] running {label}")
        run_model(label, panel, target_col, args)
        summary = build_summary(args.horizon)
        current = summary.loc[summary["Label"] == label]
        if not current.empty:
            print(current.to_string(index=False))

    summary = build_summary(args.horizon)
    cost_bps_list = [
        float(value.strip())
        for value in args.cost_scenarios_bps.split(",")
        if value.strip()
    ]
    build_supporting_tables(args.horizon, cost_bps_list)
    print("\n[trade-constraint-ml] current summary:")
    print(summary.to_string(index=False))
    print(f"\n[trade-constraint-ml] outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
