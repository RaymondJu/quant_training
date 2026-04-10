# -*- coding: utf-8 -*-
"""
多模型 Walk-Forward 对比
包含：LightGBM / Random Forest / Ridge Regression

统一的走步前向训练框架，任意 sklearn 兼容估计器均可接入。

输出（output/ml/）：
  model_comparison.csv          → 各模型绩效汇总
  model_comparison_nav.png      → 净值对比图
  predictions_{model}.parquet   → 各模型样本外预测分数
  feature_importance_{model}.csv/png → 特征重要性

Usage:
    python ml/model_comparison.py
"""
from __future__ import annotations

import copy
import os
import sys
import warnings
from typing import Any

from catboost import CatBoostRegressor
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LGBM_TRAIN_WINDOW, OUTPUT_DIR, PROCESSED_DIR, RAW_DIR, TOP_N_STOCKS, TRANSACTION_COST
from portfolio.performance import summarize_returns

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

FACTOR_COLS = [
    "EP", "BP", "SP", "MOM_12_1", "REV_1M", "ROE_TTM",
    "GPM_change", "VOL_20D", "IVOL", "TURN_1M", "AMIHUD",
    "SIZE", "BETA_60D", "ABTURN_1M", "OCF_QUALITY", "ASSET_GROWTH",
]
VAL_WINDOW = 3
IMPORTANCE_LAST_N = 12

# ======== 模型配置 ========
LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

MODELS: dict[str, Any] = {
    "LightGBM": lgb.LGBMRegressor(**LGBM_PARAMS),
    "CatBoost": CatBoostRegressor(
        loss_function="RMSE",
        iterations=300,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=3.0,
        random_seed=42,
        thread_count=1,
        verbose=False,
    ),
    "XGBoost": XGBRegressor(
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
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=20,
        max_features=0.7,
        random_state=42,
        n_jobs=1,
    ),
    # RidgeCV 自动从 alphas 中选最优正则强度（用训练集内 LOO 交叉验证）
    "Ridge": RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], fit_intercept=True),
}

# Ridge 需要标准化（LightGBM/RF 不需要）
NEEDS_SCALING = {"Ridge"}


def get_model_family(model_name: str) -> str:
    """Map suffixed model labels back to the model family name."""
    for family in ("LightGBM", "CatBoost", "XGBoost", "RandomForest", "Ridge"):
        if model_name.startswith(family):
            return family
    return model_name


# =====================================================
#  数据加载
# =====================================================

def load_factor_panel() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "factor_panel.parquet")
    df = pd.read_parquet(path)
    if not hasattr(df["year_month"].dtype, "freq"):
        df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df


def load_benchmark_returns() -> pd.Series:
    path = os.path.join(RAW_DIR, "index_hs300_daily.parquet")
    idx = pd.read_parquet(path).copy()
    idx["date"] = pd.to_datetime(idx["date"])
    month_end = idx.sort_values("date").groupby(idx["date"].dt.to_period("M")).tail(1).copy()
    month_end["ret"] = month_end["close"].pct_change()
    month_end["year_month"] = month_end["date"].dt.to_period("M") - 1
    s = month_end.set_index("year_month")["ret"].dropna()
    return pd.Series(s.values, index=pd.PeriodIndex(s.index, freq="M"))


# =====================================================
#  Walk-Forward 核心
# =====================================================

def walk_forward_single_model(
    panel: pd.DataFrame,
    model_name: str,
    model: Any,
    features: list[str],
    train_window: int = LGBM_TRAIN_WINDOW,
    val_window: int = VAL_WINDOW,
) -> tuple[pd.DataFrame, list[float], np.ndarray | None]:
    """
    对单个模型执行走步前向训练与预测。

    Returns
    -------
    predictions : pd.DataFrame [stock_code, year_month, score]
    monthly_ic  : list[float]
    mean_importance : np.ndarray | None  (shape: n_features)
    """
    all_months = sorted(panel["year_month"].unique())
    start_idx = train_window + val_window
    pred_months = all_months[start_idx:]

    all_preds: list[pd.DataFrame] = []
    monthly_ic: list[float] = []
    importance_accum: list[np.ndarray] = []
    importance_start = max(0, len(pred_months) - IMPORTANCE_LAST_N)
    family = get_model_family(model_name)
    needs_scale = family in NEEDS_SCALING

    for i, pred_month in enumerate(pred_months):
        pred_idx = all_months.index(pred_month)
        train_months = all_months[pred_idx - val_window - train_window: pred_idx - val_window]
        val_months   = all_months[pred_idx - val_window: pred_idx]

        train_data = panel[panel["year_month"].isin(train_months)].dropna(
            subset=features + ["ret_next_month"]
        )
        val_data = panel[panel["year_month"].isin(val_months)].dropna(
            subset=features + ["ret_next_month"]
        )
        pred_data = panel[panel["year_month"] == pred_month].dropna(subset=features)

        if len(train_data) < 50 or pred_data.empty:
            continue

        X_train, y_train = train_data[features].values, train_data["ret_next_month"].values
        X_pred = pred_data[features].values

        # 标准化（Ridge）
        if needs_scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_pred  = scaler.transform(X_pred)
            if len(val_data) > 0:
                scaler.transform(val_data[features].values)  # 与训练集同参数

        # LightGBM 单独处理（需要 early stopping）
        if family in {"LightGBM", "CatBoost", "XGBoost"} and len(val_data) >= 10:
            X_val = val_data[features].values
            y_val = val_data["ret_next_month"].values
            fitted = copy.deepcopy(model)
            if family == "LightGBM":
                fitted.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                )
            elif family == "CatBoost":
                fitted.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=30,
                    verbose=False,
                )
            else:
                fitted.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
        else:
            fitted = copy.deepcopy(model)
            fitted.fit(X_train, y_train)

        scores = fitted.predict(X_pred)
        pred_df = pred_data[["stock_code", "year_month"]].copy()
        pred_df["score"] = scores
        all_preds.append(pred_df)

        # 月度 IC
        if "ret_next_month" in pred_data.columns:
            tmp = pred_data[["ret_next_month"]].copy()
            tmp["score"] = scores
            tmp = tmp.dropna()
            if len(tmp) > 10:
                ic = tmp["score"].corr(tmp["ret_next_month"], method="spearman")
                monthly_ic.append(ic)

        # 特征重要性
        if i >= importance_start:
            if family == "LightGBM":
                imp = fitted.booster_.feature_importance(importance_type="gain").astype(float)
                importance_accum.append(imp)
            elif family == "CatBoost":
                importance_accum.append(fitted.get_feature_importance())
            elif family == "XGBoost":
                importance_accum.append(fitted.feature_importances_)
            elif family == "RandomForest":
                importance_accum.append(fitted.feature_importances_)
            elif family == "Ridge":
                importance_accum.append(np.abs(fitted.coef_))

    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    mean_imp = np.mean(importance_accum, axis=0) if importance_accum else None
    return predictions, monthly_ic, mean_imp


# =====================================================
#  组合回测（复用 evaluate.py 逻辑）
# =====================================================

def backtest_from_scores(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    benchmark: pd.Series,
    top_n: int = TOP_N_STOCKS,
    cost: float = TRANSACTION_COST,
) -> tuple[pd.Series, pd.Series]:
    """
    月度选股回测，返回 (strategy_returns, turnover)，index=year_month。
    """
    returns = panel[["stock_code", "year_month", "ret_next_month"]]
    scored = predictions.merge(returns, on=["stock_code", "year_month"], how="inner")
    months = sorted(scored["year_month"].unique())

    ret_list, to_list, month_list = [], [], []
    prev_holdings: set[str] = set()

    for ym in months:
        cross = scored[scored["year_month"] == ym].dropna(subset=["score", "ret_next_month"])
        if cross.empty:
            continue
        top = cross.sort_values("score", ascending=False).head(top_n)["stock_code"].tolist()
        if not top:
            continue

        enter = set(top) - prev_holdings
        exit_ = prev_holdings - set(top)
        to = (len(enter) + len(exit_)) / max(len(prev_holdings | set(top)), 1)
        trade_cost = to * cost

        period_ret = cross[cross["stock_code"].isin(top)]["ret_next_month"].mean()
        ret_list.append(period_ret - trade_cost)
        to_list.append(to)
        month_list.append(ym)
        prev_holdings = set(top)

    idx = pd.PeriodIndex(month_list, freq="M")
    return pd.Series(ret_list, index=idx), pd.Series(to_list, index=idx)


# =====================================================
#  可视化
# =====================================================

def plot_feature_importance(mean_imp: np.ndarray, features: list[str],
                             model_name: str, save_dir: str):
    family = get_model_family(model_name)
    imp_df = pd.DataFrame({
        "Feature": features, "Importance": mean_imp
    }).sort_values("Importance", ascending=False)

    imp_df.to_csv(
        os.path.join(save_dir, f"feature_importance_{model_name}.csv"),
        index=False, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#d32f2f"] + ["#1976d2"] * (len(imp_df) - 1)
    ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color=bar_colors[::-1])
    if family == "Ridge":
        label = "Mean |Coefficient|"
    elif family == "LightGBM":
        label = "Mean Gain"
    elif family == "CatBoost":
        label = "Mean Feature Importance"
    elif family == "XGBoost":
        label = "Mean Feature Importance"
    else:
        label = "Feature Importance"
    ax.set_xlabel(label)
    ax.set_title(f"{model_name} Feature Importance (last {IMPORTANCE_LAST_N} months)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"feature_importance_{model_name}.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_model_nav_comparison(
    strategy_series: dict[str, pd.Series],
    benchmark: pd.Series,
    save_dir: str,
):
    """所有 ML 模型净值对比图"""
    colors = {
        "LightGBM": "#d32f2f",
        "CatBoost": "#6a1b9a",
        "XGBoost": "#00838f",
        "RandomForest": "#e65100",
        "Ridge": "#7b1fa2",
    }

    # 共同起始月（以最晚开始的策略为准）
    starts = [s.index.min() for s in strategy_series.values() if not s.empty]
    common_start = max(starts) if starts else None

    fig, ax = plt.subplots(figsize=(13, 6))
    for name, rets in strategy_series.items():
        if rets.empty:
            continue
        sliced = rets[rets.index >= common_start] if common_start else rets
        nav = (1 + sliced).cumprod()
        dates = [p.to_timestamp() for p in nav.index]
        ax.plot(dates, nav.values, label=name, color=colors.get(name, "gray"), linewidth=2)

    # 沪深300
    bm = benchmark[benchmark.index >= common_start] if common_start else benchmark
    bm_nav = (1 + bm.fillna(0)).cumprod()
    ax.plot([p.to_timestamp() for p in bm_nav.index], bm_nav.values,
            label="HS300", color="black", linewidth=1.5, linestyle="--")

    ax.set_title("ML Model NAV Comparison (Walk-Forward, Common Window)")
    ax.set_ylabel("Cumulative NAV")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()

    path = os.path.join(save_dir, "model_comparison_nav.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] 净值对比图已保存: {path}")


# =====================================================
#  主流程
# =====================================================

def main():
    save_dir = os.path.join(OUTPUT_DIR, "ml")
    os.makedirs(save_dir, exist_ok=True)

    print("[model_comparison] 加载数据...")
    panel = load_factor_panel()
    benchmark = load_benchmark_returns()
    features = [c for c in FACTOR_COLS if c in panel.columns]

    perf_rows = []
    strategy_series: dict[str, pd.Series] = {}

    for model_name, model in MODELS.items():
        print(f"\n{'='*55}")
        print(f"  [{model_name}] 开始 Walk-Forward 训练...")
        print(f"{'='*55}")

        predictions, monthly_ic, mean_imp = walk_forward_single_model(
            panel, model_name, model, features
        )

        if predictions.empty:
            print(f"  [WARN] {model_name} 无预测结果，跳过")
            continue

        # 保存预测
        pred_path = os.path.join(PROCESSED_DIR, f"predictions_{model_name.lower()}.parquet")
        predictions.to_parquet(pred_path, index=False)

        # IC 统计
        if monthly_ic:
            ic_arr = np.array(monthly_ic)
            print(f"  样本外 Rank IC: mean={ic_arr.mean():.4f}, "
                  f"std={ic_arr.std():.4f}, IR={ic_arr.mean()/max(ic_arr.std(), 1e-8):.3f}, "
                  f"IC>0: {(ic_arr>0).mean():.1%}, N={len(ic_arr)}")

        # 特征重要性
        if mean_imp is not None:
            plot_feature_importance(mean_imp, features, model_name, save_dir)

        # 组合回测
        strategy_rets, turnover = backtest_from_scores(predictions, panel, benchmark)
        strategy_series[model_name] = strategy_rets

        # 保存各模型月度收益 CSV（供 compare_strategies.py 加载）
        nav_df = pd.DataFrame({
            "year_month": strategy_rets.index.astype(str),
            "strategy_ret": strategy_rets.values,
        })
        nav_df.to_csv(
            os.path.join(save_dir, f"nav_{model_name.lower()}.csv"),
            index=False, encoding="utf-8-sig"
        )

        perf = summarize_returns(
            strategy_rets,
            benchmark_returns=benchmark.reindex(strategy_rets.index),
            turnover=turnover,
        )
        perf["Model"] = model_name
        perf_rows.append(perf)

        print(f"  组合绩效: 年化={perf['Ann_Return']:.2%}, "
              f"Sharpe={perf['Sharpe']:.3f}, "
              f"超额={perf.get('Excess_Ann_Return', float('nan')):.2%}, "
              f"IR={perf.get('Info_Ratio', float('nan')):.3f}, "
              f"换手={perf.get('Avg_Turnover', float('nan')):.1%}")

    # 汇总绩效表
    if perf_rows:
        comp_df = pd.DataFrame(perf_rows).set_index("Model")
        print("\n" + "=" * 65)
        print("  ML Model Performance Comparison")
        print("=" * 65)
        display = ["Ann_Return", "Sharpe", "Max_Drawdown", "Excess_Ann_Return",
                   "Info_Ratio", "Avg_Turnover"]
        print(comp_df[[c for c in display if c in comp_df.columns]].to_string(
            float_format="{:.4f}".format
        ))
        comp_df.to_csv(
            os.path.join(save_dir, "model_comparison.csv"), encoding="utf-8-sig"
        )

    # 净值对比图
    plot_model_nav_comparison(strategy_series, benchmark, save_dir)
    print(f"\n[model_comparison] 全部输出已保存至 {save_dir}/")


if __name__ == "__main__":
    main()
