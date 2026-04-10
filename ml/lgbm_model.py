# -*- coding: utf-8 -*-
"""
LightGBM 滚动预测模型

Walk-forward训练：
  - 训练窗口：过去 LGBM_TRAIN_WINDOW 个月 (24)
  - 验证窗口：紧接 3 个月（用于 early stopping）
  - 预测：验证期的下一个月

输出：
  data/processed/lgbm_predictions.parquet  → stock_code, year_month, lgbm_score
  output/ml/shap_importance.csv            → 跨期均值 SHAP 重要性
  output/ml/shap_importance.png

Usage:
    python ml/lgbm_model.py
"""
from __future__ import annotations

import os
import sys

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LGBM_TRAIN_WINDOW, OUTPUT_DIR, PROCESSED_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FACTOR_COLS = [
    "EP", "BP", "SP", "MOM_12_1", "REV_1M", "ROE_TTM",
    "GPM_change", "VOL_20D", "IVOL", "TURN_1M", "AMIHUD",
    "SIZE", "BETA_60D", "ABTURN_1M", "OCF_QUALITY", "ASSET_GROWTH",
]
VAL_WINDOW = 3          # validation months for early stopping
EARLY_STOPPING = 30
IMPORTANCE_LAST_N = 12  # average feature importance over last N months

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


def load_factor_panel() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "factor_panel.parquet")
    df = pd.read_parquet(path)
    if not hasattr(df["year_month"].dtype, "freq"):
        df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df


def _split_windows(
    all_months: list,
    pred_idx: int,
    train_window: int,
    val_window: int,
) -> tuple[list, list, object]:
    """返回 (train_months, val_months, pred_month)"""
    pred_month = all_months[pred_idx]
    val_months = all_months[pred_idx - val_window: pred_idx]
    train_months = all_months[pred_idx - val_window - train_window: pred_idx - val_window]
    return train_months, val_months, pred_month


def run_walk_forward() -> tuple[pd.DataFrame, dict]:
    """
    走步前向训练并产出样本外预测分数。

    Returns
    -------
    predictions : pd.DataFrame  [stock_code, year_month, lgbm_score]
    shap_dict   : dict          {month -> shap_values array}  (last SHAP_LAST_N months)
    """
    panel = load_factor_panel()
    features = [c for c in FACTOR_COLS if c in panel.columns]
    all_months = sorted(panel["year_month"].unique())

    # 第一个可预测月的索引
    start_idx = LGBM_TRAIN_WINDOW + VAL_WINDOW
    pred_months = all_months[start_idx:]

    print(f"[lgbm] 共 {len(all_months)} 个月, 样本外预测 {len(pred_months)} 个月")
    print(f"[lgbm] 预测区间: {pred_months[0]} ~ {pred_months[-1]}")
    print(f"[lgbm] 特征: {features}")

    all_preds = []
    importance_list: list[np.ndarray] = []
    monthly_ic: list[float] = []

    # 只对最后 N 个月计入重要性（前期样本少时数值不稳定）
    importance_start_idx = max(0, len(pred_months) - IMPORTANCE_LAST_N)

    for pred_month in pred_months:
        pred_idx = all_months.index(pred_month)
        train_months, val_months, _ = _split_windows(
            all_months, pred_idx, LGBM_TRAIN_WINDOW, VAL_WINDOW
        )

        train_data = panel[panel["year_month"].isin(train_months)].dropna(
            subset=features + ["ret_next_month"]
        )
        val_data = panel[panel["year_month"].isin(val_months)].dropna(
            subset=features + ["ret_next_month"]
        )
        pred_data = panel[panel["year_month"] == pred_month].dropna(subset=features)

        if len(train_data) < 50 or len(val_data) < 10 or pred_data.empty:
            continue

        X_train, y_train = train_data[features], train_data["ret_next_month"]
        X_val, y_val = val_data[features], val_data["ret_next_month"]
        X_pred = pred_data[features]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        scores = model.predict(X_pred)
        pred_df = pred_data[["stock_code", "year_month"]].copy()
        pred_df["lgbm_score"] = scores
        all_preds.append(pred_df)

        # 月度样本外 IC
        if "ret_next_month" in pred_data.columns:
            merged = pred_data[["ret_next_month"]].copy()
            merged["lgbm_score"] = scores
            merged = merged.dropna()
            if len(merged) > 10:
                ic = merged["lgbm_score"].corr(merged["ret_next_month"], method="spearman")
                monthly_ic.append(ic)

        # 记录特征重要性（gain，最后 N 个月）
        month_pos = pred_months.index(pred_month)
        if month_pos >= importance_start_idx:
            importance_list.append(model.booster_.feature_importance(importance_type="gain"))

    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    if monthly_ic:
        ic_arr = np.array(monthly_ic)
        print(f"\n[lgbm] 样本外 Rank IC: mean={ic_arr.mean():.4f}, "
              f"std={ic_arr.std():.4f}, IR={ic_arr.mean()/ic_arr.std():.3f}, "
              f"IC>0: {(ic_arr>0).mean():.1%}")

    return predictions, {"importance_list": importance_list, "features": features}


def save_feature_importance(importance_data: dict, save_dir: str) -> pd.DataFrame:
    """计算跨期平均 Gain 重要性，保存图表和CSV。"""
    imp_list = importance_data.get("importance_list", [])
    features = importance_data.get("features", [])
    if not imp_list or not features:
        print("[lgbm] 无特征重要性数据")
        return pd.DataFrame()

    mean_importance = np.mean(imp_list, axis=0)
    importance_df = pd.DataFrame({
        "Feature": features,
        "Mean_Gain": mean_importance,
    }).sort_values("Mean_Gain", ascending=False)

    importance_df.to_csv(
        os.path.join(save_dir, "shap_importance.csv"), index=False, encoding="utf-8-sig"
    )

    # 图表
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_bar = ["#d32f2f" if i == 0 else "#1976d2" for i in range(len(importance_df))]
    ax.barh(importance_df["Feature"][::-1], importance_df["Mean_Gain"][::-1], color=colors_bar[::-1])
    ax.set_xlabel("Mean Gain Importance")
    ax.set_title(f"LightGBM Feature Importance (Gain, last {IMPORTANCE_LAST_N} months)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "shap_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[lgbm] 特征重要性 (Gain) Top5:")
    print(importance_df.head(5).to_string(index=False))

    return importance_df


def main():
    save_dir = os.path.join(OUTPUT_DIR, "ml")
    os.makedirs(save_dir, exist_ok=True)

    print("[lgbm] 开始 Walk-Forward 训练...")
    predictions, importance_data = run_walk_forward()

    if predictions.empty:
        print("[lgbm] ERROR: 无预测结果")
        return

    out_path = os.path.join(PROCESSED_DIR, "lgbm_predictions.parquet")
    predictions.to_parquet(out_path, index=False)
    print(f"\n[lgbm] 预测结果已保存: {out_path}")
    print(f"  共 {len(predictions)} 条记录, {predictions['year_month'].nunique()} 个月")

    save_feature_importance(importance_data, save_dir)
    print(f"\n[lgbm] 全部输出已保存至 {save_dir}/")


if __name__ == "__main__":
    main()
