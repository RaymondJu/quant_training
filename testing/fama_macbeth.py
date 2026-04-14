# -*- coding: utf-8 -*-
"""
Fama-MacBeth 截面回归

每期截面回归: R_i = a + b1*Factor1 + b2*Factor2 + ... + e_i
对系数时间序列取均值和 t 统计量 (Newey-West 调整)

输出:
    - 单因子 FM 回归结果
    - 多因子 FM 回归结果
    - 系数时间序列

Usage:
    python testing/fama_macbeth.py
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, OUTPUT_DIR

FACTOR_COLS = ["EP", "BP", "SP", "MOM_12_1", "REV_1M", "ROE_TTM",
               "GPM_change", "VOL_20D", "IVOL", "TURN_1M", "AMIHUD",
               "SIZE", "BETA_60D", "ABTURN_1M", "OCF_QUALITY", "ASSET_GROWTH"]


def load_factor_panel():
    path = os.path.join(PROCESSED_DIR, "factor_panel.parquet")
    df = pd.read_parquet(path)
    if "year_month" not in df.columns and "year_month_str" in df.columns:
        df["year_month"] = pd.PeriodIndex(df["year_month_str"], freq="M")
    if "in_universe" in df.columns:
        df = df[df["in_universe"]].copy()
    return df


def fama_macbeth_regression(panel, factor_cols, ret_col="ret_next_month", min_obs=30):
    """
    Fama-MacBeth 两步回归

    Step 1: 每期截面回归 R_i = a + sum(bk * Factork_i) + e_i
    Step 2: 对系数时间序列取均值, 用 Newey-West t 统计量

    Returns
    -------
    coef_series : pd.DataFrame
        index=year_month, columns=factor_cols + ['const']
    summary : pd.DataFrame
        index=factor_cols + ['const'], columns=[mean, t_stat, p_value, t_nw]
    """
    periods = sorted(panel["year_month"].unique())
    coef_list = []

    for ym in periods:
        cross = panel[panel["year_month"] == ym].copy()
        valid = cross[factor_cols + [ret_col]].dropna()
        if len(valid) < min_obs:
            continue

        X = sm.add_constant(valid[factor_cols])
        y = valid[ret_col]

        try:
            model = sm.OLS(y, X).fit()
            coefs = model.params.to_dict()
            coefs["year_month"] = ym
            coef_list.append(coefs)
        except Exception:
            continue

    if not coef_list:
        return pd.DataFrame(), pd.DataFrame()

    coef_df = pd.DataFrame(coef_list).set_index("year_month")

    # Step 2: Newey-West t 统计量
    all_cols = ["const"] + factor_cols
    rows = []
    for col in all_cols:
        if col not in coef_df.columns:
            continue
        series = coef_df[col].dropna()
        n = len(series)
        mean_val = series.mean()
        std_val = series.std()

        # 简单 t 统计量
        t_simple = mean_val / (std_val / np.sqrt(n)) if std_val > 0 else 0

        # Newey-West 调整 (lag = int(4*(n/100)^(2/9)))
        nw_lag = int(4 * (n / 100) ** (2 / 9))
        try:
            nw_result = sm.OLS(series.values, np.ones(n)).fit(
                cov_type="HAC", cov_kwds={"maxlags": nw_lag}
            )
            t_nw = nw_result.tvalues[0]
            p_nw = nw_result.pvalues[0]
        except Exception:
            t_nw = t_simple
            p_nw = 2 * (1 - sp_stats.t.cdf(abs(t_simple), df=n - 1))

        rows.append({
            "Variable": col,
            "Mean": mean_val,
            "Std": std_val,
            "T_simple": t_simple,
            "T_NW": t_nw,
            "P_NW": p_nw,
            "Significant": "***" if abs(t_nw) > 2.576 else
                          "**" if abs(t_nw) > 1.96 else
                          "*" if abs(t_nw) > 1.645 else "",
            "N_months": n,
        })

    summary = pd.DataFrame(rows).set_index("Variable")
    return coef_df, summary


def run_fama_macbeth():
    """执行 Fama-MacBeth 分析"""
    print("[FM] 加载因子面板...")
    panel = load_factor_panel()
    available = [c for c in FACTOR_COLS if c in panel.columns]

    save_dir = os.path.join(OUTPUT_DIR, "fama_macbeth")
    os.makedirs(save_dir, exist_ok=True)

    # === 1. 单因子 FM 回归 ===
    print("\n[FM] ====== 单因子 Fama-MacBeth 回归 ======")
    single_rows = []
    for factor in available:
        _, summary = fama_macbeth_regression(panel, [factor])
        if summary.empty:
            continue
        row = summary.loc[factor].to_dict()
        row["Factor"] = factor
        single_rows.append(row)

    single_summary = pd.DataFrame(single_rows).set_index("Factor")
    print("\n" + "=" * 70)
    print("  Single-Factor Fama-MacBeth Results")
    print("=" * 70)
    display_cols = ["Mean", "T_NW", "P_NW", "Significant"]
    print(single_summary[display_cols].to_string(float_format="{:.4f}".format))
    single_summary.to_csv(
        os.path.join(save_dir, "fm_single_factor.csv"), encoding="utf-8-sig"
    )

    # === 2. 多因子 FM 回归 ===
    print("\n[FM] ====== 多因子 Fama-MacBeth 回归 ======")
    coef_df, multi_summary = fama_macbeth_regression(panel, available)

    if not multi_summary.empty:
        print("\n" + "=" * 70)
        print("  Multi-Factor Fama-MacBeth Results")
        print("=" * 70)
        print(multi_summary[display_cols].to_string(float_format="{:.4f}".format))
        multi_summary.to_csv(
            os.path.join(save_dir, "fm_multi_factor.csv"), encoding="utf-8-sig"
        )
        coef_df.to_csv(
            os.path.join(save_dir, "fm_coef_timeseries.csv"), encoding="utf-8-sig"
        )

    # === 3. 因子相关性矩阵 ===
    print("\n[FM] 因子 Rank 相关系数矩阵...")
    factor_data = panel[available].dropna()
    corr_matrix = factor_data.corr(method="spearman")
    corr_matrix.to_csv(
        os.path.join(save_dir, "factor_correlation.csv"), encoding="utf-8-sig"
    )

    # 检查高相关因子对
    high_corr = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            c = corr_matrix.iloc[i, j]
            if abs(c) > 0.5:
                high_corr.append((available[i], available[j], c))
    if high_corr:
        print("  高相关因子对 (|corr| > 0.5):")
        for f1, f2, c in high_corr:
            print(f"    {f1} vs {f2}: {c:.4f}")
    else:
        print("  无高相关因子对 (|corr| > 0.5)")

    print(f"\n[FM] 结果已保存至 {save_dir}/")
    return single_summary, multi_summary


if __name__ == "__main__":
    run_fama_macbeth()
