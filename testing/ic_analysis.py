# -*- coding: utf-8 -*-
"""
IC分析: Rank IC (Spearman), IC均值, ICIR, IC>0比例, IC衰减曲线

对每个因子, 逐月计算因子值与下月收益率的截面 Rank 相关系数。

输出:
    - IC时间序列 DataFrame
    - IC汇总统计表
    - IC衰减曲线 (lag 1~6 个月)
    - 可视化图表保存至 output/

Usage:
    python testing/ic_analysis.py
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIVE_FACTOR_COLS, PROCESSED_DIR, OUTPUT_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FACTOR_COLS = ACTIVE_FACTOR_COLS


def load_factor_panel():
    path = os.path.join(PROCESSED_DIR, "factor_panel.parquet")
    df = pd.read_parquet(path)
    if "year_month" not in df.columns and "year_month_str" in df.columns:
        df["year_month"] = pd.PeriodIndex(df["year_month_str"], freq="M")
    return df


def compute_rank_ic_series(panel, factor_col, ret_col="ret_next_month"):
    """
    逐月计算 Rank IC (Spearman相关系数)

    Returns
    -------
    pd.Series : index=year_month, values=rank_ic
    """
    results = {}
    for ym, grp in panel.groupby("year_month"):
        valid = grp[[factor_col, ret_col]].dropna()
        if len(valid) < 20:
            continue
        if valid[factor_col].nunique() <= 1 or valid[ret_col].nunique() <= 1:
            continue
        ic, _ = stats.spearmanr(valid[factor_col], valid[ret_col])
        results[ym] = ic
    return pd.Series(results, name=factor_col)


def compute_ic_decay(panel, factor_col, max_lag=6):
    """
    IC衰减曲线: 因子值与未来 lag=1,2,...,max_lag 个月收益率的 Rank IC

    Returns
    -------
    pd.Series : index=lag, values=mean_rank_ic
    """
    df = panel[["stock_code", "year_month", factor_col, "ret_next_month"]].copy()
    df = df.sort_values(["stock_code", "year_month"])

    decay = {}
    for lag in range(1, max_lag + 1):
        # shift ret_next_month by (lag-1) additional periods
        col_name = f"ret_lag{lag}"
        df[col_name] = df.groupby("stock_code")["ret_next_month"].shift(-(lag - 1))
        ic_series = {}
        for ym, grp in df.groupby("year_month"):
            valid = grp[[factor_col, col_name]].dropna()
            if len(valid) < 20:
                continue
            if valid[factor_col].nunique() <= 1 or valid[col_name].nunique() <= 1:
                continue
            ic, _ = stats.spearmanr(valid[factor_col], valid[col_name])
            ic_series[ym] = ic
        if ic_series:
            decay[lag] = np.mean(list(ic_series.values()))
    return pd.Series(decay, name=factor_col)


def ic_summary_table(ic_dict):
    """
    生成IC汇总统计表

    Parameters
    ----------
    ic_dict : dict[str, pd.Series]
        因子名 → IC时间序列

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, ic_series in ic_dict.items():
        ic = ic_series.dropna()
        rows.append({
            "Factor": name,
            "IC_mean": ic.mean(),
            "IC_std": ic.std(),
            "ICIR": ic.mean() / ic.std() if ic.std() > 0 else 0,
            "IC>0_pct": (ic > 0).mean(),
            "IC_abs_mean": ic.abs().mean(),
            "T_stat": ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0,
            "N_months": len(ic),
        })
    return pd.DataFrame(rows).set_index("Factor")


def plot_ic_series(ic_dict, save_dir):
    """IC时间序列图 + 直方图"""
    n_factors = len(ic_dict)
    fig, axes = plt.subplots(n_factors, 2, figsize=(16, 3 * n_factors))
    if n_factors == 1:
        axes = axes.reshape(1, -1)

    for i, (name, ic_series) in enumerate(ic_dict.items()):
        ic = ic_series.dropna()
        dates = [p.to_timestamp() for p in ic.index]

        # IC 时间序列条形图
        ax = axes[i, 0]
        colors = ["#d32f2f" if v < 0 else "#1976d2" for v in ic.values]
        ax.bar(dates, ic.values, color=colors, alpha=0.7, width=25)
        ax.axhline(y=ic.mean(), color="black", linestyle="--", linewidth=1,
                   label=f"Mean={ic.mean():.4f}")
        icir = ic.mean() / ic.std() if ic.std() > 0 else 0
        ax.set_title(f"{name} Rank IC Series (ICIR={icir:.2f})")
        ax.legend(fontsize=8)
        ax.set_ylabel("Rank IC")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # IC 分布直方图
        ax2 = axes[i, 1]
        ax2.hist(ic.values, bins=30, color="#1976d2", alpha=0.7, edgecolor="white")
        ax2.axvline(x=ic.mean(), color="red", linestyle="--",
                    label=f"Mean={ic.mean():.4f}")
        ax2.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax2.set_title(f"{name} IC Distribution")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "ic_series_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ic] IC时间序列图已保存: {path}")


def plot_ic_decay(decay_dict, save_dir):
    """IC衰减曲线图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, decay in decay_dict.items():
        ax.plot(decay.index, decay.values, marker="o", label=name)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Mean Rank IC")
    ax.set_title("IC Decay Curve")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "ic_decay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ic] IC衰减曲线已保存: {path}")


def run_ic_analysis():
    """执行完整IC分析"""
    print("[ic] 加载因子面板...")
    panel = load_factor_panel()
    available = [c for c in FACTOR_COLS if c in panel.columns]

    # 1. 计算 Rank IC 时间序列
    print("[ic] 计算各因子 Rank IC 时间序列...")
    ic_dict = {}
    for col in available:
        ic_dict[col] = compute_rank_ic_series(panel, col)

    # 2. 汇总统计表
    summary = ic_summary_table(ic_dict)
    print("\n" + "=" * 70)
    print("  Rank IC Summary")
    print("=" * 70)
    print(summary.to_string(float_format="{:.4f}".format))

    # 3. IC衰减曲线
    print("\n[ic] 计算IC衰减曲线...")
    decay_dict = {}
    for col in available:
        decay_dict[col] = compute_ic_decay(panel, col)

    # 4. 可视化
    save_dir = os.path.join(OUTPUT_DIR, "ic_analysis")
    os.makedirs(save_dir, exist_ok=True)

    plot_ic_series(ic_dict, save_dir)
    plot_ic_decay(decay_dict, save_dir)

    # 5. 保存结果
    summary.to_csv(os.path.join(save_dir, "ic_summary.csv"), encoding="utf-8-sig")

    ic_df = pd.DataFrame(ic_dict)
    ic_df.index.name = "year_month"
    ic_df.to_csv(os.path.join(save_dir, "ic_timeseries.csv"), encoding="utf-8-sig")

    print(f"\n[ic] 结果已保存至 {save_dir}/")
    return summary, ic_dict, decay_dict


if __name__ == "__main__":
    run_ic_analysis()
