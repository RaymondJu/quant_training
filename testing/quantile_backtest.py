# -*- coding: utf-8 -*-
"""
分层回测: 按因子值分5组, 计算等权组合月度收益

输出:
    - 5组累计收益曲线
    - 多空组合(Q5-Q1)净值曲线
    - 分年度分层效果
    - 各组Sharpe/最大回撤

Usage:
    python testing/quantile_backtest.py
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIVE_FACTOR_COLS, PROCESSED_DIR, OUTPUT_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FACTOR_COLS = ACTIVE_FACTOR_COLS
N_QUANTILES = 5


def load_factor_panel():
    path = os.path.join(PROCESSED_DIR, "factor_panel.parquet")
    df = pd.read_parquet(path)
    if "year_month" not in df.columns and "year_month_str" in df.columns:
        df["year_month"] = pd.PeriodIndex(df["year_month_str"], freq="M")
    return df


def quantile_backtest_single(panel, factor_col, n_q=N_QUANTILES):
    """
    对单个因子做分层回测

    Parameters
    ----------
    panel : pd.DataFrame
    factor_col : str
    n_q : int

    Returns
    -------
    quantile_returns : pd.DataFrame
        index=year_month, columns=Q1..Q5, values=等权月度收益
    """
    results = []
    for ym, grp in panel.groupby("year_month"):
        valid = grp[[factor_col, "ret_next_month"]].dropna()
        if len(valid) < n_q * 5:
            continue
        try:
            valid["quantile"] = pd.qcut(
                valid[factor_col], q=n_q, labels=False, duplicates="drop"
            ) + 1
        except ValueError:
            continue

        q_ret = valid.groupby("quantile")["ret_next_month"].mean()
        q_ret.name = ym
        results.append(q_ret)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.index.name = "year_month"
    df.columns = [f"Q{int(c)}" for c in df.columns]
    return df


def compute_quantile_stats(q_returns):
    """计算各分位组的绩效统计"""
    rows = []
    for col in q_returns.columns:
        r = q_returns[col].dropna()
        ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + r).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        rows.append({
            "Group": col,
            "Ann_Return": ann_ret,
            "Ann_Vol": ann_vol,
            "Sharpe": sharpe,
            "Max_Drawdown": max_dd,
            "Monthly_WinRate": (r > 0).mean(),
        })

    # 多空组合
    if "Q1" in q_returns.columns and f"Q{N_QUANTILES}" in q_returns.columns:
        ls = q_returns[f"Q{N_QUANTILES}"] - q_returns["Q1"]
        ls = ls.dropna()
        ann_ret = (1 + ls).prod() ** (12 / len(ls)) - 1
        ann_vol = ls.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + ls).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        rows.append({
            "Group": "L/S(Q5-Q1)",
            "Ann_Return": ann_ret,
            "Ann_Vol": ann_vol,
            "Sharpe": sharpe,
            "Max_Drawdown": max_dd,
            "Monthly_WinRate": (ls > 0).mean(),
        })

    return pd.DataFrame(rows).set_index("Group")


def plot_quantile_backtest(q_returns, factor_name, save_dir):
    """分层回测可视化: 累计收益曲线 + 多空净值"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    dates = [p.to_timestamp() for p in q_returns.index]
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(q_returns.columns)))

    # 累计收益曲线
    for i, col in enumerate(q_returns.columns):
        cum = (1 + q_returns[col]).cumprod()
        ax1.plot(dates, cum.values, label=col, color=colors[i], linewidth=1.5)

    ax1.set_title(f"{factor_name} - Quantile Cumulative Returns")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 多空净值
    if "Q1" in q_returns.columns and f"Q{N_QUANTILES}" in q_returns.columns:
        ls = q_returns[f"Q{N_QUANTILES}"] - q_returns["Q1"]
        cum_ls = (1 + ls).cumprod()
        ax2.plot(dates, cum_ls.values, color="#1976d2", linewidth=1.5)
        ax2.fill_between(dates, 1, cum_ls.values,
                         where=cum_ls.values >= 1, color="#1976d2", alpha=0.1)
        ax2.fill_between(dates, 1, cum_ls.values,
                         where=cum_ls.values < 1, color="#d32f2f", alpha=0.1)
        ax2.axhline(y=1, color="black", linestyle="--", alpha=0.3)
        sharpe = ls.mean() / ls.std() * np.sqrt(12) if ls.std() > 0 else 0
        ax2.set_title(f"{factor_name} - Long/Short (Q5-Q1), Sharpe={sharpe:.2f}")
    ax2.set_ylabel("Cumulative Return")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = os.path.join(save_dir, f"quantile_{factor_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def annual_quantile_returns(q_returns):
    """分年度各分位收益"""
    q_returns = q_returns.copy()
    q_returns["year"] = [p.year for p in q_returns.index]
    quantile_cols = [c for c in q_returns.columns if c != "year"]
    annual_rows = []
    for year, grp in q_returns.groupby("year"):
        annual_rows.append(pd.Series((1 + grp[quantile_cols]).prod() - 1, name=year))
    return pd.DataFrame(annual_rows)


def run_quantile_backtest():
    """执行全部因子的分层回测"""
    print("[quantile] 加载因子面板...")
    panel = load_factor_panel()
    available = [c for c in FACTOR_COLS if c in panel.columns]

    save_dir = os.path.join(OUTPUT_DIR, "quantile_backtest")
    os.makedirs(save_dir, exist_ok=True)

    all_stats = {}

    for factor in available:
        print(f"[quantile] {factor}...")
        q_ret = quantile_backtest_single(panel, factor)
        if q_ret.empty:
            print(f"  [WARN] {factor} 无有效数据, 跳过")
            continue

        stats = compute_quantile_stats(q_ret)
        all_stats[factor] = stats
        plot_quantile_backtest(q_ret, factor, save_dir)

        # 保存分年度数据
        annual = annual_quantile_returns(q_ret)
        annual.to_csv(
            os.path.join(save_dir, f"annual_{factor}.csv"), encoding="utf-8-sig"
        )

    # 汇总多空组合统计
    ls_rows = []
    for factor, stats in all_stats.items():
        if "L/S(Q5-Q1)" in stats.index:
            row = stats.loc["L/S(Q5-Q1)"].to_dict()
            row["Factor"] = factor
            ls_rows.append(row)

    if ls_rows:
        ls_summary = pd.DataFrame(ls_rows).set_index("Factor")
        print("\n" + "=" * 70)
        print("  Long/Short (Q5-Q1) Summary")
        print("=" * 70)
        print(ls_summary.to_string(float_format="{:.4f}".format))
        ls_summary.to_csv(
            os.path.join(save_dir, "long_short_summary.csv"), encoding="utf-8-sig"
        )

    print(f"\n[quantile] 结果已保存至 {save_dir}/")
    return all_stats


if __name__ == "__main__":
    run_quantile_backtest()
