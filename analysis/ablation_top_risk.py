# -*- coding: utf-8 -*-
"""
Ablation 实验：顶部风险过滤层效果对比

统一实验窗口：2017-10 ~ 2025-11（98 个月）
注：全部 baseline/ML 使用同一窗口，避免口径混淆（参见 ML_WINDOW_AUDIT.md）

对比组合：
  ICIR-weight (no filter)      vs  ICIR-weight + TopRisk(20%)
  Ridge       (no filter)      vs  Ridge       + TopRisk(20%)
  XGBoost     (no filter)      vs  XGBoost     + TopRisk(20%)

输出文件（output/ablation/）:
  performance_comparison.csv  — 绩效汇总（含 Calmar）
  nav_comparison.png          — 6条NAV + benchmark
  drawdown_comparison.png     — 回撤曲线 + 2018/2022/2024标注
  excluded_stocks_log.csv     — 每月被剔除股票记录
  sensitivity.png             — Sharpe/Calmar 对剔除比例的敏感性

Usage:
    python analysis/ablation_top_risk.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INDEX_NAME, OUTPUT_DIR, PROCESSED_DIR, RAW_DIR,
    TOP_N_STOCKS, TRANSACTION_COST, TOP_RISK_FILTER_PCT,
)
from ml.model_comparison import backtest_from_scores, load_benchmark_returns
from portfolio.backtest import run_portfolio_backtest
from portfolio.performance import summarize_returns
from portfolio.risk_filter import apply_top_risk_filter

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ABLATION_START = pd.Period("2017-10", freq="M")
ABLATION_END   = pd.Period("2025-11", freq="M")
SAVE_DIR       = os.path.join(OUTPUT_DIR, "ablation")


# =====================================================================
#  数据加载
# =====================================================================

def load_risk_panel() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "factor_risk.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"风控因子文件不存在: {path}\n"
            "请先运行: python factors/risk.py"
        )
    df = pd.read_parquet(path)
    if not hasattr(df["year_month"].dtype, "freq"):
        df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df


def load_predictions(model_name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"predictions_{model_name.lower()}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"预测文件不存在: {path}\n"
            "请先运行: python ml/model_comparison.py"
        )
    df = pd.read_parquet(path)
    if not hasattr(df["year_month"].dtype, "freq"):
        df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df


def load_factor_panel() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "factor_panel.parquet")
    df = pd.read_parquet(path)
    if not hasattr(df["year_month"].dtype, "freq"):
        df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df


# =====================================================================
#  绩效计算（含 Calmar）
# =====================================================================

def compute_performance(
    rets: pd.Series,
    benchmark: pd.Series,
    turnover: pd.Series | None = None,
    label: str = "",
) -> dict:
    """计算绩效指标，在 summarize_returns 基础上追加 Calmar。"""
    # 截取统一窗口
    rets = rets[(rets.index >= ABLATION_START) & (rets.index <= ABLATION_END)]
    bm   = benchmark.reindex(rets.index)
    to   = turnover.reindex(rets.index) if turnover is not None else None

    perf = summarize_returns(rets, benchmark_returns=bm, turnover=to)
    perf_dict = perf.to_dict()

    ann_ret  = perf_dict.get("Ann_Return", np.nan)
    max_dd   = perf_dict.get("Max_Drawdown", np.nan)
    calmar   = ann_ret / abs(max_dd) if (pd.notna(max_dd) and max_dd != 0) else np.nan
    perf_dict["Calmar"] = calmar
    perf_dict["Label"]  = label
    perf_dict["N_months"] = len(rets)

    return perf_dict


# =====================================================================
#  各策略回测
# =====================================================================

def run_icir_strategies(
    risk_panel: pd.DataFrame,
    benchmark: pd.Series,
    log_records_with: list,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """返回 (rets_no_filter, to_no_filter, rets_with_filter, to_with_filter)"""
    print("\n[ablation] ICIR-weight baseline (no filter)...")
    # 关闭过滤：risk_panel=None
    import config as cfg
    orig_flag = cfg.ENABLE_TOP_RISK_FILTER
    cfg.ENABLE_TOP_RISK_FILTER = False
    result_no, _ = run_portfolio_backtest(method="icir", risk_panel=None,
                                          save_root="ablation/icir_no_filter")
    cfg.ENABLE_TOP_RISK_FILTER = orig_flag

    rets_no = result_no.set_index("year_month")["net_ret"]
    to_no   = result_no.set_index("year_month")["turnover"]
    rets_no.index = pd.PeriodIndex(rets_no.index, freq="M")
    to_no.index   = pd.PeriodIndex(to_no.index,   freq="M")

    print("[ablation] ICIR-weight baseline (with filter)...")
    cfg.ENABLE_TOP_RISK_FILTER = True
    log_path = os.path.join(SAVE_DIR, "excluded_stocks_log.csv")
    result_with, _ = run_portfolio_backtest(
        method="icir", risk_panel=risk_panel, log_path=log_path,
        save_root="ablation/icir_with_filter",
    )
    cfg.ENABLE_TOP_RISK_FILTER = orig_flag

    rets_with = result_with.set_index("year_month")["net_ret"]
    to_with   = result_with.set_index("year_month")["turnover"]
    rets_with.index = pd.PeriodIndex(rets_with.index, freq="M")
    to_with.index   = pd.PeriodIndex(to_with.index,   freq="M")

    return rets_no, to_no, rets_with, to_with


def run_ml_strategy(
    model_name: str,
    factor_panel: pd.DataFrame,
    benchmark: pd.Series,
    risk_panel: pd.DataFrame | None,
    filter_pct: float,
    log_records: list | None,
) -> tuple[pd.Series, pd.Series]:
    """运行单个 ML 策略，返回 (rets, turnover)。"""
    predictions = load_predictions(model_name)
    rets, to = backtest_from_scores(
        predictions, factor_panel, benchmark,
        risk_panel=risk_panel,
        filter_pct=filter_pct,
        log_records=log_records,
    )
    return rets, to


# =====================================================================
#  可视化
# =====================================================================

def _to_nav(rets: pd.Series, start: pd.Period, end: pd.Period) -> tuple[list, list]:
    """截取窗口后转 NAV，返回 (dates_list, nav_array)。"""
    sliced = rets[(rets.index >= start) & (rets.index <= end)]
    nav = (1 + sliced.fillna(0)).cumprod()
    dates = [p.to_timestamp() for p in nav.index]
    return dates, nav.values


def _to_drawdown(rets: pd.Series, start: pd.Period, end: pd.Period) -> tuple[list, list]:
    sliced = rets[(rets.index >= start) & (rets.index <= end)]
    nav = (1 + sliced.fillna(0)).cumprod()
    peak = nav.cummax()
    dd = (nav / peak - 1)
    dates = [p.to_timestamp() for p in dd.index]
    return dates, dd.values


STRATEGY_STYLES = {
    "ICIR (no filter)":    {"color": "#1976d2", "ls": "--", "lw": 1.6},
    "ICIR + TopRisk":      {"color": "#1976d2", "ls": "-",  "lw": 2.0},
    "Ridge (no filter)":   {"color": "#9c27b0", "ls": "--", "lw": 1.6},
    "Ridge + TopRisk":     {"color": "#9c27b0", "ls": "-",  "lw": 2.0},
    "XGBoost (no filter)": {"color": "#00838f", "ls": "--", "lw": 1.6},
    "XGBoost + TopRisk":   {"color": "#00838f", "ls": "-",  "lw": 2.0},
}

# 大回撤标注区间
DRAWDOWN_WINDOWS = [
    ("2018-01", "2018-12", "2018 bear"),
    ("2021-12", "2022-10", "2022 bear"),
    ("2024-01", "2024-09", "2024 correction"),
]


def plot_nav_comparison(all_rets: dict[str, pd.Series], benchmark: pd.Series):
    fig, ax = plt.subplots(figsize=(14, 6))

    bm = benchmark[(benchmark.index >= ABLATION_START) & (benchmark.index <= ABLATION_END)]
    bm_nav = (1 + bm.fillna(0)).cumprod()
    ax.plot([p.to_timestamp() for p in bm_nav.index], bm_nav.values,
            color="gray", lw=1.4, ls=":", label=INDEX_NAME)

    for name, rets in all_rets.items():
        dates, nav = _to_nav(rets, ABLATION_START, ABLATION_END)
        style = STRATEGY_STYLES.get(name, {"color": "black", "ls": "-", "lw": 1.5})
        ax.plot(dates, nav, label=name, **style)

    ax.set_title(f"Strategy NAV Comparison with Top Risk Filter\n"
                 f"({ABLATION_START} ~ {ABLATION_END}, {TOP_RISK_FILTER_PCT:.0%} filter)")
    ax.set_ylabel("Cumulative NAV")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "nav_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablation] NAV图已保存: {path}")


def plot_drawdown_comparison(all_rets: dict[str, pd.Series], benchmark: pd.Series):
    fig, ax = plt.subplots(figsize=(14, 6))

    # 标注大回撤区间
    for ws, we, label in DRAWDOWN_WINDOWS:
        wstart = pd.Period(ws, freq="M").to_timestamp()
        wend   = pd.Period(we, freq="M").to_timestamp()
        ax.axvspan(wstart, wend, alpha=0.08, color="red")
        ax.text(wstart, -0.02, label, fontsize=7, color="darkred", va="top")

    bm = benchmark[(benchmark.index >= ABLATION_START) & (benchmark.index <= ABLATION_END)]
    bm_nav = (1 + bm.fillna(0)).cumprod()
    bm_peak = bm_nav.cummax()
    bm_dd = bm_nav / bm_peak - 1
    ax.fill_between([p.to_timestamp() for p in bm_dd.index], bm_dd.values, 0,
                    color="gray", alpha=0.15, label=INDEX_NAME)

    for name, rets in all_rets.items():
        dates, dd = _to_drawdown(rets, ABLATION_START, ABLATION_END)
        style = STRATEGY_STYLES.get(name, {"color": "black", "ls": "-", "lw": 1.5})
        ax.plot(dates, dd, label=name, **style)

    ax.set_title("Drawdown Comparison (2018 / 2022 / 2024 bear windows highlighted)")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "drawdown_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablation] 回撤图已保存: {path}")


# =====================================================================
#  敏感性分析
# =====================================================================

def sensitivity_analysis(
    factor_panel: pd.DataFrame,
    risk_panel: pd.DataFrame,
    benchmark: pd.Series,
):
    """遍历 filter_pct，记录 Sharpe / Calmar 变化。"""
    print("\n[ablation] 开始敏感性分析...")
    filter_pcts = [0.10, 0.15, 0.20, 0.25, 0.30]
    results: dict[str, list] = {
        "filter_pct": filter_pcts,
        "ICIR_Sharpe": [], "ICIR_Calmar": [],
        "Ridge_Sharpe": [], "Ridge_Calmar": [],
    }

    # ICIR baseline — 先跑一次无过滤版本（不受 pct 影响）
    import config as cfg

    for pct in filter_pcts:
        print(f"  filter_pct={pct:.0%}...")

        # ICIR
        orig_flag = cfg.ENABLE_TOP_RISK_FILTER
        cfg.ENABLE_TOP_RISK_FILTER = True
        orig_pct = cfg.TOP_RISK_FILTER_PCT
        cfg.TOP_RISK_FILTER_PCT = pct
        result_icir, _ = run_portfolio_backtest(
            method="icir", risk_panel=risk_panel,
            save_root=f"ablation/sensitivity/icir_{int(pct*100)}pct",
        )
        cfg.ENABLE_TOP_RISK_FILTER = orig_flag
        cfg.TOP_RISK_FILTER_PCT = orig_pct

        rets_icir = result_icir.set_index("year_month")["net_ret"]
        rets_icir.index = pd.PeriodIndex(rets_icir.index, freq="M")
        perf_icir = compute_performance(rets_icir, benchmark, label="ICIR")
        results["ICIR_Sharpe"].append(perf_icir.get("Sharpe", np.nan))
        results["ICIR_Calmar"].append(perf_icir.get("Calmar", np.nan))

        # Ridge
        preds_ridge = load_predictions("ridge")
        rets_ridge, to_ridge = backtest_from_scores(
            preds_ridge, factor_panel, benchmark,
            risk_panel=risk_panel, filter_pct=pct,
        )
        perf_ridge = compute_performance(rets_ridge, benchmark, to_ridge, label="Ridge")
        results["Ridge_Sharpe"].append(perf_ridge.get("Sharpe", np.nan))
        results["Ridge_Calmar"].append(perf_ridge.get("Calmar", np.nan))

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pct_labels = [f"{p:.0%}" for p in filter_pcts]

    for ax, metric, title in [
        (axes[0], "Sharpe", "Sharpe Ratio vs Filter %"),
        (axes[1], "Calmar", "Calmar Ratio vs Filter %"),
    ]:
        ax.plot(pct_labels, results[f"ICIR_{metric}"], "o-", color="#1976d2",
                label="ICIR + TopRisk", linewidth=2)
        ax.plot(pct_labels, results[f"Ridge_{metric}"], "s-", color="#9c27b0",
                label="Ridge + TopRisk", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Top Risk Filter %")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Sensitivity Analysis: Top Risk Filter Percentage", fontsize=12)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "sensitivity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablation] 敏感性分析图已保存: {path}")

    # 保存数据
    sens_df = pd.DataFrame(results)
    sens_df.to_csv(os.path.join(SAVE_DIR, "sensitivity.csv"), index=False, encoding="utf-8-sig")
    return sens_df


# =====================================================================
#  主流程
# =====================================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[ablation] 统一实验窗口: {ABLATION_START} ~ {ABLATION_END} (98个月)")
    print(f"[ablation] 风控剔除比例: {TOP_RISK_FILTER_PCT:.0%}")
    print(f"[ablation] 注: 已知幸存者偏差（静态{INDEX_NAME}成分股），此处不修复")

    risk_panel  = load_risk_panel()
    factor_panel = load_factor_panel()
    benchmark   = load_benchmark_returns()

    all_rets: dict[str, pd.Series] = {}
    all_to:   dict[str, pd.Series] = {}
    log_records_filter: list = []

    # ---- ICIR ----
    rets_icir_no, to_icir_no, rets_icir_with, to_icir_with = run_icir_strategies(
        risk_panel, benchmark, log_records_filter
    )
    all_rets["ICIR (no filter)"] = rets_icir_no
    all_rets["ICIR + TopRisk"]   = rets_icir_with
    all_to["ICIR (no filter)"]   = to_icir_no
    all_to["ICIR + TopRisk"]     = to_icir_with

    # ---- Ridge ----
    print("\n[ablation] Ridge (no filter)...")
    rets_ridge_no, to_ridge_no = run_ml_strategy(
        "ridge", factor_panel, benchmark, risk_panel=None, filter_pct=0.0, log_records=None
    )
    print("[ablation] Ridge + TopRisk...")
    rets_ridge_with, to_ridge_with = run_ml_strategy(
        "ridge", factor_panel, benchmark, risk_panel, TOP_RISK_FILTER_PCT, log_records_filter
    )
    all_rets["Ridge (no filter)"] = rets_ridge_no
    all_rets["Ridge + TopRisk"]   = rets_ridge_with
    all_to["Ridge (no filter)"]   = to_ridge_no
    all_to["Ridge + TopRisk"]     = to_ridge_with

    # ---- XGBoost ----
    print("\n[ablation] XGBoost (no filter)...")
    rets_xgb_no, to_xgb_no = run_ml_strategy(
        "xgboost", factor_panel, benchmark, risk_panel=None, filter_pct=0.0, log_records=None
    )
    print("[ablation] XGBoost + TopRisk...")
    rets_xgb_with, to_xgb_with = run_ml_strategy(
        "xgboost", factor_panel, benchmark, risk_panel, TOP_RISK_FILTER_PCT, log_records_filter
    )
    all_rets["XGBoost (no filter)"] = rets_xgb_no
    all_rets["XGBoost + TopRisk"]   = rets_xgb_with
    all_to["XGBoost (no filter)"]   = to_xgb_no
    all_to["XGBoost + TopRisk"]     = to_xgb_with

    # ---- 保存 excluded_stocks_log ----
    if log_records_filter:
        log_df = pd.DataFrame(log_records_filter)
        log_path = os.path.join(SAVE_DIR, "excluded_stocks_log.csv")
        log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
        print(f"[ablation] 剔除记录已保存: {log_path} ({len(log_df)} 条)")

    # ---- 绩效汇总 ----
    print("\n[ablation] 计算绩效...")
    perf_rows = []
    for name, rets in all_rets.items():
        to = all_to.get(name)
        pd_to = pd.Series(to.values, index=to.index) if to is not None else None
        p = compute_performance(rets, benchmark, pd_to, label=name)
        perf_rows.append(p)

    perf_df = pd.DataFrame(perf_rows).set_index("Label")
    display_cols = [
        "Ann_Return", "Ann_Vol", "Sharpe", "Max_Drawdown", "Calmar",
        "Monthly_WinRate", "Excess_Ann_Return", "Info_Ratio", "Avg_Turnover", "N_months"
    ]
    disp = perf_df[[c for c in display_cols if c in perf_df.columns]]

    print("\n" + "=" * 75)
    print(f"  Ablation: Top Risk Filter ({TOP_RISK_FILTER_PCT:.0%}), Window: {ABLATION_START}~{ABLATION_END}")
    print("=" * 75)
    for row in disp.itertuples():
        print(f"  {row.Index:<25} Ann={row.Ann_Return:.2%}  Sharpe={row.Sharpe:.3f}"
              f"  MaxDD={row.Max_Drawdown:.2%}  Calmar={row.Calmar:.3f}")

    perf_df.to_csv(os.path.join(SAVE_DIR, "performance_comparison.csv"), encoding="utf-8-sig")
    print(f"\n[ablation] 绩效表已保存: {SAVE_DIR}/performance_comparison.csv")

    # ---- 图表 ----
    plot_nav_comparison(all_rets, benchmark)
    plot_drawdown_comparison(all_rets, benchmark)

    # ---- 敏感性分析 ----
    sensitivity_analysis(factor_panel, risk_panel, benchmark)

    print(f"\n[ablation] 全部输出已保存至 {SAVE_DIR}/")


if __name__ == "__main__":
    main()
