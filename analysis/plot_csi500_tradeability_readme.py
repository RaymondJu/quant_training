from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_UNIVERSE", "csi500")

from csi500_daily_alpha_pipeline import _forward_compound_return
from data.benchmark import load_benchmark_daily_returns


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue": {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
    "neutral": {"base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"},
}

FONT_FAMILY = [
    "Microsoft YaHei",
    "Microsoft JhengHei",
    "Noto Sans CJK SC",
    "SimHei",
    "Segoe UI",
    "DejaVu Sans",
    "sans-serif",
]
MONO_FONT_FAMILY = ["Microsoft YaHei", "Consolas", "DejaVu Sans Mono", "monospace"]

VALIDATION_DIR = ROOT / "output" / "csi500" / "daily_alpha" / "trade_constraint_validation"
OUTPUT_DIR = ROOT / "docs" / "readme_assets"

RETURN_FILES = {
    "IC-weight": "ic_weight_returns.csv",
    "IC-weight size-neutral": "ic_weight_size_neutral_returns.csv",
    "Ridge": "ridge_returns.csv",
    "Ridge size-neutral": "ridge_size_neutral_returns.csv",
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

STRATEGY_LABELS = {
    "IC-weight": "IC 加权",
    "IC-weight size-neutral": "IC 加权（市值中性）",
    "Ridge": "Ridge",
    "Ridge size-neutral": "Ridge（市值中性）",
    "RidgeCV": "RidgeCV",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "RandomForest": "随机森林",
    "LightGBM Optuna": "LightGBM Optuna",
    "XGBoost Optuna": "XGBoost Optuna",
    "CatBoost Optuna": "CatBoost Optuna",
    "RandomForest Optuna": "随机森林 Optuna",
}


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "svg.fonttype": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        },
    )


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=78, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=112, break_long_words=False)
    title_lines = title.count("\n") + 1
    fig.subplots_adjust(top=max(0.64, 0.73 - 0.04 * (title_lines - 1)))
    left = ax.get_position().x0
    fig.text(
        left,
        0.975,
        title,
        ha="left",
        va="top",
        fontsize=19,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.925 - 0.04 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=11.5,
        color=TOKENS["muted"],
    )


def save_figure(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_text = re.sub(r"<!--.*?-->", "", svg_text, flags=re.DOTALL)
    svg_text = re.sub(r"<metadata>.*?</metadata>", "", svg_text, flags=re.DOTALL)
    svg_text = re.sub(
        r"(?<![\w#])(-?\d+\.\d{3,})",
        lambda match: f"{float(match.group(1)):.2f}".rstrip("0").rstrip("."),
        svg_text,
    )
    svg_text = re.sub(r"\s+", " ", svg_text).strip()
    svg_path.write_text(
        svg_text + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def load_returns() -> dict[str, pd.DataFrame]:
    frames = {}
    for label, filename in RETURN_FILES.items():
        path = VALIDATION_DIR / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        frames[label] = frame.sort_values("date")
    return frames


def plot_latest_nav(frames: dict[str, pd.DataFrame]) -> None:
    selected_labels = [
        "Ridge",
        "LightGBM",
        "XGBoost",
        "CatBoost",
        "RandomForest",
    ]
    selected = {label: frames[label] for label in selected_labels}
    common_start = max(frame["date"].min() for frame in selected.values())
    common_end = min(frame["date"].max() for frame in selected.values())

    nav_rows = []
    for label, frame in selected.items():
        part = frame.loc[frame["date"].between(common_start, common_end), ["date", "strategy_ret"]].copy()
        part["nav"] = (1.0 + part["strategy_ret"]).cumprod()
        part["series"] = STRATEGY_LABELS[label]
        nav_rows.append(part[["date", "nav", "series"]])

    benchmark = load_benchmark_daily_returns().copy()
    benchmark["benchmark_fwd_5d"] = _forward_compound_return(benchmark["index_ret"], 5)
    ridge_dates = frames["Ridge"].loc[frames["Ridge"]["date"].between(common_start, common_end), "date"]
    benchmark = benchmark.set_index("date").reindex(ridge_dates).dropna(subset=["benchmark_fwd_5d"])
    benchmark_nav = pd.DataFrame(
        {
            "date": benchmark.index,
            "nav": (1.0 + benchmark["benchmark_fwd_5d"]).cumprod().values,
            "series": "中证 500 基准",
        }
    )
    plot_df = pd.concat([*nav_rows, benchmark_nav], ignore_index=True)

    styles = {
        "CatBoost": (COLORS["blue"]["mid"], "-", 2.2),
        "LightGBM": (COLORS["olive"]["mid"], "-", 1.6),
        "XGBoost": (COLORS["orange"]["mid"], "-", 1.6),
        "随机森林": (COLORS["pink"]["mid"], "-", 1.6),
        "Ridge": (COLORS["gold"]["mid"], "--", 1.8),
        "中证 500 基准": (COLORS["neutral"]["dark"], ":", 1.5),
    }

    fig, ax = plt.subplots(figsize=(11, 6.2))
    for label in styles:
        part = plot_df[plot_df["series"] == label]
        color, linestyle, linewidth = styles[label]
        ax.plot(
            part["date"],
            part["nav"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

    ax.set_ylabel("累计净值")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fx"))
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=3,
        borderaxespad=0,
        fontsize=10.5,
    )
    sns.despine(ax=ax)
    add_chart_header(
        fig,
        ax,
        "最新交易约束下的默认机器学习策略净值",
        f"共同对比区间：{common_start:%Y-%m-%d} 至 {common_end:%Y-%m-%d}；"
        "每 5 个交易日调仓，等权持有前 50 名，已扣除 30 个基点换手成本。",
    )
    save_figure(fig, "csi500_daily_latest_nav")


def plot_optuna_comparison(frames: dict[str, pd.DataFrame]) -> None:
    pairs = [
        ("LightGBM", "LightGBM Optuna"),
        ("XGBoost", "XGBoost Optuna"),
        ("CatBoost", "CatBoost Optuna"),
        ("RandomForest", "RandomForest Optuna"),
    ]
    common_start = max(frames[label]["date"].min() for pair in pairs for label in pair)
    common_end = min(frames[label]["date"].max() for pair in pairs for label in pair)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    for ax, (base_label, tuned_label) in zip(axes.flat, pairs):
        for label, color, linestyle, linewidth in [
            (base_label, COLORS["blue"]["mid"], "-", 2.0),
            (tuned_label, COLORS["orange"]["mid"], "--", 1.8),
        ]:
            part = frames[label].loc[
                frames[label]["date"].between(common_start, common_end),
                ["date", "strategy_ret"],
            ].copy()
            part["nav"] = (1.0 + part["strategy_ret"]).cumprod()
            legend_label = "默认参数" if label == base_label else "Optuna"
            ax.plot(
                part["date"],
                part["nav"],
                label=legend_label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        ax.set_title(STRATEGY_LABELS[base_label], fontsize=14, fontweight="semibold", loc="left")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fx"))
        ax.legend(frameon=False, fontsize=10.5, loc="upper left")
        locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        sns.despine(ax=ax)

    axes[0, 0].set_ylabel("累计净值")
    axes[1, 0].set_ylabel("累计净值")
    fig.subplots_adjust(top=0.79, hspace=0.28, wspace=0.16)
    left = axes[0, 0].get_position().x0
    fig.text(
        left,
        0.975,
        "默认参数与 Optuna 调参后的净值对比",
        ha="left",
        va="top",
        fontsize=19,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.925,
        "四组 Optuna 模型均未超过对应默认参数；验证期 Rank IC 最优不等于组合收益最优。",
        ha="left",
        va="top",
        fontsize=11.5,
        color=TOKENS["muted"],
    )
    save_figure(fig, "csi500_ml_default_vs_optuna")


def plot_constraint_impact() -> None:
    df = pd.read_csv(VALIDATION_DIR / "performance_before_after.csv")
    order = ["IC-weight", "IC-weight size-neutral", "Ridge", "Ridge size-neutral"]
    df = df.set_index("Label").loc[order].reset_index()
    df["策略"] = df["Label"].map(STRATEGY_LABELS)
    y = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6.6),
        gridspec_kw={"width_ratios": [1.12, 1.18]},
    )

    before = df["Ann_Return_Before"] * 100
    after = df["Ann_Return_After"] * 100
    ax1.hlines(y, after, before, color=COLORS["neutral"]["base"], linewidth=2)
    ax1.scatter(
        before,
        y,
        s=58,
        facecolors=TOKENS["panel"],
        edgecolors=COLORS["neutral"]["dark"],
        linewidths=1.2,
        label="加入卖出约束前",
        zorder=3,
    )
    ax1.scatter(
        after,
        y,
        s=62,
        facecolors=COLORS["orange"]["base"],
        edgecolors=COLORS["orange"]["dark"],
        linewidths=1.0,
        label="当前版本",
        zorder=4,
    )
    for i, row in df.iterrows():
        ax1.text(
            (before.iloc[i] + after.iloc[i]) / 2,
            i - 0.14,
            f"{row['Ann_Return_Change_pp']:+.2f} 个百分点",
            ha="center",
            va="center",
            fontsize=10,
            color=COLORS["orange"]["dark"],
            family=MONO_FONT_FAMILY[0],
        )
    ax1.set_yticks(y, df["策略"])
    ax1.invert_yaxis()
    ax1.set_xlabel("年化收益率")
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax1.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=2,
        borderaxespad=0,
        fontsize=10.5,
    )
    ax1.grid(axis="y", visible=False)

    suspension = df["Suspension_Locks"]
    limit_down = df["Limit_Down_Locks"]
    ax2.barh(
        y,
        suspension,
        color=COLORS["blue"]["base"],
        edgecolor=COLORS["blue"]["dark"],
        linewidth=1.0,
        label="停牌锁仓",
    )
    ax2.barh(
        y,
        limit_down,
        left=suspension,
        color=COLORS["orange"]["base"],
        edgecolor=COLORS["orange"]["dark"],
        linewidth=1.0,
        label="跌停锁仓",
    )
    for i, row in df.iterrows():
        total = row["Suspension_Locks"] + row["Limit_Down_Locks"]
        ax2.text(
            total + 5,
            i,
            f"{int(total)} 次锁仓 / {int(row['Locked_Rebalances'])} 次调仓受影响",
            ha="left",
            va="center",
            fontsize=10,
            color=TOKENS["muted"],
        )
    ax2.set_yticks(y, [])
    ax2.invert_yaxis()
    ax2.set_xlabel("持仓锁定事件数")
    ax2.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=2,
        borderaxespad=0,
        fontsize=10.5,
    )
    ax2.grid(axis="y", visible=False)
    sns.despine(ax=ax1)
    sns.despine(ax=ax2, left=True)
    add_chart_header(
        fig,
        ax1,
        "卖出约束对市值中性策略影响更明显",
        "对比加入停牌与跌停卖出约束前后的年化收益；锁仓按持仓逐只计数，"
        "因此锁仓次数可能高于受影响的调仓次数。",
    )
    fig.subplots_adjust(wspace=0.22)
    save_figure(fig, "csi500_tradeability_impact")


def main() -> None:
    use_chart_theme()
    frames = load_returns()
    plot_latest_nav(frames)
    plot_optuna_comparison(frames)
    plot_constraint_impact()
    print(f"Saved README charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
