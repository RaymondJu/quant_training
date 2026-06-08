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

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["Consolas", "DejaVu Sans Mono", "monospace"]

VALIDATION_DIR = ROOT / "output" / "csi500" / "daily_alpha" / "trade_constraint_validation"
OUTPUT_DIR = ROOT / "docs" / "readme_assets"

RETURN_FILES = {
    "IC-weight": "ic_weight_returns.csv",
    "IC-weight size-neutral": "ic_weight_size_neutral_returns.csv",
    "Ridge": "ridge_returns.csv",
    "Ridge size-neutral": "ridge_size_neutral_returns.csv",
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
        fontsize=15,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.925 - 0.04 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
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
        frame = pd.read_csv(VALIDATION_DIR / filename, parse_dates=["date"])
        frames[label] = frame.sort_values("date")
    return frames


def plot_latest_nav(frames: dict[str, pd.DataFrame]) -> None:
    common_start = max(frame["date"].min() for frame in frames.values())
    common_end = min(frame["date"].max() for frame in frames.values())

    nav_rows = []
    for label, frame in frames.items():
        part = frame.loc[frame["date"].between(common_start, common_end), ["date", "strategy_ret"]].copy()
        part["nav"] = (1.0 + part["strategy_ret"]).cumprod()
        part["series"] = label
        nav_rows.append(part[["date", "nav", "series"]])

    benchmark = load_benchmark_daily_returns().copy()
    benchmark["benchmark_fwd_5d"] = _forward_compound_return(benchmark["index_ret"], 5)
    ridge_dates = frames["Ridge"].loc[
        frames["Ridge"]["date"].between(common_start, common_end), "date"
    ]
    benchmark = benchmark.set_index("date").reindex(ridge_dates).dropna(subset=["benchmark_fwd_5d"])
    benchmark_nav = pd.DataFrame(
        {
            "date": benchmark.index,
            "nav": (1.0 + benchmark["benchmark_fwd_5d"]).cumprod().values,
            "series": "CSI500 benchmark",
        }
    )
    plot_df = pd.concat([*nav_rows, benchmark_nav], ignore_index=True)

    styles = {
        "Ridge": (COLORS["blue"]["mid"], "-", 2.0),
        "Ridge size-neutral": (COLORS["blue"]["base"], "--", 1.4),
        "IC-weight": (COLORS["gold"]["mid"], "-", 1.4),
        "IC-weight size-neutral": (COLORS["gold"]["base"], "--", 1.4),
        "CSI500 benchmark": (COLORS["neutral"]["dark"], ":", 1.5),
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

    ax.set_ylabel("Cumulative NAV")
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
        fontsize=8.5,
    )
    sns.despine(ax=ax)
    add_chart_header(
        fig,
        ax,
        "Latest tradeability-constrained strategy NAV",
        f"Common comparison window {common_start:%Y-%m-%d} to {common_end:%Y-%m-%d}; "
        "5-trading-day rebalance, Top 50, net of 30 bps turnover cost.",
    )
    save_figure(fig, "csi500_daily_latest_nav")


def plot_constraint_impact() -> None:
    df = pd.read_csv(VALIDATION_DIR / "performance_before_after.csv")
    order = [
        "IC-weight",
        "IC-weight size-neutral",
        "Ridge",
        "Ridge size-neutral",
    ]
    df = df.set_index("Label").loc[order].reset_index()
    y = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 5.8),
        gridspec_kw={"width_ratios": [1.15, 1]},
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
        label="Before exit constraints",
        zorder=3,
    )
    ax1.scatter(
        after,
        y,
        s=62,
        facecolors=COLORS["orange"]["base"],
        edgecolors=COLORS["orange"]["dark"],
        linewidths=1.0,
        label="Latest",
        zorder=4,
    )
    for i, row in df.iterrows():
        ax1.text(
            (before.iloc[i] + after.iloc[i]) / 2,
            i - 0.14,
            f"{row['Ann_Return_Change_pp']:+.2f}pp",
            ha="center",
            va="center",
            fontsize=8,
            color=COLORS["orange"]["dark"],
            family=MONO_FONT_FAMILY[0],
        )
    ax1.set_yticks(y, df["Label"])
    ax1.invert_yaxis()
    ax1.set_xlabel("Annualized return")
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax1.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=2,
        borderaxespad=0,
        fontsize=8.5,
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
        label="Suspension locks",
    )
    ax2.barh(
        y,
        limit_down,
        left=suspension,
        color=COLORS["orange"]["base"],
        edgecolor=COLORS["orange"]["dark"],
        linewidth=1.0,
        label="Limit-down locks",
    )
    for i, row in df.iterrows():
        total = row["Suspension_Locks"] + row["Limit_Down_Locks"]
        ax2.text(
            total + 5,
            i,
            f"{int(total)} events / {int(row['Locked_Rebalances'])} rebalances",
            ha="left",
            va="center",
            fontsize=8,
            color=TOKENS["muted"],
        )
    ax2.set_yticks(y, [])
    ax2.invert_yaxis()
    ax2.set_xlabel("Locked holding events")
    ax2.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=2,
        borderaxespad=0,
        fontsize=8.5,
    )
    ax2.grid(axis="y", visible=False)
    sns.despine(ax=ax1)
    sns.despine(ax=ax2, left=True)
    add_chart_header(
        fig,
        ax1,
        "Exit constraints mainly reduce the size-neutral variants",
        "Annualized return before vs. after suspension and limit-down exit constraints; "
        "lock counts are holding-level events and can exceed the number of affected rebalances.",
    )
    fig.subplots_adjust(wspace=0.22)
    save_figure(fig, "csi500_tradeability_impact")


def main() -> None:
    use_chart_theme()
    frames = load_returns()
    plot_latest_nav(frames)
    plot_constraint_impact()
    print(f"Saved README charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
