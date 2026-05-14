# -*- coding: utf-8 -*-
"""Daily technical-factor experiment.

This is intentionally separate from the monthly factor pipeline:
    - one row per stock per trading day
    - daily technical/price-volume features only
    - forward 5-trading-day return as the default target
    - 5-trading-day rebalance by default
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_OUTPUT_DIR, BASE_PROCESSED_DIR, TRANSACTION_COST, UNIVERSE_ID
from data.benchmark import load_benchmark_daily_returns
from factors.utils import load_daily_prices


FEATURE_COLS = [
    "REV_1D",
    "MOM_5D",
    "MOM_20D",
    "MOM_60D",
    "VOL_20D",
    "TURN_5D",
    "TURN_20D",
    "AMIHUD_20D",
    "VOLUME_RATIO_5_20",
    "BIAS_20",
    "RANGE_20D",
    "SIZE",
    "BETA_60D",
]

BASE_COST_BPS = TRANSACTION_COST * 10_000


def _scoped_dir(base_dir: str, universe: str, experiment: str) -> Path:
    base = Path(base_dir)
    if universe == "csi300":
        return base / experiment
    return base / universe / experiment


def _zscore_cross_section(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("date")
    for col in cols:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform(lambda x: x.std(ddof=0))
        out[col] = (out[col] - mean) / std.replace(0, np.nan)
    return out


def _forward_compound_return(returns: pd.Series, horizon: int) -> pd.Series:
    shifted = (1.0 + returns).shift(-1)
    return shifted.iloc[::-1].rolling(horizon).apply(np.prod, raw=True).iloc[::-1] - 1.0


def build_daily_panel(horizon: int = 5) -> pd.DataFrame:
    print("[daily-tech] loading daily prices...")
    daily = load_daily_prices().copy()
    daily = daily.sort_values(["stock_code", "date"]).reset_index(drop=True)
    daily["ret_1d"] = daily["pct_change"] / 100.0
    daily["dollar_volume"] = daily["turnover"].replace(0, np.nan)

    print("[daily-tech] loading benchmark returns...")
    benchmark = load_benchmark_daily_returns()
    daily = daily.merge(benchmark, on="date", how="left")

    print("[daily-tech] computing daily technical features...")
    parts = []
    for stock_code, group in daily.groupby("stock_code", sort=False):
        g = group.sort_values("date").copy()
        close = g["close"]
        ret = g["ret_1d"]

        g["REV_1D"] = -ret
        g["MOM_5D"] = close / close.shift(5) - 1.0
        g["MOM_20D"] = close / close.shift(20) - 1.0
        g["MOM_60D"] = close / close.shift(60) - 1.0
        g["VOL_20D"] = ret.rolling(20, min_periods=15).std()
        g["TURN_5D"] = g["turnover_rate"].rolling(5, min_periods=3).mean()
        g["TURN_20D"] = g["turnover_rate"].rolling(20, min_periods=15).mean()
        g["AMIHUD_20D"] = (ret.abs() / g["dollar_volume"]).rolling(20, min_periods=15).mean()
        vol_5 = g["volume"].rolling(5, min_periods=3).mean()
        vol_20 = g["volume"].rolling(20, min_periods=15).mean()
        g["VOLUME_RATIO_5_20"] = vol_5 / vol_20.replace(0, np.nan) - 1.0
        ma_20 = close.rolling(20, min_periods=15).mean()
        g["BIAS_20"] = close / ma_20.replace(0, np.nan) - 1.0
        high_20 = g["high"].rolling(20, min_periods=15).max()
        low_20 = g["low"].rolling(20, min_periods=15).min()
        g["RANGE_20D"] = high_20 / low_20.replace(0, np.nan) - 1.0
        g["SIZE"] = -np.log((close * g["outstanding_share"]).replace(0, np.nan))

        cov = ret.rolling(60, min_periods=40).cov(g["index_ret"])
        var = g["index_ret"].rolling(60, min_periods=40).var()
        g["BETA_60D"] = cov / var.replace(0, np.nan)
        # Signal is observed after today's close; trade from next open.
        g[f"ret_fwd_{horizon}d"] = g["open"].shift(-(horizon + 1)) / g["open"].shift(-1) - 1.0
        parts.append(g[["stock_code", "date", f"ret_fwd_{horizon}d", *FEATURE_COLS]])

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.dropna(subset=[f"ret_fwd_{horizon}d"]).copy()

    print("[daily-tech] cross-sectional winsorize/zscore...")
    for col in FEATURE_COLS:
        lower = panel.groupby("date")[col].transform(lambda x: x.quantile(0.01))
        upper = panel.groupby("date")[col].transform(lambda x: x.quantile(0.99))
        panel[col] = panel[col].clip(lower=lower, upper=upper)
    panel = _zscore_cross_section(panel, FEATURE_COLS)
    panel[FEATURE_COLS] = panel[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"[daily-tech] panel rows={len(panel):,}, dates={panel['date'].nunique()}, stocks={panel['stock_code'].nunique()}")
    return panel.sort_values(["date", "stock_code"]).reset_index(drop=True)


def compute_daily_ic(panel: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[daily-tech] computing daily rank IC...")
    records = []
    for date, group in panel.groupby("date", sort=True):
        row = {"date": date}
        for col in FEATURE_COLS:
            valid = group[[col, target_col]].dropna()
            if len(valid) < 30 or valid[col].nunique() <= 1:
                row[col] = np.nan
            else:
                row[col] = valid[col].rank().corr(valid[target_col].rank())
        records.append(row)
    ic = pd.DataFrame(records).sort_values("date")

    summary_rows = []
    for col in FEATURE_COLS:
        s = ic[col].dropna()
        summary_rows.append(
            {
                "Factor": col,
                "IC_mean": s.mean(),
                "IC_std": s.std(),
                "ICIR": s.mean() / s.std() if s.std() > 0 else np.nan,
                "IC>0_pct": (s > 0).mean(),
                "N_days": len(s),
            }
        )
    return ic, pd.DataFrame(summary_rows)


def _make_rebalance_dates(dates: list[pd.Timestamp], warmup_days: int, step: int) -> list[pd.Timestamp]:
    return dates[warmup_days::step]


def _normalize_weights(raw: pd.Series) -> pd.Series:
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = raw.abs().sum()
    if denom < 1e-12:
        return pd.Series(1.0 / len(raw), index=raw.index)
    return raw / denom


def _neutralize_score_by_size(cross: pd.DataFrame, score_col: str = "score") -> pd.Series:
    valid = cross[[score_col, "SIZE"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 30 or valid["SIZE"].std(ddof=0) < 1e-12:
        return cross[score_col]
    x = np.column_stack([np.ones(len(valid)), valid["SIZE"].values])
    y = valid[score_col].values
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    result = cross[score_col].copy()
    result.loc[valid.index] = residual
    return result


def _apply_cost(returns: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    out = returns.copy()
    out["strategy_ret"] = out["gross_ret"] - (cost_bps / 10_000.0) * out["turnover"]
    out["cost_bps"] = cost_bps
    return out


def backtest_ic_weight(
    panel: pd.DataFrame,
    ic: pd.DataFrame,
    target_col: str,
    horizon: int,
    top_n: int,
    ic_window: int,
    cost_bps: float,
    size_neutral_score: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[daily-tech] backtesting IC-weight portfolio...")
    dates = sorted(panel["date"].unique())
    rebalance_dates = _make_rebalance_dates(dates, warmup_days=ic_window + 60, step=horizon)
    ic_indexed = ic.set_index("date")

    prev_holdings: set[str] = set()
    returns = []
    weight_rows = []
    for date in rebalance_dates:
        cross = panel[panel["date"] == date].copy()
        hist = ic_indexed.loc[ic_indexed.index < date, FEATURE_COLS].tail(ic_window)
        raw = hist.mean()
        weights = _normalize_weights(raw)
        cross["score"] = cross[FEATURE_COLS].mul(weights, axis=1).sum(axis=1)
        if size_neutral_score:
            cross["score"] = _neutralize_score_by_size(cross)
        selected = cross.nlargest(top_n, "score")
        holdings = set(selected["stock_code"])
        turnover = 1.0 if not prev_holdings else len(holdings.symmetric_difference(prev_holdings)) / (2 * top_n)
        gross_ret = selected[target_col].mean()
        net_ret = gross_ret - (cost_bps / 10_000.0) * turnover
        returns.append(
            {
                "date": date,
                "strategy_ret": net_ret,
                "gross_ret": gross_ret,
                "turnover": turnover,
                "n_holdings": len(selected),
            }
        )
        weight_rows.append({"date": date, **weights.to_dict()})
        prev_holdings = holdings

    return pd.DataFrame(returns), pd.DataFrame(weight_rows)


def backtest_ridge(
    panel: pd.DataFrame,
    target_col: str,
    horizon: int,
    top_n: int,
    train_days: int,
    start_days: int,
    cost_bps: float,
    size_neutral_score: bool = False,
) -> pd.DataFrame:
    print("[daily-tech] backtesting Ridge walk-forward...")
    dates = sorted(panel["date"].unique())
    rebalance_dates = _make_rebalance_dates(dates, warmup_days=start_days, step=horizon)

    prev_holdings: set[str] = set()
    returns = []
    for i, date in enumerate(rebalance_dates, start=1):
        date_pos = dates.index(date)
        train_dates = dates[max(0, date_pos - train_days):date_pos]
        train = panel[panel["date"].isin(train_dates)].dropna(subset=[target_col])
        pred = panel[panel["date"] == date].copy()
        if len(train) < 1000 or pred.empty:
            continue

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[FEATURE_COLS])
        y_train = train[target_col].values
        x_pred = scaler.transform(pred[FEATURE_COLS])

        model = Ridge(alpha=10.0)
        model.fit(x_train, y_train)
        pred["score"] = model.predict(x_pred)
        if size_neutral_score:
            pred["score"] = _neutralize_score_by_size(pred)
        selected = pred.nlargest(top_n, "score")
        holdings = set(selected["stock_code"])
        turnover = 1.0 if not prev_holdings else len(holdings.symmetric_difference(prev_holdings)) / (2 * top_n)
        gross_ret = selected[target_col].mean()
        net_ret = gross_ret - (cost_bps / 10_000.0) * turnover
        returns.append(
            {
                "date": date,
                "strategy_ret": net_ret,
                "gross_ret": gross_ret,
                "turnover": turnover,
                "n_holdings": len(selected),
            }
        )
        prev_holdings = holdings
        if i % 50 == 0:
            print(f"  ridge progress: {i}/{len(rebalance_dates)} rebalances")

    return pd.DataFrame(returns)


def backtest_lightgbm(
    panel: pd.DataFrame,
    target_col: str,
    horizon: int,
    top_n: int,
    train_days: int,
    start_days: int,
    cost_bps: float,
    max_train_rows: int = 60_000,
    size_neutral_score: bool = False,
) -> pd.DataFrame:
    print("[daily-tech] backtesting LightGBM walk-forward...")
    dates = sorted(panel["date"].unique())
    rebalance_dates = _make_rebalance_dates(dates, warmup_days=start_days, step=horizon)

    prev_holdings: set[str] = set()
    returns = []
    rng = np.random.default_rng(42)
    for i, date in enumerate(rebalance_dates, start=1):
        date_pos = dates.index(date)
        train_dates = dates[max(0, date_pos - train_days):date_pos]
        train = panel[panel["date"].isin(train_dates)].dropna(subset=[target_col])
        pred = panel[panel["date"] == date].copy()
        if len(train) < 1000 or pred.empty:
            continue
        if len(train) > max_train_rows:
            idx = rng.choice(train.index.to_numpy(), size=max_train_rows, replace=False)
            train = train.loc[idx]

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(train[FEATURE_COLS], train[target_col])
        pred["score"] = model.predict(pred[FEATURE_COLS])
        if size_neutral_score:
            pred["score"] = _neutralize_score_by_size(pred)
        selected = pred.nlargest(top_n, "score")
        holdings = set(selected["stock_code"])
        turnover = 1.0 if not prev_holdings else len(holdings.symmetric_difference(prev_holdings)) / (2 * top_n)
        gross_ret = selected[target_col].mean()
        net_ret = gross_ret - (cost_bps / 10_000.0) * turnover
        returns.append(
            {
                "date": date,
                "strategy_ret": net_ret,
                "gross_ret": gross_ret,
                "turnover": turnover,
                "n_holdings": len(selected),
            }
        )
        prev_holdings = holdings
        if i % 50 == 0:
            print(f"  lightgbm progress: {i}/{len(rebalance_dates)} rebalances")

    return pd.DataFrame(returns)


def summarize_period_returns(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    horizon: int,
    label: str,
) -> pd.Series:
    if strategy.empty:
        return pd.Series({"Label": label})
    returns = strategy.set_index("date")["strategy_ret"].dropna()
    bench = benchmark.set_index("date")[f"benchmark_fwd_{horizon}d"].reindex(returns.index).dropna()
    returns = returns.reindex(bench.index)
    periods_per_year = 252 / horizon

    ann_ret = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    nav = (1 + returns).cumprod()
    max_dd = (nav / nav.cummax() - 1).min()
    bench_ann = (1 + bench).prod() ** (periods_per_year / len(bench)) - 1
    excess = returns - bench
    excess_ann = (1 + excess).prod() ** (periods_per_year / len(excess)) - 1
    tracking_error = excess.std() * np.sqrt(periods_per_year)

    return pd.Series(
        {
            "Label": label,
            "Ann_Return": ann_ret,
            "Ann_Vol": ann_vol,
            "Sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "Max_Drawdown": max_dd,
            "WinRate": (returns > 0).mean(),
            "Periods": len(returns),
            "Avg_Turnover": strategy["turnover"].mean(),
            "Benchmark_Ann_Return": bench_ann,
            "Excess_Ann_Return": excess_ann,
            "Tracking_Error": tracking_error,
            "Info_Ratio": excess_ann / tracking_error if tracking_error > 0 else np.nan,
        }
    )


def build_cost_sensitivity(
    return_sets: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    horizon: int,
    cost_bps_list: list[float],
) -> pd.DataFrame:
    rows = []
    for label, returns in return_sets.items():
        for cost_bps in cost_bps_list:
            adjusted = _apply_cost(returns, cost_bps)
            summary = summarize_period_returns(adjusted, benchmark, horizon, label)
            summary["Cost_Bps"] = cost_bps
            rows.append(summary)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--ic-window", type=int, default=252)
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--start-days", type=int, default=756)
    parser.add_argument("--cost-bps", type=float, default=BASE_COST_BPS)
    parser.add_argument("--cost-scenarios-bps", default="30,60,100")
    parser.add_argument("--skip-ridge", action="store_true")
    parser.add_argument("--run-lightgbm", action="store_true")
    parser.add_argument("--lgbm-max-train-rows", type=int, default=60_000)
    args = parser.parse_args()

    experiment = "daily_tech"
    processed_dir = _scoped_dir(BASE_PROCESSED_DIR, UNIVERSE_ID, experiment)
    output_dir = _scoped_dir(BASE_OUTPUT_DIR, UNIVERSE_ID, experiment)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_col = f"ret_fwd_{args.horizon}d"
    panel = build_daily_panel(horizon=args.horizon)
    panel.to_parquet(processed_dir / "daily_tech_panel.parquet", index=False)

    ic, ic_summary = compute_daily_ic(panel, target_col)
    ic.to_csv(output_dir / "daily_ic_timeseries.csv", index=False, encoding="utf-8-sig")
    ic_summary.to_csv(output_dir / "daily_ic_summary.csv", index=False, encoding="utf-8-sig")

    benchmark = load_benchmark_daily_returns().copy()
    benchmark[f"benchmark_fwd_{args.horizon}d"] = _forward_compound_return(
        benchmark["index_ret"], args.horizon
    )

    summaries = []
    return_sets: dict[str, pd.DataFrame] = {}

    ic_returns, ic_weights = backtest_ic_weight(
        panel, ic, target_col, args.horizon, args.top_n, args.ic_window, args.cost_bps
    )
    ic_returns.to_csv(output_dir / "ic_weight_returns.csv", index=False, encoding="utf-8-sig")
    ic_weights.to_csv(output_dir / "ic_weight_factor_weights.csv", index=False, encoding="utf-8-sig")
    summaries.append(summarize_period_returns(ic_returns, benchmark, args.horizon, "IC-weight"))
    return_sets["IC-weight"] = ic_returns

    ic_neutral_returns, ic_neutral_weights = backtest_ic_weight(
        panel,
        ic,
        target_col,
        args.horizon,
        args.top_n,
        args.ic_window,
        args.cost_bps,
        size_neutral_score=True,
    )
    ic_neutral_returns.to_csv(output_dir / "ic_weight_size_neutral_returns.csv", index=False, encoding="utf-8-sig")
    ic_neutral_weights.to_csv(
        output_dir / "ic_weight_size_neutral_factor_weights.csv", index=False, encoding="utf-8-sig"
    )
    summaries.append(summarize_period_returns(ic_neutral_returns, benchmark, args.horizon, "IC-weight size-neutral"))
    return_sets["IC-weight size-neutral"] = ic_neutral_returns

    if not args.skip_ridge:
        ridge_returns = backtest_ridge(
            panel, target_col, args.horizon, args.top_n, args.train_days, args.start_days, args.cost_bps
        )
        ridge_returns.to_csv(output_dir / "ridge_returns.csv", index=False, encoding="utf-8-sig")
        summaries.append(summarize_period_returns(ridge_returns, benchmark, args.horizon, "Ridge"))
        return_sets["Ridge"] = ridge_returns

        ridge_neutral_returns = backtest_ridge(
            panel,
            target_col,
            args.horizon,
            args.top_n,
            args.train_days,
            args.start_days,
            args.cost_bps,
            size_neutral_score=True,
        )
        ridge_neutral_returns.to_csv(output_dir / "ridge_size_neutral_returns.csv", index=False, encoding="utf-8-sig")
        summaries.append(summarize_period_returns(ridge_neutral_returns, benchmark, args.horizon, "Ridge size-neutral"))
        return_sets["Ridge size-neutral"] = ridge_neutral_returns

    if args.run_lightgbm:
        lgbm_returns = backtest_lightgbm(
            panel,
            target_col,
            args.horizon,
            args.top_n,
            args.train_days,
            args.start_days,
            args.cost_bps,
            max_train_rows=args.lgbm_max_train_rows,
        )
        lgbm_returns.to_csv(output_dir / "lightgbm_returns.csv", index=False, encoding="utf-8-sig")
        summaries.append(summarize_period_returns(lgbm_returns, benchmark, args.horizon, "LightGBM"))
        return_sets["LightGBM"] = lgbm_returns

    cost_bps_list = [float(x.strip()) for x in args.cost_scenarios_bps.split(",") if x.strip()]
    cost_sensitivity = build_cost_sensitivity(return_sets, benchmark, args.horizon, cost_bps_list)
    cost_sensitivity.to_csv(output_dir / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "performance_summary.csv", index=False, encoding="utf-8-sig")
    print("\n[daily-tech] performance summary:")
    print(summary.to_string(index=False))
    print(f"\n[daily-tech] outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
