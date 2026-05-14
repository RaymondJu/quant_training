# -*- coding: utf-8 -*-
"""
Run the CSI500 technical-only factor pipeline.

This reuses raw CSI500 price/volume/industry/benchmark data and writes all
processed data plus reports to:
    data/processed/csi500/technical_only
    output/csi500/technical_only
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "output" / "csi500" / "technical_only" / "run_logs"

STEPS = [
    "data/clean.py",
    "factors/momentum.py",
    "factors/volatility.py",
    "factors/liquidity.py",
    "factors/additional.py",
    "factors/risk.py",
    "factors/preprocess.py",
    "testing/ic_analysis.py",
    "testing/quantile_backtest.py",
    "testing/fama_macbeth.py",
    "portfolio/backtest.py",
    "ml/model_comparison.py",
    "analysis/compare_strategies.py",
    "analysis/ablation_top_risk.py",
    "analysis/ablation_risk_v2.py",
]


def run_step(script: str, env: dict[str, str]) -> None:
    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{started}_{Path(script).stem}.log"
    print(f"\n=== RUN {script} ===")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, script],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        print(f"FAILED {script}; see {log_path}")
        raise SystemExit(proc.returncode)
    print(f"OK {script}; log={log_path}")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["QT_UNIVERSE"] = "csi500"
    env["QT_FACTOR_SET"] = "technical"

    print("CSI500 technical-only pipeline")
    print(f"root={ROOT}")
    print(f"log_dir={LOG_DIR}")
    for script in STEPS:
        run_step(script, env)
    print("\nALL DONE")


if __name__ == "__main__":
    main()
