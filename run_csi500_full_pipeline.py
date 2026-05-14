from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "output" / "csi500" / "run_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
stdout_path = LOG_DIR / f"csi500_full_{timestamp}.log"
stderr_path = LOG_DIR / f"csi500_full_{timestamp}.err.log"

steps = [
    ["python", "data/download.py"],
    ["python", "data/clean.py"],
    ["python", "factors/value.py"],
    ["python", "factors/momentum.py"],
    ["python", "factors/quality.py"],
    ["python", "factors/volatility.py"],
    ["python", "factors/liquidity.py"],
    ["python", "factors/additional.py"],
    ["python", "factors/risk.py"],
    ["python", "factors/preprocess.py"],
    ["python", "testing/ic_analysis.py"],
    ["python", "testing/quantile_backtest.py"],
    ["python", "testing/fama_macbeth.py"],
    ["python", "portfolio/backtest.py"],
    ["python", "ml/model_comparison.py"],
    ["python", "analysis/compare_strategies.py"],
    ["python", "analysis/ablation_top_risk.py"],
    ["python", "analysis/ablation_risk_v2.py"],
]


def log_line(handle, message: str) -> None:
    handle.write(message + "\n")
    handle.flush()


def main() -> int:
    env = os.environ.copy()
    env["QT_UNIVERSE"] = "csi500"
    env.setdefault("QT_DOWNLOAD_SLEEP", "0.5")

    with stdout_path.open("w", encoding="utf-8", errors="replace") as out, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as err:
        log_line(out, f"ROOT={ROOT}")
        log_line(out, "QT_UNIVERSE=csi500")
        log_line(out, f"QT_DOWNLOAD_SLEEP={env.get('QT_DOWNLOAD_SLEEP')}")
        log_line(out, f"STDOUT={stdout_path}")
        log_line(out, f"STDERR={stderr_path}")

        for step in steps:
            started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line(out, f"[{started}] START {' '.join(step)}")

            proc = subprocess.Popen(
                step,
                cwd=ROOT,
                env=env,
                stdout=out,
                stderr=err,
                text=True,
            )
            code = proc.wait()

            ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if code != 0:
                log_line(out, f"[{ended}] FAIL  {' '.join(step)} (exit={code})")
                return code

            log_line(out, f"[{ended}] DONE  {' '.join(step)}")
            time.sleep(0.2)

        log_line(out, f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
