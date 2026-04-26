# -*- coding: utf-8 -*-
"""
Global configuration for the A-share multi-factor project.

The pipeline now supports multiple index universes through the `QT_UNIVERSE`
environment variable. Default remains the legacy CSI300 setup so existing data
and outputs keep working without relocation.
"""
import os


# ==================== Project Paths ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BASE_RAW_DIR = os.path.join(DATA_DIR, "raw")
BASE_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


# ==================== Universe Config ====================
UNIVERSE_SPECS = {
    "csi300": {
        "index_code": "000300",
        "index_name": "沪深300",
        "index_symbol": "sh000300",
        "benchmark_etf": "510300",
        "constituents_filename": "hs300_stocks.csv",  # legacy compatibility
        "index_daily_filename": "index_hs300_daily.parquet",  # legacy compatibility
        "total_return_filename": "index_hs300_total_return.parquet",
    },
    "csi500": {
        "index_code": "000905",
        "index_name": "中证500",
        "index_symbol": "sh000905",
        "benchmark_etf": "510500",
        "constituents_filename": "index_constituents.csv",
        "index_daily_filename": "index_daily.parquet",
        "total_return_filename": "index_total_return.parquet",
    },
}

UNIVERSE_ID = os.environ.get("QT_UNIVERSE", "csi300").strip().lower()
if UNIVERSE_ID not in UNIVERSE_SPECS:
    raise ValueError(
        f"Unsupported QT_UNIVERSE={UNIVERSE_ID!r}. "
        f"Expected one of: {sorted(UNIVERSE_SPECS)}"
    )

UNIVERSE_SPEC = UNIVERSE_SPECS[UNIVERSE_ID]
INDEX_CODE = UNIVERSE_SPEC["index_code"]
INDEX_NAME = UNIVERSE_SPEC["index_name"]
INDEX_SYMBOL = UNIVERSE_SPEC["index_symbol"]
BENCHMARK_ETF = UNIVERSE_SPEC["benchmark_etf"]
BENCHMARK_NAME = f"{INDEX_NAME} / {BENCHMARK_ETF} NAV Benchmark"


def _scope_dir(base_dir: str, universe_id: str | None = None) -> str:
    """
    Keep CSI300 on legacy root paths; isolate other universes under subfolders.
    """
    uid = (universe_id or UNIVERSE_ID).lower()
    return base_dir if uid == "csi300" else os.path.join(base_dir, uid)


def get_raw_dir(universe_id: str | None = None) -> str:
    return _scope_dir(BASE_RAW_DIR, universe_id)


def get_processed_dir(universe_id: str | None = None) -> str:
    return _scope_dir(BASE_PROCESSED_DIR, universe_id)


def get_output_dir(universe_id: str | None = None) -> str:
    return _scope_dir(BASE_OUTPUT_DIR, universe_id)


def get_constituents_path(universe_id: str | None = None) -> str:
    uid = (universe_id or UNIVERSE_ID).lower()
    raw_dir = get_raw_dir(uid)
    filename = UNIVERSE_SPECS[uid]["constituents_filename"]
    return os.path.join(raw_dir, filename)


def get_index_daily_path(universe_id: str | None = None) -> str:
    uid = (universe_id or UNIVERSE_ID).lower()
    raw_dir = get_raw_dir(uid)
    filename = UNIVERSE_SPECS[uid]["index_daily_filename"]
    return os.path.join(raw_dir, filename)


def get_total_return_index_path(universe_id: str | None = None) -> str:
    uid = (universe_id or UNIVERSE_ID).lower()
    raw_dir = get_raw_dir(uid)
    filename = UNIVERSE_SPECS[uid]["total_return_filename"]
    return os.path.join(raw_dir, filename)


def get_benchmark_nav_path(universe_id: str | None = None) -> str:
    uid = (universe_id or UNIVERSE_ID).lower()
    raw_dir = get_raw_dir(uid)
    etf = UNIVERSE_SPECS[uid]["benchmark_etf"]
    return os.path.join(raw_dir, f"etf_{etf}_nav_daily.parquet")


def get_benchmark_qfq_path(universe_id: str | None = None) -> str:
    uid = (universe_id or UNIVERSE_ID).lower()
    raw_dir = get_raw_dir(uid)
    etf = UNIVERSE_SPECS[uid]["benchmark_etf"]
    return os.path.join(raw_dir, f"etf_{etf}_qfq_daily.parquet")


RAW_DIR = get_raw_dir()
PROCESSED_DIR = get_processed_dir()
OUTPUT_DIR = get_output_dir()

# Ensure active universe directories exist
for d in [BASE_RAW_DIR, BASE_PROCESSED_DIR, BASE_OUTPUT_DIR, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)


# ==================== Time Range ====================
START_DATE = "20150101"
END_DATE = "20251231"


# ==================== Rebalance Params ====================
REBALANCE_FREQ = "M"
TOP_N_STOCKS = 50
TRANSACTION_COST = 0.003


# ==================== Factor Preprocessing ====================
MAD_MULTIPLIER = 3
MIN_LIST_DAYS = 120
MISSING_THRESHOLD = 0.30


# ==================== Rolling Windows ====================
IC_ROLLING_WINDOW = 12
LGBM_TRAIN_WINDOW = 24
LSTM_TRAIN_WINDOW = 36
LSTM_VAL_WINDOW = 12
LSTM_LOOKBACK = 12


# ==================== Top-Risk Filter ====================
ENABLE_TOP_RISK_FILTER = True
TOP_RISK_FILTER_PCT = 0.20
TOP_RISK_WINDOWS = {
    "bias_window": 20,
    "upshadow_window": 20,
    "vol_spike_window": 6,
    "ret_window": 6,
}


# ==================== Download Config ====================
DOWNLOAD_SLEEP = 2.0
MAX_RETRY = 5
