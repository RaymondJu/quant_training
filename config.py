# -*- coding: utf-8 -*-
"""
全局配置文件
A股多因子选股项目
"""
import os

# ==================== 路径配置 ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# 确保目录存在
for d in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 时间范围 ====================
START_DATE = "20150101"
END_DATE = "20251231"

# ==================== 选股池 ====================
INDEX_CODE = "000300"   # 沪深300
INDEX_NAME = "沪深300"

# ==================== 调仓参数 ====================
REBALANCE_FREQ = "M"        # 月度调仓
TOP_N_STOCKS = 50           # 选前50只
TRANSACTION_COST = 0.003    # 单边千三

# ==================== 因子预处理 ====================
MAD_MULTIPLIER = 3          # 去极值: median ± 3*MAD
MIN_LIST_DAYS = 120         # 次新股剔除: 上市不满120交易日
MISSING_THRESHOLD = 0.30    # 缺失值>30%的特征丢弃

# ==================== 滚动窗口 ====================
IC_ROLLING_WINDOW = 12      # IC加权: 过去12个月
LGBM_TRAIN_WINDOW = 24      # LightGBM: 过去24个月
LSTM_TRAIN_WINDOW = 36      # LSTM: 过去36个月
LSTM_VAL_WINDOW = 12        # LSTM验证: 12个月
LSTM_LOOKBACK = 12          # LSTM回望窗口: 12个月

# ==================== 顶部风险过滤层 ====================
ENABLE_TOP_RISK_FILTER = True   # False时完全跳过，复现原始结果
TOP_RISK_FILTER_PCT    = 0.20   # 剔除风控分最高的20%
TOP_RISK_WINDOWS = {
    "bias_window":      20,     # BIAS_20 均线窗口（交易日）
    "upshadow_window":  20,     # UPSHADOW_20 计算窗口（交易日）
    "vol_spike_window":  6,     # VOL_SPIKE_6M 基期月数
    "ret_window":        6,     # RET_6M 累计月数
}

# ==================== AKShare 下载配置 ====================
DOWNLOAD_SLEEP = 2.0        # 每次请求间隔(秒), 避免频率限制
MAX_RETRY = 5               # 最大重试次数
