"""
General AI Trading System - Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# TRADING CONFIGURATION
# ============================================================

# Symbols để theo dõi
SYMBOLS = ["BTC/USDT", "ETH/USDT"]

# Timeframe mặc định
DEFAULT_TIMEFRAME = "1h"

# Vốn ban đầu (Paper Trading)
INITIAL_CAPITAL = 10000.0

# ============================================================
# EXCHANGE CONFIGURATION (CCXT)
# ============================================================

EXCHANGE_ID = "binance"
EXCHANGE_CONFIG = {
    "apiKey": os.getenv("EXCHANGE_API_KEY", ""),
    "secret": os.getenv("EXCHANGE_SECRET", ""),
    "sandbox": True,  # Paper trading mode
    "enableRateLimit": True,
    "options": {
        "defaultType": "spot",
    }
}

# ============================================================
# TECHNICAL INDICATORS
# ============================================================

INDICATOR_CONFIG = {
    # Trend
    "sma_periods": [10, 20, 50],
    "ema_periods": [9, 21, 55],
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "adx_period": 14,

    # Momentum
    "rsi_period": 14,
    "stoch": {"k": 14, "d": 3},
    "cci_period": 20,
    "willr_period": 14,

    # Volatility
    "bb": {"period": 20, "std": 2},
    "atr_period": 14,

    # Volume
    "obv": True,
    "mfi_period": 14,
}

# ============================================================
# AI/ML MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 5,            # giảm để tránh overfitting
        "learning_rate": 0.03,     # nhỏ hơn, kết hợp early stopping
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 3,     # regularization
        "gamma": 0.1,              # regularization
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "early_stopping_rounds": 30,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 7,
        "min_samples_split": 10,   # tăng để tránh overfitting
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",  # xử lý imbalanced labels
        "random_state": 42,
        "n_jobs": -1,
    },
    "lstm": {
        "hidden_size": 128,        # tăng capacity
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "epochs": 80,
        "batch_size": 64,
        "sequence_length": 48,    # 48 giờ ngữ cảnh
        "patience": 15,            # early stopping
    },
    "ensemble_weights": {
        "xgboost": 0.45,
        "random_forest": 0.35,
        "lstm": 0.20,             # giảm trọng số LSTM vì ít data hơn
    },
    # Tỷ lệ train/test split
    "test_size": 0.2,
    # Số nến tối thiểu để train
    "min_data_points": 500,
    # Ngưỡng xác suất để tạo signal (giảm từ 0.6 xuống 0.52)
    "confidence_threshold": 0.52,
    # Số nến tương lai để tính target (12 giờ trên khung 1H)
    "target_periods": 12,
    # Ngưỡng return tối thiểu để label (lọc nhiễu sideway)
    "min_return_threshold": 0.005,  # 0.5%
}

# ============================================================
# RISK MANAGEMENT
# ============================================================

RISK_CONFIG = {
    # Tỷ lệ rủi ro tối đa mỗi lệnh (% vốn)
    "max_risk_per_trade": 0.02,  # 2%
    # Stop loss mặc định (%)
    "default_stop_loss": 0.03,  # 3%
    # Take profit mặc định (%)
    "default_take_profit": 0.06,  # 6%
    # Drawdown tối đa cho phép (%)
    "max_drawdown": 0.15,  # 15%
    # Số lệnh mở tối đa
    "max_open_positions": 3,
    # Giới hạn loss trong ngày (%)
    "daily_loss_limit": 0.05,  # 5%
}

# ============================================================
# BACKTESTING
# ============================================================

BACKTEST_CONFIG = {
    # Phí giao dịch (%)
    "commission": 0.001,  # 0.1%
    # Slippage (%)
    "slippage": 0.0005,  # 0.05%
    # Số ngày backtest mặc định
    "default_days": 90,
}

# ============================================================
# SERVER CONFIGURATION
# ============================================================

SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
}

# ============================================================
# DATABRICKS CONFIGURATION (Free Edition)
# ============================================================

DATABRICKS_CONFIG = {
    "host": "dbc-a7cfc4c3-861a.cloud.databricks.com",
    "token": os.getenv("DATABRICKS_TOKEN", ""),
    "http_path": "/sql/1.0/warehouses/9774393a6fd42c4a",
    "catalog": "workspace",  # Thay đổi ở đây
    "schema": "default",    # Dùng default hoặc tạo schema mới trong workspace
    "enabled": True,
}

MLFLOW_CONFIG = {
    "tracking_uri": "databricks",
    "experiment_name": "/Shared/AI_Trading_Experiments",
    "enabled": DATABRICKS_CONFIG["enabled"],
}

# ============================================================
# DATA PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Tạo thư mục nếu chưa có
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
