import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_artifacts" / "lstm_v1"

MODEL_PATH = str(MODEL_DIR / "multi_target_lstm.pth")
X_MEAN_PATH = str(MODEL_DIR / "x_mean.npy")
X_STD_PATH = str(MODEL_DIR / "x_std.npy")
Y_MEAN_PATH = str(MODEL_DIR / "y_mean.npy")
Y_STD_PATH = str(MODEL_DIR / "y_std.npy")
FEATURE_COLS_PATH = str(MODEL_DIR / "feature_cols.json")

TARGETS = ["uhl_ed", "uhl_wait_24h", "uhl_wait_75plus"]

SEQ_LEN = 14
FORECAST_DAYS = 7

# ADLS
STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME", "smarteddatalake2026")
FILE_SYSTEM_NAME = os.getenv("FILE_SYSTEM_NAME", "smarted-data")
LATEST_FEATURES_PATH = os.getenv("LATEST_FEATURES_PATH", "serving/features/latest_features.csv")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY", "")
# =========================
# Azure OpenAI / Foundry
# =========================
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

# =========================
# API URLs
# 如果 model_api 和 api_main 本地分别跑在不同端口
# =========================
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://127.0.0.1:8001/predict")
RISK_CONTROL_API_URL = os.getenv("RISK_CONTROL_API_URL", "http://127.0.0.1:8000/forecast-risk-control")