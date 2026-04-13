from io import BytesIO
import os
import pandas as pd

from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

from config import STORAGE_ACCOUNT_NAME, FILE_SYSTEM_NAME, LATEST_FEATURES_PATH

# 👉 新增：读取环境变量
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY", "")


def read_latest_features_from_adls() -> pd.DataFrame:
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net"

    # ✅ 优先用 key（Azure 上）
    if AZURE_STORAGE_KEY:
        service_client = DataLakeServiceClient(
            account_url=account_url,
            credential=AZURE_STORAGE_KEY,
        )
    else:
        # ✅ 本地 fallback
        credential = DefaultAzureCredential()
        service_client = DataLakeServiceClient(
            account_url=account_url,
            credential=credential,
        )

    fs_client = service_client.get_file_system_client(FILE_SYSTEM_NAME)
    file_client = fs_client.get_file_client(LATEST_FEATURES_PATH)

    downloaded = file_client.download_file()
    raw = downloaded.readall()

    if LATEST_FEATURES_PATH.endswith(".csv"):
        df = pd.read_csv(BytesIO(raw))
    elif LATEST_FEATURES_PATH.endswith(".parquet"):
        df = pd.read_parquet(BytesIO(raw))
    else:
        raise ValueError(f"Unsupported file format: {LATEST_FEATURES_PATH}")

    if "date" not in df.columns:
        raise ValueError("latest features file must contain 'date' column")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    return df