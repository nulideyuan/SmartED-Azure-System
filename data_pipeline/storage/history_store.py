from io import BytesIO
import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

from data_pipeline.config import STORAGE_ACCOUNT_NAME, FILE_SYSTEM_NAME

HISTORY_PARQUET_PATH = "serving/history/uhl_daily_master.parquet"
HISTORY_CSV_PATH = "serving/history/uhl_daily_master.csv"


def get_filesystem_client():
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net"
    credential = DefaultAzureCredential()
    service_client = DataLakeServiceClient(account_url=account_url, credential=credential)
    return service_client.get_file_system_client(FILE_SYSTEM_NAME)


def _read_parquet_if_exists(path: str) -> pd.DataFrame:
    fs_client = get_filesystem_client()
    file_client = fs_client.get_file_client(path)

    try:
        raw = file_client.download_file().readall()
        return pd.read_parquet(BytesIO(raw))
    except Exception:
        return pd.DataFrame()


def _upload_parquet(path: str, df: pd.DataFrame):
    fs_client = get_filesystem_client()
    file_client = fs_client.get_file_client(path)

    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    file_client.upload_data(buffer.read(), overwrite=True)


def _upload_csv(path: str, df: pd.DataFrame):
    fs_client = get_filesystem_client()
    file_client = fs_client.get_file_client(path)
    file_client.upload_data(df.to_csv(index=False).encode("utf-8"), overwrite=True)


def append_daily_row(new_row_df: pd.DataFrame):
    """
    new_row_df 预期只有1行：当天真实数据
    """
    if new_row_df.empty:
        raise ValueError("new_row_df is empty")

    df_old = _read_parquet_if_exists(HISTORY_PARQUET_PATH)

    df_all = pd.concat([df_old, new_row_df], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.normalize()

    # 按 date 去重，保留最新
    df_all = (
        df_all.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    _upload_parquet(HISTORY_PARQUET_PATH, df_all)
    _upload_csv(HISTORY_CSV_PATH, df_all)

    return df_all