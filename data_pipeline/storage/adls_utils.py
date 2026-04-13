from io import BytesIO
from azure.storage.filedatalake import DataLakeServiceClient

from data_pipeline.config import STORAGE_ACCOUNT_NAME, FILE_SYSTEM_NAME, AZURE_STORAGE_KEY

def get_filesystem_client():
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net"
    service_client = DataLakeServiceClient(
        account_url=account_url,
        credential=AZURE_STORAGE_KEY
    )
    return service_client.get_file_system_client(FILE_SYSTEM_NAME)


def upload_bytes(path: str, data: bytes, overwrite: bool = True):
    fs_client = get_filesystem_client()
    file_client = fs_client.get_file_client(path)
    file_client.upload_data(data, overwrite=overwrite)


def upload_text(path: str, text: str, overwrite: bool = True):
    upload_bytes(path, text.encode("utf-8"), overwrite=overwrite)


def upload_dataframe_parquet(path: str, df):
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    upload_bytes(path, buffer.read(), overwrite=True)


def upload_dataframe_csv(path: str, df):
    upload_bytes(path, df.to_csv(index=False).encode("utf-8"), overwrite=True)