import os

STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME", "smarteddatalake2026")
FILE_SYSTEM_NAME = os.getenv("FILE_SYSTEM_NAME", "smarted-data")

OPEN_METEO_LAT = float(os.getenv("OPEN_METEO_LAT", "52.6647"))
OPEN_METEO_LON = float(os.getenv("OPEN_METEO_LON", "-8.6231"))