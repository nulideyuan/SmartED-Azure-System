import json
from datetime import datetime
import pandas as pd

print("[boot] main.py loaded", flush=True)

def run_pipeline():
    print("[boot] importing config...", flush=True)
    from data_pipeline.config import OPEN_METEO_LAT, OPEN_METEO_LON

    print("[boot] importing adls utils...", flush=True)
    from data_pipeline.storage.adls_utils import (
        upload_text,
        upload_bytes,
        upload_dataframe_parquet,
        upload_dataframe_csv,
    )

    print("[boot] importing weather...", flush=True)
    from data_pipeline.sources.weather import fetch_weather

    print("[boot] importing daily report...", flush=True)
    from data_pipeline.sources.daily_report import get_latest_daily_uhl_df

    print("[boot] importing weekly report...", flush=True)
    from data_pipeline.sources.weekly_report import get_latest_weekly_attendance_df

    print("[boot] importing latest_features...", flush=True)
    from data_pipeline.transform.latest_features import build_latest_features

    print("[boot] importing daily_row...", flush=True)
    from data_pipeline.transform.daily_row import build_daily_master_row

    print("[boot] importing history_store...", flush=True)
    from data_pipeline.storage.history_store import append_daily_row

    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    yyyy, mm, dd = run_date.split("-")

    print("[1] fetching weather...", flush=True)
    weather_raw, weather_df = fetch_weather(OPEN_METEO_LAT, OPEN_METEO_LON)

    print("[2] uploading weather...", flush=True)
    upload_text(
        f"raw/weather/{yyyy}/{mm}/{dd}/weather_raw.json",
        json.dumps(weather_raw, ensure_ascii=False, indent=2)
    )
    upload_dataframe_parquet(
        f"processed/weather/date={run_date}/weather.parquet",
        weather_df
    )

    print("[3] fetching daily report...", flush=True)
    daily_html, daily_df = get_latest_daily_uhl_df(pd.Timestamp(run_date))

    print("[4] uploading daily report...", flush=True)
    upload_text(
        f"raw/daily_report/{yyyy}/{mm}/{dd}/daily_report_raw.html",
        daily_html
    )
    upload_dataframe_parquet(
        f"processed/daily_report/date={run_date}/daily_report.parquet",
        daily_df
    )

    print("[5] fetching weekly report...", flush=True)
    pdf_url, weekly_pdf_bytes, weekly_df = get_latest_weekly_attendance_df()

    print("[6] uploading weekly report...", flush=True)
    upload_bytes(
        f"raw/weekly_report/{yyyy}/{mm}/{dd}/weekly_report_raw.pdf",
        weekly_pdf_bytes
    )
    upload_text(
        f"raw/weekly_report/{yyyy}/{mm}/{dd}/weekly_report_url.txt",
        pdf_url
    )
    upload_dataframe_parquet(
        f"processed/weekly_report/date={run_date}/weekly_report.parquet",
        weekly_df
    )

    print("[7] building daily master row...", flush=True)
    daily_master_row = build_daily_master_row(daily_df, weekly_df, weather_df)

    print("[8] appending history...", flush=True)
    history_df = append_daily_row(daily_master_row)

    print("[9] building latest features...", flush=True)
    latest_df = build_latest_features(history_df)

    print("[10] uploading latest features...", flush=True)
    upload_dataframe_parquet(
        "serving/features/latest_features.parquet",
        latest_df
    )
    upload_dataframe_csv(
        "serving/features/latest_features.csv",
        latest_df
    )

    print("[done] Pipeline run complete.", flush=True)

if __name__ == "__main__":
    run_pipeline()