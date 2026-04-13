import pandas as pd
import holidays

def build_daily_master_row(daily_df: pd.DataFrame, weekly_df: pd.DataFrame, weather_df: pd.DataFrame):
    """
    输出只有一行：当天真实主表记录
    """
    if daily_df.empty:
        raise ValueError("daily_df is empty")

    row = daily_df.copy()

    # weekly
    if weekly_df is not None and not weekly_df.empty and "attendance_weekly" in weekly_df.columns:
        row["attendance_weekly"] = weekly_df["attendance_weekly"].iloc[0]
    else:
        row["attendance_weekly"] = None

    # weather：只取当天
    row_date = pd.to_datetime(row["date"].iloc[0]).normalize()

    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.normalize()
    weather_today = weather_df[weather_df["date"] == row_date]

    if not weather_today.empty:
        row["temperature_min"] = weather_today["temperature_min"].iloc[0]
        row["precipitation_sum"] = weather_today["precipitation_sum"].iloc[0]
        row["windspeed_max"] = weather_today["windspeed_max"].iloc[0]
    else:
        row["temperature_min"] = None
        row["precipitation_sum"] = None
        row["windspeed_max"] = None

    # calendar
    ie_holidays = holidays.Ireland()
    row["day_of_week"] = row_date.dayofweek
    row["is_weekend"] = int(row_date.dayofweek in [5, 6])
    row["is_holiday"] = int(row_date.date() in ie_holidays)
    row["month"] = row_date.month

    return row.reset_index(drop=True)