import numpy as np
import pandas as pd


FEATURE_COLS = [
    "uhl_dtoc",
    "uhl_wait_24h",
    "uhl_wait_75plus",
    "uhl_surge",
    "attendance_weekly",
    "attendance_lag_7",
    "attendance_lag_14",
    "attend_spike",
    "attend_trend",
    "trolley_lag_1",
    "trolley_lag_2",
    "trolley_lag_3",
    "trolley_lag_7",
    "trolley_roll7_mean",
    "trolley_roll7_std",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "month",
    "week_sin",
    "week_cos",
    "temperature_min",
    "precipitation_sum",
    "windspeed_max",
]


def build_latest_features(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    从历史主表生成模型输入特征表。
    输入: 每天一行的历史真实数据
    输出: 包含 feature_cols + date + 参考列 的 latest_features
    """

    if history_df is None or history_df.empty:
        raise ValueError("history_df is empty")

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    # -------------------------
    # 保底检查基础列
    # -------------------------
    required_base_cols = [
        "date",
        "uhl_ed",
        "uhl_surge",
        "uhl_dtoc",
        "uhl_wait_24h",
        "uhl_wait_75plus",
        "attendance_weekly",
        "temperature_min",
        "precipitation_sum",
        "windspeed_max",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "month",
    ]
    missing = [c for c in required_base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"history_df missing required columns: {missing}")

    # -------------------------
    # Week cyclic features
    # -------------------------
    iso = df["date"].dt.isocalendar()
    week_num = iso.week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * week_num / 52)
    df["week_cos"] = np.cos(2 * np.pi * week_num / 52)

    # -------------------------
    # Attendance lag/trend
    # -------------------------
    df["attendance_lag_7"] = df["attendance_weekly"].shift(7)
    df["attendance_lag_14"] = df["attendance_weekly"].shift(14)

    df["attend_spike"] = df["attendance_weekly"] - df["attendance_lag_7"]
    df["attend_trend"] = df["attendance_lag_7"] - df["attendance_lag_14"]

    # -------------------------
    # Trolley lag / rolling
    # 这里默认用 uhl_ed 作为 trolley proxy
    # -------------------------
    df["trolley_lag_1"] = df["uhl_ed"].shift(1)
    df["trolley_lag_2"] = df["uhl_ed"].shift(2)
    df["trolley_lag_3"] = df["uhl_ed"].shift(3)
    df["trolley_lag_7"] = df["uhl_ed"].shift(7)

    df["trolley_roll7_mean"] = df["uhl_ed"].shift(1).rolling(7).mean()
    df["trolley_roll7_std"] = df["uhl_ed"].shift(1).rolling(7).std()

    # -------------------------
    # 只保留模型真正需要的列 + date + 一些参考列
    # -------------------------
    out_cols = ["date"] + FEATURE_COLS + [
        "uhl_ed",
        "uhl_ward",
        "uhl_total",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    out = df[out_cols].copy()

    # -------------------------
    # 对早期不足 lag 的记录，先做填充
    # 你现在是为了线上服务稳定，先用 ffill/bfill
    # -------------------------
    feature_only_cols = [c for c in FEATURE_COLS if c in out.columns]
    out[feature_only_cols] = out[feature_only_cols].ffill().bfill()

    return out.reset_index(drop=True)