import json
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    MODEL_PATH, X_MEAN_PATH, X_STD_PATH, Y_MEAN_PATH, Y_STD_PATH,
    FEATURE_COLS_PATH, TARGETS, SEQ_LEN, FORECAST_DAYS
)


class MultiTargetLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def load_model(device="cpu"):
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)

    x_mean = np.load(X_MEAN_PATH)
    x_std = np.load(X_STD_PATH)
    y_mean = np.load(Y_MEAN_PATH)
    y_std = np.load(Y_STD_PATH)

    model = MultiTargetLSTM(
        input_size=len(feature_cols),
        output_size=len(TARGETS)
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model, feature_cols, x_mean, x_std, y_mean, y_std


def validate_input_df(df: pd.DataFrame, feature_cols: list[str]):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if len(df) < SEQ_LEN:
        raise ValueError(f"Need at least {SEQ_LEN} rows, got {len(df)}")


def fill_missing_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    defaults = {
        "attendance_lag_7": df["attendance_weekly"] if "attendance_weekly" in df.columns else 0,
        "attendance_lag_14": df["attendance_weekly"] if "attendance_weekly" in df.columns else 0,
        "attend_spike": 0,
        "attend_trend": 0,
        "trolley_lag_1": df["uhl_ed"] if "uhl_ed" in df.columns else 0,
        "trolley_lag_2": df["uhl_ed"] if "uhl_ed" in df.columns else 0,
        "trolley_lag_3": df["uhl_ed"] if "uhl_ed" in df.columns else 0,
        "trolley_lag_7": df["uhl_ed"] if "uhl_ed" in df.columns else 0,
        "trolley_roll7_mean": df["uhl_ed"] if "uhl_ed" in df.columns else 0,
        "trolley_roll7_std": 0,
    }

    for col, val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    return df


def add_future_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_rows = df[df["date"].isna()]
    if not bad_rows.empty:
        print("\n[debug] rows with bad date:")
        print(bad_rows)

    df = df.dropna(subset=["date"]).reset_index(drop=True)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month

    iso = df["date"].dt.isocalendar()
    week_num = iso.week.astype("Int64").fillna(1).astype(int)

    df["week_sin"] = np.sin(2 * np.pi * week_num / 52)
    df["week_cos"] = np.cos(2 * np.pi * week_num / 52)

    if "is_holiday" in df.columns:
        df["is_holiday"] = df["is_holiday"].fillna(0)
    else:
        df["is_holiday"] = 0

    return df


def extend_future_rows(history_df: pd.DataFrame, forecast_days: int) -> pd.DataFrame:
    df = history_df.copy()
    last_date = df["date"].max()

    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )
    future_df = pd.DataFrame({"date": future_dates})

    for col in df.columns:
        if col != "date" and col not in future_df.columns:
            future_df[col] = np.nan

    out = pd.concat([df, future_df], ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    out = add_future_calendar_features(out)

    carry_forward_cols = [
        "uhl_dtoc",
        "uhl_surge",
        "attendance_weekly",
        "attendance_lag_7",
        "attendance_lag_14",
        "attend_spike",
        "attend_trend",
        "temperature_min",
        "precipitation_sum",
        "windspeed_max",
    ]
    for col in carry_forward_cols:
        if col in out.columns:
            out[col] = out[col].ffill()

    return out


def recompute_recursive_features(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    if "trolley_lag_1" in df.columns and idx - 1 >= 0:
        df.loc[idx, "trolley_lag_1"] = df.loc[idx - 1, "uhl_ed"]

    if "trolley_lag_2" in df.columns and idx - 2 >= 0:
        df.loc[idx, "trolley_lag_2"] = df.loc[idx - 2, "uhl_ed"]

    if "trolley_lag_3" in df.columns and idx - 3 >= 0:
        df.loc[idx, "trolley_lag_3"] = df.loc[idx - 3, "uhl_ed"]

    if "trolley_lag_7" in df.columns and idx - 7 >= 0:
        df.loc[idx, "trolley_lag_7"] = df.loc[idx - 7, "uhl_ed"]

    if "trolley_roll7_mean" in df.columns:
        hist = df.loc[max(0, idx - 7): idx - 1, "uhl_ed"].dropna()
        if len(hist) > 0:
            df.loc[idx, "trolley_roll7_mean"] = hist.mean()

    if "trolley_roll7_std" in df.columns:
        hist = df.loc[max(0, idx - 7): idx - 1, "uhl_ed"].dropna()
        if len(hist) > 1:
            df.loc[idx, "trolley_roll7_std"] = hist.std()
        elif len(hist) == 1:
            df.loc[idx, "trolley_roll7_std"] = 0.0

    return df


def recursive_forecast_from_latest_features(latest_features_df: pd.DataFrame, device="cpu") -> pd.DataFrame:
    model, feature_cols, x_mean, x_std, y_mean, y_std = load_model(device)

    df = latest_features_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    # 先清理坏行
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    validate_input_df(df, feature_cols)
    df = fill_missing_for_inference(df)

    # ⚠️ 一定要在清洗后算历史长度
    hist_len = len(df)

    df = extend_future_rows(df, FORECAST_DAYS)

    preds = []

    for step in range(FORECAST_DAYS):
        idx = hist_len + step
        df = recompute_recursive_features(df, idx)

        seq = df.loc[idx - SEQ_LEN: idx - 1, feature_cols].copy()
        seq = seq.ffill().bfill()

        if seq.shape[0] != SEQ_LEN:
            raise ValueError(f"Expected {SEQ_LEN} rows, got {seq.shape[0]}")

        x = seq.values.astype(np.float32)
        x = (x - x_mean) / (x_std + 1e-8)
        x = torch.tensor(x[None], dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_scaled = model(x).cpu().numpy()[0]

        pred = pred_scaled * y_std.flatten() + y_mean.flatten()

        df.loc[idx, "uhl_ed"] = float(pred[0])
        df.loc[idx, "uhl_wait_24h"] = float(pred[1])
        df.loc[idx, "uhl_wait_75plus"] = float(pred[2])

        preds.append({
            "date": str(pd.to_datetime(df.loc[idx, "date"]).date()),
            "uhl_ed_pred": float(pred[0]),
            "uhl_wait_24h_pred": float(pred[1]),
            "uhl_wait_75plus_pred": float(pred[2]),
        })

    return pd.DataFrame(preds)