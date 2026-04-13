from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from services.adls_reader import read_latest_features_from_adls
from services.inference import recursive_forecast_from_latest_features
from config import SEQ_LEN

app = FastAPI(title="SmartED Model API")


class PredictRequest(BaseModel):
    anchor_date: Optional[str] = None
    device: str = "cpu"
    note: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        latest_df = read_latest_features_from_adls()
        latest_df["date"] = pd.to_datetime(latest_df["date"], errors="coerce").dt.normalize()
        latest_df = latest_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if req.anchor_date:
            anchor_date = pd.to_datetime(req.anchor_date, errors="coerce")
            if pd.isna(anchor_date):
                raise HTTPException(status_code=400, detail="Invalid anchor_date format. Use YYYY-MM-DD.")
            anchor_date = anchor_date.normalize()

            latest_df = latest_df[latest_df["date"] <= anchor_date].copy()

            if latest_df.empty:
                raise HTTPException(
                    status_code=400,
                    detail=f"No data available on or before {anchor_date.date()}."
                )

        if len(latest_df) < SEQ_LEN:
            last_available = latest_df["date"].max().date() if not latest_df.empty else None
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Current date cannot be predicted because historical data is insufficient.",
                    "required_seq_len": SEQ_LEN,
                    "available_rows": int(len(latest_df)),
                    "last_available_date": str(last_available),
                }
            )

        pred_df = recursive_forecast_from_latest_features(latest_df, device=req.device)

        used_anchor_date = str(latest_df["date"].max().date())

        return {
            "anchor_date_used": used_anchor_date,
            "predictions": pred_df.to_dict(orient="records")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))