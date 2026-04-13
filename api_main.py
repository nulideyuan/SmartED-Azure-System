from fastapi import FastAPI, HTTPException, Query
from services.forecast_service import run_forecast_service

app = FastAPI(title="SmartED Risk Control API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast-risk-control")
def forecast_risk_control(
    anchor_date: str | None = Query(default=None),
    device: str = Query(default="cpu")
):
    try:
        result = run_forecast_service(anchor_date=anchor_date, device=device)

        if result.get("status") == "error":
            return result

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }