# services/forecast_service.py

from datetime import date
import pandas as pd
import requests

from services.model_client import call_model_api
from services.risk_engine import evaluate_risk
from services.missing_checker import check_missing_inputs
from services.llm_explainer import build_explanation


def run_forecast_service(anchor_date=None, device="cpu"):
    anchor_date = anchor_date or date.today()
    anchor_date_str = str(anchor_date)

    try:
        pred_json = call_model_api(anchor_date=anchor_date_str, device=device)
    except requests.HTTPError as e:
        response = e.response
        detail = None
        try:
            detail = response.json()
        except Exception:
            detail = response.text

        return {
            "status": "error",
            "message": "Forecast request failed.",
            "detail": detail,
        }

    pred_df = pd.DataFrame(pred_json["predictions"])

    meta = {
        "daily_report_status": "adls_latest",
        "weekly_attendance_status": "adls_latest",
        "weather_status": "adls_latest",
        "future_covariates_status": "carried_forward",
    }

    risk_result = evaluate_risk(pred_df)
    data_quality = check_missing_inputs(meta=meta)

    week_ending = pd.to_datetime(pred_df["date"].min()) - pd.Timedelta(days=1)

    forecast_rows = []
    daily_risk_map = {
        str(pd.to_datetime(x["date"]).date()): x
        for x in risk_result["daily_risks"]
    }

    for _, row in pred_df.iterrows():
        d = str(pd.to_datetime(row["date"]).date())
        day_risk = daily_risk_map[d]

        forecast_rows.append({
            "date": d,
            "uhl_ed_pred": float(row["uhl_ed_pred"]),
            "uhl_wait_24h_pred": float(row["uhl_wait_24h_pred"]),
            "uhl_wait_75plus_pred": float(row["uhl_wait_75plus_pred"]),
            "risk": day_risk["risk"],
            "reasons": day_risk["reasons"],
            "drivers": day_risk.get("drivers", []),
            "recommended_actions": day_risk.get("recommended_actions", []),
        })

    overall_drivers = []
    seen_driver_keys = set()

    overall_actions = []
    seen_actions = set()

    for item in forecast_rows:
        for driver in item.get("drivers", []):
            key = (driver.get("name"), driver.get("level"))
            if key not in seen_driver_keys:
                seen_driver_keys.add(key)
                overall_drivers.append(driver)

        for action in item.get("recommended_actions", []):
            if action not in seen_actions:
                seen_actions.add(action)
                overall_actions.append(action)

    llm_explanation = build_explanation(
        forecast_rows=forecast_rows,
        overall_risk_72h=risk_result["overall_risk_72h"],
        overall_risk_7d=risk_result["overall_risk_7d"],
        peak_day=str(pd.to_datetime(risk_result["peak_day"]).date()),
        peak_driver=risk_result["peak_driver"],
        recommended_actions=overall_actions,
        data_quality=data_quality,
    )

    return {
        "status": "ok",
        "run_time": pd.Timestamp.utcnow().isoformat(),
        "source_date": str(anchor_date),
        "anchor_date_used": pred_json.get("anchor_date_used"),
        "week_ending": str(week_ending.date()),
        "forecast_horizon_days": len(pred_df),

        "forecast": forecast_rows,

        "overall_risk_72h": risk_result["overall_risk_72h"],
        "overall_risk_7d": risk_result["overall_risk_7d"],
        "peak_day": str(pd.to_datetime(risk_result["peak_day"]).date()),
        "peak_driver": risk_result["peak_driver"],

        "drivers": overall_drivers,
        "recommended_actions": overall_actions,
        "data_quality": data_quality,

        # LLM结果
        "llm_explanation": llm_explanation,

        # 方便前端直接显示一行摘要
        "summary": llm_explanation.get("executive_summary", "")
    }