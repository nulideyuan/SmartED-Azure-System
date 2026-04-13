import pandas as pd


def get_recommended_actions(risk, drivers):
    actions = []

    driver_names = {d["name"] for d in drivers}
    high_driver_names = {d["name"] for d in drivers if d["level"] == "high"}

    if "uhl_wait_24h" in driver_names:
        actions.append("Review 24-hour wait pressure and escalation capacity")

    if "uhl_wait_75plus" in driver_names:
        actions.append("Prioritize elderly discharge coordination")

    if "uhl_ed" in high_driver_names:
        actions.append("Review surge bed readiness")

    if risk == "High":
        actions.append("Activate operational escalation review")

    if not actions:
        actions.append("Continue standard monitoring")

    return actions


def get_daily_risk(row, thresholds=None):
    if thresholds is None:
        thresholds = {
            "uhl_ed_high": 45,
            "uhl_ed_medium": 30,
            "uhl_wait_24h_high": 20,
            "uhl_wait_24h_medium": 12,
            "uhl_wait_75plus_high": 10,
            "uhl_wait_75plus_medium": 6,
        }

    reasons = []
    drivers = []

    ed = float(row["uhl_ed_pred"])
    wait24 = float(row["uhl_wait_24h_pred"])
    wait75 = float(row["uhl_wait_75plus_pred"])

    if ed >= thresholds["uhl_ed_high"]:
        reasons.append(f"High trolley count ({ed:.1f})")
        drivers.append({"name": "uhl_ed", "value": round(ed, 1), "level": "high"})
    elif ed >= thresholds["uhl_ed_medium"]:
        reasons.append(f"Elevated trolley count ({ed:.1f})")
        drivers.append({"name": "uhl_ed", "value": round(ed, 1), "level": "medium"})

    if wait24 >= thresholds["uhl_wait_24h_high"]:
        reasons.append(f"High 24h waiting ({wait24:.1f})")
        drivers.append({"name": "uhl_wait_24h", "value": round(wait24, 1), "level": "high"})
    elif wait24 >= thresholds["uhl_wait_24h_medium"]:
        reasons.append(f"Elevated 24h waiting ({wait24:.1f})")
        drivers.append({"name": "uhl_wait_24h", "value": round(wait24, 1), "level": "medium"})

    if wait75 >= thresholds["uhl_wait_75plus_high"]:
        reasons.append(f"High 75+ waiting ({wait75:.1f})")
        drivers.append({"name": "uhl_wait_75plus", "value": round(wait75, 1), "level": "high"})
    elif wait75 >= thresholds["uhl_wait_75plus_medium"]:
        reasons.append(f"Elevated 75+ waiting ({wait75:.1f})")
        drivers.append({"name": "uhl_wait_75plus", "value": round(wait75, 1), "level": "medium"})

    high_hits = sum([
        ed >= thresholds["uhl_ed_high"],
        wait24 >= thresholds["uhl_wait_24h_high"],
        wait75 >= thresholds["uhl_wait_75plus_high"],
    ])

    medium_hits = sum([
        ed >= thresholds["uhl_ed_medium"],
        wait24 >= thresholds["uhl_wait_24h_medium"],
        wait75 >= thresholds["uhl_wait_75plus_medium"],
    ])

    if high_hits >= 2:
        risk = "High"
    elif high_hits >= 1 or medium_hits >= 2:
        risk = "Medium"
    else:
        risk = "Low"

    actions = get_recommended_actions(risk, drivers)

    return {
        "risk": risk,
        "reasons": reasons if reasons else ["No major threshold breaches"],
        "drivers": drivers,
        "recommended_actions": actions
    }


def evaluate_risk(pred_df):
    daily_risks = []

    for _, row in pred_df.iterrows():
        day_result = get_daily_risk(row)

        daily_risks.append({
            "date": str(pd.to_datetime(row["date"]).date()),
            "risk": day_result["risk"],
            "reasons": day_result["reasons"],
            "drivers": day_result.get("drivers", []),
            "recommended_actions": day_result.get("recommended_actions", []),
        })

    first_3_days = daily_risks[:3]
    all_7_days = daily_risks

    first_3_risks = [x["risk"] for x in first_3_days]
    all_7_risks = [x["risk"] for x in all_7_days]

    if "High" in first_3_risks:
        overall_risk_72h = "High"
    elif "Medium" in first_3_risks:
        overall_risk_72h = "Medium"
    else:
        overall_risk_72h = "Low"

    if "High" in all_7_risks:
        overall_risk_7d = "High"
    elif "Medium" in all_7_risks:
        overall_risk_7d = "Medium"
    else:
        overall_risk_7d = "Low"

    risk_rank = {"Low": 0, "Medium": 1, "High": 2}
    peak_item = max(daily_risks, key=lambda x: risk_rank.get(x["risk"], 0))
    peak_day = peak_item["date"]

    peak_driver = "No major threshold breaches"
    if peak_item.get("drivers"):
        peak_driver = peak_item["drivers"][0]["name"]

    driver_name_map = {
        "uhl_ed": "ED trolley pressure",
        "uhl_wait_24h": "24-hour waiting pressure",
        "uhl_wait_75plus": "75+ waiting pressure",
    }
    peak_driver = driver_name_map.get(peak_driver, peak_driver)

    return {
        "daily_risks": daily_risks,
        "overall_risk_72h": overall_risk_72h,
        "overall_risk_7d": overall_risk_7d,
        "peak_day": peak_day,
        "peak_driver": peak_driver,
    }