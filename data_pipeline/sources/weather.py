import requests
import pandas as pd


def fetch_weather(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "forecast_days": 7,
        "timezone": "Europe/Dublin"
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    daily = payload.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "temperature_min": daily.get("temperature_2m_min", []),
        "precipitation_sum": daily.get("precipitation_sum", []),
        "windspeed_max": daily.get("windspeed_10m_max", []),
    })

    return payload, df