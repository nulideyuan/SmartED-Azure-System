import requests
from config import MODEL_API_URL

def call_model_api(anchor_date=None, device="cpu"):
    payload = {
        "anchor_date": anchor_date,
        "device": device
    }
    response = requests.post(MODEL_API_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()