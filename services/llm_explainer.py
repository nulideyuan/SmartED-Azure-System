# services/llm_explainer.py

import json
from typing import Any, Dict, List

from openai import OpenAI

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_BASE_URL,
    AZURE_OPENAI_DEPLOYMENT_NAME,
)


def _get_client() -> OpenAI:
    if not AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY is missing.")
    if not AZURE_OPENAI_BASE_URL:
        raise ValueError("AZURE_OPENAI_BASE_URL is missing.")
    if not AZURE_OPENAI_DEPLOYMENT_NAME:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME is missing.")

    return OpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=AZURE_OPENAI_BASE_URL,
    )


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = _strip_code_fence(text)
    return json.loads(text)


def _build_fallback_explanation(
    forecast_rows: List[Dict[str, Any]],
    overall_risk_72h: str,
    overall_risk_7d: str,
    peak_day: str,
    peak_driver: str,
    recommended_actions: List[str],
    data_quality: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "executive_summary": (
            f"The forecast indicates {overall_risk_72h.lower()} risk over the next 72 hours "
            f"and {overall_risk_7d.lower()} risk over the next 7 days. "
            f"The peak pressure day is {peak_day}, mainly driven by {peak_driver}."
        ),
        "overall_risk_72h": overall_risk_72h,
        "overall_risk_7d": overall_risk_7d,
        "peak_day": peak_day,
        "peak_driver": peak_driver,
        "main_risk_drivers": [peak_driver],
        "recommended_actions": recommended_actions[:5],
        "what_to_watch_next_72h": [
            "Monitor changes in 24-hour waits",
            "Monitor older patient waiting pressure",
            "Review trolley pressure trend"
        ],
        "data_quality_note": (
            "Input sources were loaded from ADLS latest snapshots. "
            f"Daily report: {data_quality.get('daily_report_status', 'unknown')}; "
            f"Weekly attendance: {data_quality.get('weekly_attendance_status', 'unknown')}; "
            f"Weather: {data_quality.get('weather_status', 'unknown')}."
        )
    }


def build_explanation(
    forecast_rows: List[Dict[str, Any]],
    overall_risk_72h: str,
    overall_risk_7d: str,
    peak_day: str,
    peak_driver: str,
    recommended_actions: List[str],
    data_quality: Dict[str, Any],
) -> Dict[str, Any]:
    """
    输入结构化预测+规则风险结果，输出 LLM 解释。
    """
    client = _get_client()

    payload = {
        "forecast_rows": forecast_rows,
        "overall_risk_72h": overall_risk_72h,
        "overall_risk_7d": overall_risk_7d,
        "peak_day": peak_day,
        "peak_driver": peak_driver,
        "recommended_actions": recommended_actions,
        "data_quality": data_quality,
    }

    system_prompt = """
You are a hospital operations risk assistant for emergency department forecasting.

Your task:
- Read the structured ED forecast and risk outputs.
- Produce concise operational guidance for hospital managers.
- Use only the provided information.
- Do not invent causes like staffing, weather, ambulance arrivals, influenza, or bed shortages unless they are explicitly present.
- Keep the answer professional, practical, and suitable for a dashboard.

Return STRICTLY valid JSON with this structure:
{
  "executive_summary": "2-4 sentence summary",
  "overall_risk_72h": "Low or Medium or High",
  "overall_risk_7d": "Low or Medium or High",
  "peak_day": "YYYY-MM-DD",
  "peak_driver": "short phrase",
  "main_risk_drivers": ["...", "...", "..."],
  "recommended_actions": ["...", "...", "..."],
  "what_to_watch_next_72h": ["...", "...", "..."],
  "data_quality_note": "one short sentence"
}
""".strip()

    user_prompt = f"""
Structured input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content
        parsed = _safe_json_loads(content)

        # 兜底补字段
        parsed.setdefault("overall_risk_72h", overall_risk_72h)
        parsed.setdefault("overall_risk_7d", overall_risk_7d)
        parsed.setdefault("peak_day", peak_day)
        parsed.setdefault("peak_driver", peak_driver)
        parsed.setdefault("recommended_actions", recommended_actions[:5])

        return parsed

    except Exception:
        return _build_fallback_explanation(
            forecast_rows=forecast_rows,
            overall_risk_72h=overall_risk_72h,
            overall_risk_7d=overall_risk_7d,
            peak_day=peak_day,
            peak_driver=peak_driver,
            recommended_actions=recommended_actions,
            data_quality=data_quality,
        )