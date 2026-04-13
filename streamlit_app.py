import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import MODEL_API_URL, RISK_CONTROL_API_URL

st.set_page_config(
    page_title="SmartED ED Pressure Forecast",
    layout="wide"
)


# -------------------------
# API
# -------------------------
def call_prediction_api(anchor_date: str | None):
    payload = {
        "anchor_date": anchor_date if anchor_date else None,
        "device": "cpu",
    }
    resp = requests.post(MODEL_API_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def call_risk_control_api(anchor_date: str | None):
    params = {
        "anchor_date": anchor_date if anchor_date else None,
        "device": "cpu",
    }
    resp = requests.get(RISK_CONTROL_API_URL, params=params, timeout=180)
    resp.raise_for_status()
    return resp.json()


# -------------------------
# Error handling
# -------------------------
def extract_error_payload(http_error: requests.HTTPError) -> dict:
    try:
        payload = http_error.response.json()
    except Exception:
        return {"message": http_error.response.text}

    detail = payload.get("detail")
    if isinstance(detail, dict):
        return detail
    if isinstance(detail, str):
        return {"message": detail}
    if "message" in payload:
        return payload
    return payload


def render_error_panel(error_payload: dict):
    message = error_payload.get("message") or error_payload.get("detail") or "Request failed."

    st.markdown("## Forecast Status")
    st.markdown(
        f"""
        <div style="
            border:1px solid #fcd34d;
            border-radius:16px;
            padding:18px;
            background:#fffbeb;
            color:#92400e;
            line-height:1.7;
            box-sizing:border-box;
        ">
            <div style="font-size:18px;font-weight:700;margin-bottom:8px;">
                Forecast unavailable
            </div>
            <div style="font-size:15px;">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    has_detail_fields = any(
        k in error_payload for k in ["required_seq_len", "available_rows", "last_available_date"]
    )

    if has_detail_fields:
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            render_card(
                "Required sequence length",
                str(error_payload.get("required_seq_len", "N/A")),
                min_height=100
            )
        with c2:
            render_card(
                "Available rows",
                str(error_payload.get("available_rows", "N/A")),
                min_height=100
            )
        with c3:
            render_card(
                "Last available date",
                str(error_payload.get("last_available_date", "N/A")),
                min_height=100
            )


# -------------------------
# UI helpers
# -------------------------
def risk_badge_html(risk: str) -> str:
    color_map = {
        "High": "#dc2626",
        "Medium": "#d97706",
        "Low": "#16a34a",
    }
    bg_map = {
        "High": "#fee2e2",
        "Medium": "#ffedd5",
        "Low": "#dcfce7",
    }

    color = color_map.get(risk, "#6b7280")
    bg = bg_map.get(risk, "#f3f4f6")

    return f"""
    <span style="
        display:inline-block;
        padding:5px 12px;
        border-radius:999px;
        font-size:14px;
        font-weight:600;
        color:{color};
        background:{bg};
        margin-top:8px;
    ">
        {risk}
    </span>
    """


def render_card(label: str, value: str, badge: str | None = None, min_height: int = 120):
    badge_html = risk_badge_html(badge) if badge else ""

    html = f"""
    <div style="
        border:1px solid #e5e7eb;
        border-radius:16px;
        padding:18px;
        background:white;
        min-height:{min_height}px;
        box-sizing:border-box;
    ">
        <div style="font-size:14px;color:#6b7280;margin-bottom:10px;">{label}</div>
        <div style="font-size:20px;font-weight:700;color:#111827;line-height:1.4;">{value}</div>
        <div style="margin-top:10px;">{badge_html}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_text_panel(title: str, lines: list[str]):
    items_html = "".join([f"<li style='margin-bottom:8px;'>{item}</li>" for item in lines])

    st.markdown(f"## {title}")
    st.markdown(
        f"""
        <div style="
            border:1px solid #e5e7eb;
            border-radius:16px;
            padding:18px;
            background:white;
            min-height:180px;
            box-sizing:border-box;
        ">
            <ul style="margin:0;padding-left:20px;color:#111827;line-height:1.6;">
                {items_html if items_html else "<li>No data available.</li>"}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_info_panel(title: str, text: str):
    st.markdown(f"## {title}")
    st.markdown(
        f"""
        <div style="
            border:1px solid #e5e7eb;
            border-radius:16px;
            padding:18px;
            background:white;
            color:#111827;
            line-height:1.7;
            font-size:16px;
            box-sizing:border-box;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def plot_forecast_chart(forecast_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5.2))

    ax.plot(
        forecast_df["date"],
        forecast_df["uhl_ed_pred"],
        label="UHL ED Trolleys",
        linewidth=2
    )
    ax.plot(
        forecast_df["date"],
        forecast_df["uhl_wait_24h_pred"],
        label="Patients waiting >24h",
        linewidth=2
    )
    ax.plot(
        forecast_df["date"],
        forecast_df["uhl_wait_75plus_pred"],
        label="Patients >75 waiting >24h",
        linewidth=2
    )

    ax.set_title("7-Day Forecast", fontsize=18)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted value")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.xticks(rotation=45)
    st.pyplot(fig)


def render_daily_risk_timeline(forecast_df: pd.DataFrame):
    st.markdown("## Daily Risk Timeline")

    cols = st.columns(len(forecast_df))
    for i, (_, row) in enumerate(forecast_df.iterrows()):
        risk = row["risk"]
        badge = risk_badge_html(risk)

        reasons = row["reasons"]
        if isinstance(reasons, list):
            reasons_html = "<br>".join([f"• {r}" for r in reasons])
        else:
            reasons_html = str(reasons)

        cols[i].markdown(
            f"""
            <div style="
                border:1px solid #e5e7eb;
                border-radius:14px;
                padding:14px;
                background:white;
                min-height:190px;
                box-sizing:border-box;
            ">
                <div style="font-size:13px;color:#6b7280;margin-bottom:8px;">
                    {pd.to_datetime(row["date"]).date()}
                </div>
                <div>{badge}</div>
                <div style="margin-top:12px;font-size:13px;color:#374151;line-height:1.6;">
                    {reasons_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# -------------------------
# Page sections
# -------------------------
def render_prediction_view(result: dict):
    pred_df = pd.DataFrame(result.get("predictions", []))
    if pred_df.empty:
        st.warning("No prediction data returned.")
        return

    pred_df["date"] = pd.to_datetime(pred_df["date"])

    anchor_date_used = result.get("anchor_date_used", "N/A")
    horizon = str(len(pred_df))

    st.markdown("## Prediction Output")
    c1, c2 = st.columns(2)
    with c1:
        render_card("Anchor date used", str(anchor_date_used), min_height=100)
    with c2:
        render_card("Forecast horizon", f"{horizon} days", min_height=100)

    st.markdown("")
    st.markdown("## Forecast Chart")
    plot_forecast_chart(pred_df)

    st.markdown("")
    st.markdown("## Prediction Table")
    display_df = pred_df.copy()
    display_df["date"] = display_df["date"].dt.date
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_risk_control_view(result: dict):
    if result.get("status") != "ok":
        render_error_panel(result)
        return

    forecast_df = pd.DataFrame(result.get("forecast", []))
    if forecast_df.empty:
        st.warning("No forecast data returned.")
        return

    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    llm = result.get("llm_explanation", {})
    data_quality = result.get("data_quality", {})

    run_time = result.get("run_time", "")
    run_time_display = run_time.replace("T", " ")[:16] if run_time else "N/A"
    source_date = result.get("source_date", "N/A")
    week_ending = result.get("week_ending", "N/A")
    horizon = f"Next {result.get('forecast_horizon_days', 'N/A')} days"

    st.markdown("## Latest Forecast Run")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_card("Latest forecast run", run_time_display, min_height=110)
    with c2:
        render_card("Last daily report date", source_date, min_height=110)
    with c3:
        render_card("Weekly attendance source week ending", week_ending, min_height=110)
    with c4:
        render_card("Forecast horizon", horizon, min_height=110)

    st.markdown("")
    st.markdown("## Overall Risk")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_card(
            "72h risk",
            result.get("overall_risk_72h", "N/A"),
            badge=result.get("overall_risk_72h", ""),
            min_height=130
        )
    with c2:
        render_card(
            "7-day risk",
            result.get("overall_risk_7d", "N/A"),
            badge=result.get("overall_risk_7d", ""),
            min_height=130
        )
    with c3:
        render_card("Peak day", result.get("peak_day", "N/A"), min_height=130)
    with c4:
        render_card("Peak driver", result.get("peak_driver", "N/A"), min_height=130)

    st.markdown("")
    st.markdown("## Forecast Chart")
    plot_forecast_chart(forecast_df)

    st.markdown("")
    render_daily_risk_timeline(forecast_df)

    st.markdown("")
    render_info_panel(
        "Operational Summary",
        result.get("summary", "No summary available.")
    )

    st.markdown("")
    left, right = st.columns(2)

    with left:
        render_text_panel(
            "Main Risk Drivers",
            llm.get("main_risk_drivers", [])
        )
        st.markdown("")
        render_text_panel(
            "What to Watch Next 72h",
            llm.get("what_to_watch_next_72h", [])
        )

    with right:
        render_text_panel(
            "Recommended Actions",
            llm.get("recommended_actions", result.get("recommended_actions", []))
        )

        dq_text = llm.get("data_quality_note")
        if not dq_text and data_quality:
            dq_text = (
                f"Daily report: {data_quality.get('daily_report_status', 'unknown')}; "
                f"Weekly attendance: {data_quality.get('weekly_attendance_status', 'unknown')}; "
                f"Weather: {data_quality.get('weather_status', 'unknown')}; "
                f"Future covariates: {data_quality.get('future_covariates_status', 'unknown')}."
            )
        if not dq_text:
            dq_text = "No data quality information available."

        st.markdown("")
        render_info_panel("Data Quality", dq_text)


# -------------------------
# Main
# -------------------------
def main():
    st.markdown(
        """
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
            <div>
                <div style="font-size:28px;font-weight:800;color:#0f172a;">
                    SmartED ED Pressure Forecast
                </div>
                <div style="margin-top:8px;font-size:15px;color:#64748b;">
                    Latest operational forecast for UH Limerick
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    with st.sidebar:
        anchor_date = st.text_input("Anchor date (YYYY-MM-DD)", value="2026-04-12")
        view_mode = st.radio(
            "View",
            ["Risk Control View", "Prediction Table View"],
            index=0
        )
        run_btn = st.button("Run Latest Forecast", use_container_width=True)

    if not run_btn:
        st.info("Select a date on the left and click 'Run Latest Forecast'.")
        return

    try:
        if view_mode == "Prediction Table View":
            result = call_prediction_api(anchor_date=anchor_date)
            render_prediction_view(result)
        else:
            result = call_risk_control_api(anchor_date=anchor_date)
            render_risk_control_view(result)

    except requests.HTTPError as e:
        error_payload = extract_error_payload(e)
        render_error_panel(error_payload)
    except Exception as e:
        st.error(f"Failed: {e}")


if __name__ == "__main__":
    main()