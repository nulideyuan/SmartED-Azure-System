# pipeline/sources/weekly_report.py
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

WEEKLY_PAGE_URL = "https://www2.hse.ie/services/urgent-emergency-care-weekly-update/"


def get_latest_weekly_pdf_url() -> str:
    resp = requests.get(WEEKLY_PAGE_URL, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    link = soup.find("a", string=lambda s: s and "Download the report" in s)
    if link and link.get("href"):
        href = link["href"]
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return "https://www2.hse.ie" + href
        return "https://www2.hse.ie/" + href

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" in href.lower() and "Urgent_Care_Weekly_Report" in href:
            if href.startswith("http"):
                return href
            if href.startswith("/"):
                return "https://www2.hse.ie" + href

    raise ValueError("Could not find latest weekly PDF URL")


def fetch_weekly_report_pdf() -> tuple[str, bytes]:
    pdf_url = get_latest_weekly_pdf_url()
    resp = requests.get(pdf_url, timeout=60)
    resp.raise_for_status()
    return pdf_url, resp.content


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)


def extract_week_ending_from_url(pdf_url: str) -> str | None:
    m = re.search(r"Week_ending_(\d{2}_\d{2}_\d{4})", pdf_url)
    if m:
        return m.group(1).replace("_", "/")
    return None


def extract_latest_weekly_attendance_from_bytes(pdf_bytes: bytes) -> int:
    text = extract_text_from_pdf_bytes(pdf_bytes)

    patterns = [
        r"([\d,]+)\s+patients attended ED last week",
        r"National ED Attendances.*?([\d,]+)\s+patients attended ED last week",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return int(m.group(1).replace(",", ""))

    raise ValueError("Could not extract weekly attendance from PDF")


def parse_weekly_report_pdf(pdf_url: str, pdf_bytes: bytes) -> pd.DataFrame:
    attendance = extract_latest_weekly_attendance_from_bytes(pdf_bytes)
    week_ending = extract_week_ending_from_url(pdf_url)

    return pd.DataFrame([{
        "hospital": "National",
        "week_ending": week_ending,
        "attendance_weekly": attendance,
        "pdf_url": pdf_url,
    }])


def get_latest_weekly_attendance_df() -> tuple[str, bytes, pd.DataFrame]:
    pdf_url, pdf_bytes = fetch_weekly_report_pdf()
    df = parse_weekly_report_pdf(pdf_url, pdf_bytes)
    return pdf_url, pdf_bytes, df