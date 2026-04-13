# data_pipeline/sources/daily_report.py
import urllib3
urllib3.disable_warnings()

import re
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    )
}


def build_hse_url(date_obj: pd.Timestamp) -> str:
    date_str = pd.to_datetime(date_obj).strftime("%d/%m/%Y")
    return f"https://uec.hse.ie/uec/TGAR.php?EDDATE={date_str.replace('/', '%2F')}"


def fetch_daily_report_html(date_obj: pd.Timestamp) -> tuple[str, str]:
    url = build_hse_url(date_obj)
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return url, r.text


def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\xa0", " ").split()).strip().lower()


def is_uhl_row(first_cell: str) -> bool:
    """
    More tolerant matching for UHL row names.
    """
    x = normalize_text(first_cell)

    exact_names = {
        "uh limerick",
        "university hospital limerick",
        "u.h. limerick",
        "u h limerick",
    }

    if x in exact_names:
        return True

    # fallback fuzzy match
    if "limerick" in x and ("hospital" in x or x.startswith("uh")):
        return True

    return False

def _to_int_or_none(x):
    x = str(x).replace(",", "").strip()
    if x == "":
        return None
    m = re.search(r"-?\d+", x)
    return int(m.group()) if m else None


def _to_int_or_zero(x):
    v = _to_int_or_none(x)
    return 0 if v is None else v

def extract_uhl_row(html_text: str, debug: bool = False):
    soup = BeautifulSoup(html_text, "html.parser")
    cells = soup.find_all("td")

    for i, cell in enumerate(cells):
        if cell.get_text(strip=True) == "UH Limerick":
            try:
                if debug:
                    print("[daily][debug] FOUND UH Limerick at cell index:", i)
                    for j in range(i, min(i + 15, len(cells))):
                        print(
                            f"[cell {j}] text={cells[j].get_text(strip=True)!r}, attrs={dict(cells[j].attrs)}"
                        )

                return {
                    "uhl_ed": int(cells[i + 2].get_text(strip=True) or 0),
                    "uhl_ward": int(cells[i + 3].get_text(strip=True) or 0),
                    "uhl_total": int(cells[i + 4].get_text(strip=True) or 0),
                    "uhl_surge": int(cells[i + 6].get_text(strip=True) or 0),
                    "uhl_dtoc": int(cells[i + 8].get_text(strip=True) or 0),
                    "uhl_wait_24h": int(cells[i + 10].get_text(strip=True) or 0),
                    "uhl_wait_75plus": int(cells[i + 12].get_text(strip=True) or 0),
                }

            except Exception as e:
                if debug:
                    print("[daily][debug] parse error:", e)
                return None

    return None


def get_daily_uhl(
    date_obj: pd.Timestamp,
    retries: int = 3,
    sleep_sec: float = 1.5,
    debug: bool = False
):
    last_err = None
    date_obj = pd.to_datetime(date_obj).normalize()

    for attempt in range(retries):
        try:
            url, html = fetch_daily_report_html(date_obj)
            row = extract_uhl_row(html, debug=False)

            if row is None:
                raise ValueError(f"UHL row not found for {date_obj.date()}")

            row["date"] = date_obj
            row["source_url"] = url
            return html, row

        except Exception as e:
            last_err = e
            if debug:
                print(f"[daily][debug] attempt {attempt + 1}/{retries} failed for {date_obj.date()}: {e}")
            time.sleep(sleep_sec + random.uniform(0.2, 1.0))

    raise RuntimeError(f"Failed HSE fetch for {date_obj.date()}: {last_err}")


def get_latest_daily_uhl_df(
    run_date=None,
    max_lookback_days: int = 3,
    debug: bool = True
) -> tuple[str, pd.DataFrame]:
    """
    Try requested day first, then fall back to previous days.
    """
    if run_date is None:
        run_date = pd.Timestamp.today().normalize()

    run_date = pd.to_datetime(run_date).normalize()
    last_err = None

    for offset in range(max_lookback_days + 1):
        try_date = run_date - pd.Timedelta(days=offset)

        try:
            if debug:
                print(f"[daily] trying HSE daily report for {try_date.date()}")

            html, row = get_daily_uhl(try_date, debug=debug)
            df = pd.DataFrame([row])

            df["requested_date"] = run_date
            df["source_date"] = try_date
            df["daily_report_status"] = "live" if offset == 0 else f"fallback_{offset}d"

            print(
                f"[daily] success: using {try_date.date()} "
                f"for requested date {run_date.date()} "
                f"status={df['daily_report_status'].iloc[0]}"
            )
            return html, df

        except Exception as e:
            last_err = e
            print(f"[daily] failed for {try_date.date()}: {e}")

    raise RuntimeError(
        f"Failed HSE fetch from {run_date.date()} back to "
        f"{(run_date - pd.Timedelta(days=max_lookback_days)).date()}: {last_err}"
    )