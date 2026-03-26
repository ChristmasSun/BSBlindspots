import os
import math
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "treasury_3mo.csv"
BASE_URL = "https://www.alphavantage.co/query"


def fetch_treasury_yield_raw() -> list[dict]:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise EnvironmentError("ALPHA_VANTAGE_API_KEY not found in .env")

    params = {
        "function": "TREASURY_YIELD",
        "interval": "daily",
        "maturity": "3month",
        "apikey": api_key,
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()

    if "data" not in payload:
        raise ValueError(f"Unexpected API response: {payload}")

    return payload["data"]


def parse_treasury_data(raw_data: list[dict]) -> pd.DataFrame:
    dates = []
    values = []
    for entry in raw_data:
        date_str = entry["date"]
        value_str = entry["value"]
        dates.append(date_str)
        if value_str == ".":
            values.append(math.nan)
        else:
            values.append(float(value_str) / 100.0)

    df = pd.DataFrame({"risk_free_rate": values}, index=pd.to_datetime(dates))
    df.index.name = "date"
    df = df.sort_index()
    df["risk_free_rate"] = df["risk_free_rate"].ffill()

    return df


def load_treasury_rates(force_refresh: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not force_refresh:
        df = pd.read_csv(CACHE_PATH, index_col="date", parse_dates=True)
        return df

    raw_data = fetch_treasury_yield_raw()
    df = parse_treasury_data(raw_data)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH)

    return df


def get_risk_free_rate(date: str) -> float:
    df = load_treasury_rates()
    target = pd.Timestamp(date)

    if target in df.index:
        return float(df.loc[target, "risk_free_rate"])

    mask = df.index <= target
    if not mask.any():
        return float(df.iloc[0]["risk_free_rate"])

    return float(df.loc[mask].iloc[-1]["risk_free_rate"])


def main() -> None:
    print("Fetching 3-month Treasury yield data...")
    df = load_treasury_rates(force_refresh=True)
    print(f"Cached {len(df)} records to {CACHE_PATH}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Latest rate: {df.iloc[-1]['risk_free_rate']:.4f}")
    print(df.tail())


if __name__ == "__main__":
    main()
