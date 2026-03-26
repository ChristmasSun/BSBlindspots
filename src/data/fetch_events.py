import datetime
import os
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EARNINGS_DIR = RAW_DIR / "earnings"

TARGET_TICKERS = ["SPY", "AAPL", "NVDA", "MSFT"]


def fetch_earnings_dates(ticker: str, limit: int = 20, force: bool = False) -> pd.DataFrame:
    cache_path = EARNINGS_DIR / f"{ticker}_earnings.csv"
    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    EARNINGS_DIR.mkdir(parents=True, exist_ok=True)

    tk = yf.Ticker(ticker)
    raw = tk.get_earnings_dates(limit=limit)

    if raw is None or raw.empty:
        df = pd.DataFrame(columns=["date", "ticker"])
        df.to_csv(cache_path, index=False)
        return df

    dates = []
    for idx in raw.index:
        dt = pd.Timestamp(idx)
        dates.append(dt.normalize())

    unique_dates = sorted(set(dates))

    rows = []
    for d in unique_dates:
        rows.append({"date": d, "ticker": ticker})

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


def fetch_vix(period: str = "2y", force: bool = False) -> pd.DataFrame:
    cache_path = RAW_DIR / "vix.csv"
    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        return df

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    vix = yf.Ticker("^VIX")
    hist = vix.history(period=period)

    if hist is None or hist.empty:
        df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df.to_csv(cache_path, index=False)
        return df

    hist = hist.reset_index()
    if "Date" in hist.columns:
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)

    cols_to_keep = []
    for c in ["Date", "Open", "High", "Low", "Close", "Volume"]:
        if c in hist.columns:
            cols_to_keep.append(c)

    hist = hist[cols_to_keep]
    hist.to_csv(cache_path, index=False)
    return hist


def get_days_to_next_earnings(date: datetime.date, ticker: str, earnings_df: pd.DataFrame) -> int | None:
    ticker_df = earnings_df[earnings_df["ticker"] == ticker].copy()
    if ticker_df.empty:
        return None

    target = pd.Timestamp(date).tz_localize(None)
    future_dates = []
    for _, row in ticker_df.iterrows():
        earn_date = pd.Timestamp(row["date"]).tz_localize(None)
        if earn_date >= target:
            future_dates.append(earn_date)

    if not future_dates:
        return None

    nearest = min(future_dates)
    delta = (nearest - target).days
    return int(delta)


def get_days_since_last_earnings(date: datetime.date, ticker: str, earnings_df: pd.DataFrame) -> int | None:
    ticker_df = earnings_df[earnings_df["ticker"] == ticker].copy()
    if ticker_df.empty:
        return None

    target = pd.Timestamp(date).tz_localize(None)
    past_dates = []
    for _, row in ticker_df.iterrows():
        earn_date = pd.Timestamp(row["date"]).tz_localize(None)
        if earn_date <= target:
            past_dates.append(earn_date)

    if not past_dates:
        return None

    nearest = max(past_dates)
    delta = (target - nearest).days
    return int(delta)


def get_vix_regime(vix_value: float) -> str:
    if vix_value < 15:
        return "low"
    elif vix_value <= 25:
        return "medium"
    else:
        return "high"


def fetch_all(force: bool = False) -> None:
    for ticker in TARGET_TICKERS:
        print(f"Fetching earnings dates for {ticker}...")
        df = fetch_earnings_dates(ticker, limit=20, force=force)
        print(f"  Got {len(df)} earnings dates")

    print("Fetching VIX history...")
    vix_df = fetch_vix(force=force)
    print(f"  Got {len(vix_df)} VIX records")

    print("Done.")


if __name__ == "__main__":
    fetch_all()
