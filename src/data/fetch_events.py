"""
Events data facade — re-exports all event functions for the feature builder.

Original earnings + VIX functions are preserved here. Macro event functions
(FOMC, CPI, NFP) are re-exported from src.data.events.* so that
build_features.py can import everything from this single module.
"""

import datetime
import os
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data.events.fomc import (
    get_fomc_dates,
    days_to_next_fomc,
    days_since_last_fomc,
    in_fomc_window,
)
from src.data.events.cpi import (
    get_cpi_dates,
    days_to_next_cpi,
    days_since_last_cpi,
    in_cpi_window,
)
from src.data.events.nfp import (
    get_nfp_dates,
    days_to_next_nfp,
    days_since_last_nfp,
    in_nfp_window,
)
from src.data.events.unified import (
    build_unified_calendar,
    get_macro_event_features,
    get_nearest_macro_event,
)


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


def get_earnings_direction(date: datetime.date, ticker: str, earnings_df: pd.DataFrame, stock_df: pd.DataFrame) -> float | None:
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

    stock_idx = stock_df.index.map(lambda x: pd.Timestamp(x).tz_localize(None))
    on_or_after = []
    for i in range(len(stock_idx)):
        if stock_idx[i] >= nearest:
            on_or_after.append(i)

    if len(on_or_after) < 2:
        return None

    close_before = float(stock_df.iloc[on_or_after[0]]["Close"])
    close_after = float(stock_df.iloc[on_or_after[1]]["Close"])

    if close_before <= 0:
        return None

    return (close_after - close_before) / close_before


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

    print("Fetching FOMC dates...")
    fomc_df = get_fomc_dates(force=force)
    print(f"  Got {len(fomc_df)} FOMC dates")

    print("Fetching CPI dates...")
    cpi_df = get_cpi_dates(force=force)
    print(f"  Got {len(cpi_df)} CPI dates")

    print("Fetching NFP dates...")
    nfp_df = get_nfp_dates(force=force)
    print(f"  Got {len(nfp_df)} NFP dates")

    print("Done.")


if __name__ == "__main__":
    fetch_all()
