"""
Earnings date collection — thin wrapper around yfinance.

Re-exports the core earnings logic from the existing fetch_events module
so that everything can be accessed through the events package uniformly.
Adds batch collection and the same start/end filtering interface as the
macro event modules.
"""

import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "events"

TARGET_TICKERS = ["SPY", "AAPL", "NVDA", "MSFT"]


def get_earnings_dates(
    ticker: str,
    limit: int = 20,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch earnings announcement dates for a single ticker.

    Args:
        ticker: Stock symbol (e.g. "AAPL").
        limit:  How many earnings dates to request from yfinance.
        start:  Filter to dates on or after this (YYYY-MM-DD).
        end:    Filter to dates on or before this (YYYY-MM-DD).
        force:  Re-fetch even if cached.

    Returns:
        DataFrame with columns: [date, ticker, event_type]
    """
    cache_path = RAW_DIR / f"{ticker}_earnings.csv"

    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        tk = yf.Ticker(ticker)
        raw = tk.get_earnings_dates(limit=limit)

        if raw is None or raw.empty:
            df = pd.DataFrame(columns=["date", "ticker", "event_type"])
            df.to_csv(cache_path, index=False)
            return df

        dates = []
        for idx in raw.index:
            dt = pd.Timestamp(idx)
            if dt.tzinfo is not None:
                dt = dt.tz_localize(None)
            dates.append(dt.normalize())

        unique_dates = sorted(set(dates))

        rows = []
        for d in unique_dates:
            rows.append({"date": d, "ticker": ticker, "event_type": "earnings"})

        df = pd.DataFrame(rows)
        df.to_csv(cache_path, index=False)

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    return df.reset_index(drop=True)


def get_all_earnings(
    tickers: list[str] | None = None,
    limit: int = 20,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch earnings dates for multiple tickers and concatenate.

    Args:
        tickers: List of symbols. Defaults to TARGET_TICKERS.
        limit:   Per-ticker limit passed to yfinance.
        start:   Date filter lower bound.
        end:     Date filter upper bound.
        force:   Re-fetch all.

    Returns:
        DataFrame with columns: [date, ticker, event_type], sorted by date.
    """
    if tickers is None:
        tickers = TARGET_TICKERS

    frames = []
    for t in tickers:
        df = get_earnings_dates(t, limit=limit, start=start, end=end, force=force)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "event_type"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def days_to_next_earnings(
    date: datetime.date, ticker: str, earnings_df: pd.DataFrame
) -> int | None:
    """Calendar days from `date` to the next earnings for `ticker`."""
    ticker_df = earnings_df[earnings_df["ticker"] == ticker]
    if ticker_df.empty:
        return None

    target = pd.Timestamp(date).tz_localize(None)
    future = []
    for _, row in ticker_df.iterrows():
        dt = pd.Timestamp(row["date"]).tz_localize(None)
        if dt >= target:
            future.append(dt)
    if not future:
        return None
    return int((min(future) - target).days)


def days_since_last_earnings(
    date: datetime.date, ticker: str, earnings_df: pd.DataFrame
) -> int | None:
    """Calendar days since the most recent earnings for `ticker`."""
    ticker_df = earnings_df[earnings_df["ticker"] == ticker]
    if ticker_df.empty:
        return None

    target = pd.Timestamp(date).tz_localize(None)
    past = []
    for _, row in ticker_df.iterrows():
        dt = pd.Timestamp(row["date"]).tz_localize(None)
        if dt <= target:
            past.append(dt)
    if not past:
        return None
    return int((target - max(past)).days)


if __name__ == "__main__":
    df = get_all_earnings(force=True)
    print(f"Earnings dates: {len(df)} total across {df['ticker'].nunique()} tickers")
    print(df.head(10))
    print("...")
    print(df.tail(10))
