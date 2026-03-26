import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_TICKERS = ["SPY", "AAPL", "NVDA", "MSFT"]
CACHE_MAX_AGE_HOURS = 12
HISTORY_YEARS = 2


def _cache_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker}_daily.csv"


def _cache_is_fresh(path: Path, max_age_hours: int = CACHE_MAX_AGE_HOURS) -> bool:
    if not path.exists():
        return False
    mod_time = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - mod_time
    if age < timedelta(hours=max_age_hours):
        return True
    return False


def compute_log_returns(prices: pd.Series) -> pd.Series:
    log_returns: list[float] = []
    log_returns.append(float("nan"))
    values = prices.values
    for i in range(1, len(values)):
        prev = float(values[i - 1])
        curr = float(values[i])
        if prev > 0 and curr > 0:
            log_returns.append(math.log(curr / prev))
        else:
            log_returns.append(float("nan"))
    return pd.Series(log_returns, index=prices.index, name="log_return")


def fetch_ticker(ticker: str, period_years: int = HISTORY_YEARS, force: bool = False) -> pd.DataFrame:
    cache_file = _cache_path(ticker)

    if not force and _cache_is_fresh(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)

    data = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = pd.DataFrame()
    df["Open"] = data["Open"]
    df["High"] = data["High"]
    df["Low"] = data["Low"]
    df["Close"] = data["Close"]
    df["Volume"] = data["Volume"]
    df["log_return"] = compute_log_returns(df["Close"])

    df.index.name = "Date"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file)

    return df


def fetch_all(tickers: list[str] | None = None, force: bool = False) -> dict[str, pd.DataFrame]:
    if tickers is None:
        tickers = DEFAULT_TICKERS

    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df = fetch_ticker(ticker, force=force)
        results[ticker] = df
        print(f"  {len(df)} rows, date range: {df.index.min()} to {df.index.max()}")

    return results


if __name__ == "__main__":
    fetch_all()
