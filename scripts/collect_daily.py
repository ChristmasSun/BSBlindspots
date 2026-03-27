"""Daily options chain collector for BS Blindspots.

Fetches current options chains for all target tickers and saves
date-prefixed snapshots so we accumulate daily data over time.
Idempotent: skips tickers that already have today's file.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetch_options import fetch_current_chains

TARGET_TICKERS = ["SPY", "AAPL", "NVDA", "MSFT"]
DAILY_DIR = PROJECT_ROOT / "data" / "raw" / "options" / "daily"


def _daily_path(ticker: str, date_str: str) -> Path:
    return DAILY_DIR / f"{date_str}_{ticker}.csv"


def _vix_path(date_str: str) -> Path:
    return DAILY_DIR / f"{date_str}_VIX.csv"


def collect_options(date_str: str) -> dict[str, int]:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, int] = {}

    for ticker in TARGET_TICKERS:
        path = _daily_path(ticker, date_str)
        if path.exists():
            existing = pd.read_csv(path)
            print(f"  SKIP {ticker} — already collected ({len(existing)} contracts)")
            results[ticker] = len(existing)
            continue

        print(f"  Fetching {ticker}...")
        try:
            df = fetch_current_chains(ticker, use_cache=False)
            df.to_csv(path, index=False)
            print(f"  OK   {ticker} — {len(df)} contracts saved")
            results[ticker] = len(df)
        except Exception as e:
            print(f"  FAIL {ticker} — {e}")
            results[ticker] = -1

    return results


def collect_vix(date_str: str) -> float | None:
    path = _vix_path(date_str)
    if path.exists():
        df = pd.read_csv(path)
        vix_close = float(df["Close"].iloc[0])
        print(f"  SKIP VIX — already collected (level={vix_close:.2f})")
        return vix_close

    print(f"  Fetching VIX...")
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if hist is None or hist.empty:
            print(f"  FAIL VIX — no data returned")
            return None

        hist = hist.reset_index()
        if "Date" in hist.columns:
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)

        cols_to_keep = []
        for c in ["Date", "Open", "High", "Low", "Close", "Volume"]:
            if c in hist.columns:
                cols_to_keep.append(c)

        hist = hist[cols_to_keep]
        hist.to_csv(path, index=False)

        vix_close = float(hist["Close"].iloc[0])
        print(f"  OK   VIX — level={vix_close:.2f}")
        return vix_close
    except Exception as e:
        print(f"  FAIL VIX — {e}")
        return None


def main() -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Daily Collection: {date_str} ===")
    print()

    print("[Options Chains]")
    results = collect_options(date_str)
    print()

    print("[VIX]")
    vix_level = collect_vix(date_str)
    print()

    print("=== Summary ===")
    total_contracts = 0
    for ticker in TARGET_TICKERS:
        count = results.get(ticker, -1)
        if count >= 0:
            total_contracts += count
            print(f"  {ticker}: {count} contracts")
        else:
            print(f"  {ticker}: FAILED")

    if vix_level is not None:
        print(f"  VIX: {vix_level:.2f}")
    else:
        print(f"  VIX: FAILED")

    print(f"  Total contracts: {total_contracts}")
    print(f"  Output dir: {DAILY_DIR}")
    print()


if __name__ == "__main__":
    main()
