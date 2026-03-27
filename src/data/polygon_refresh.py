from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

from src.data.export_excel import export_actual_data_workbook, export_workbook

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
POLYGON_DIR = RAW_DIR / "polygon_daily"
WATCHLIST_PATH = RAW_DIR / "polygon_top20_watchlist.json"
DEFAULT_LIMIT = 20
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_SLEEP_SECONDS = 13.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Polygon daily aggregates for a top-20 traded watchlist.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of tickers to include in the watchlist.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Recent daily lookback window for Polygon aggregates.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Pause between Polygon requests to stay inside plan limits.",
    )
    return parser.parse_args()


def build_top_traded_watchlist(limit: int = DEFAULT_LIMIT) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    for path in sorted(RAW_DIR.glob("*_daily.csv")):
        ticker = path.stem.replace("_daily", "")
        if ticker.startswith("^"):
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if df.empty or "Volume" not in df.columns:
            continue

        latest = df.iloc[-1]
        volume = float(latest.get("Volume", 0))
        close = float(latest.get("Close", 0))
        if volume <= 0:
            continue

        rows.append(
            {
                "ticker": ticker,
                "latest_volume": volume,
                "latest_close": close,
            }
        )

    rows.sort(key=lambda row: float(row["latest_volume"]), reverse=True)
    return rows[:limit]


def save_watchlist(rows: list[dict[str, float | str]]) -> None:
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "watchlist": rows,
    }
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def fetch_polygon_daily(
    rows: list[dict[str, float | str]],
    lookback_days: int,
    sleep_seconds: float,
) -> None:
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("POLYGON_API_KEY not set, skipping Polygon daily refresh.")
        return

    client = RESTClient(api_key)
    POLYGON_DIR.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    for index, row in enumerate(rows):
        ticker = str(row["ticker"])
        print(f"Refreshing Polygon daily aggregates for {ticker}...")
        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date,
                limit=5000,
            )
        except Exception as exc:
            print(f"  {ticker}: Polygon request failed ({exc})")
            if index < len(rows) - 1:
                time.sleep(sleep_seconds)
            continue

        records: list[dict[str, float | int | str | None]] = []
        for agg in aggs:
            records.append(
                {
                    "ticker": ticker,
                    "date": datetime.fromtimestamp(agg.timestamp / 1000).strftime("%Y-%m-%d"),
                    "open": agg.open,
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": agg.volume,
                    "vwap": getattr(agg, "vwap", None),
                    "transactions": getattr(agg, "transactions", None),
                }
            )

        df = pd.DataFrame(records)
        path = POLYGON_DIR / f"{ticker}_polygon_daily.csv"
        df.to_csv(path, index=False)
        print(f"  {ticker}: {len(df)} rows -> {path}")
        if index < len(rows) - 1:
            time.sleep(sleep_seconds)


def main() -> None:
    args = parse_args()
    rows = build_top_traded_watchlist(limit=args.limit)
    save_watchlist(rows)
    fetch_polygon_daily(
        rows,
        lookback_days=args.lookback_days,
        sleep_seconds=args.sleep_seconds,
    )
    workbook = export_workbook()
    actual_workbook = export_actual_data_workbook()
    print(f"Workbook updated: {workbook}")
    print(f"Actual-data workbook updated: {actual_workbook}")


if __name__ == "__main__":
    main()
