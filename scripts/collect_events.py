#!/usr/bin/env python3
"""
Collect all events data and save to data/raw/events/.

Run this script to build/refresh the full events dataset:

    uv run python scripts/collect_events.py

Options:
    --force         Re-fetch everything even if cached
    --start DATE    Only include events on or after DATE (YYYY-MM-DD)
    --end DATE      Only include events on or before DATE (YYYY-MM-DD)
    --tickers T,T   Comma-separated tickers for earnings (default: SPY,AAPL,NVDA,MSFT)

Output files (in data/raw/events/):
    fomc_dates.csv         FOMC meeting dates
    cpi_dates.csv          CPI release dates
    nfp_dates.csv          NFP/jobs report dates
    {TICKER}_earnings.csv  Per-ticker earnings dates
    unified_calendar.csv   All events merged + sorted

All CSVs have columns: date, event_type (+ ticker for earnings).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.events.fomc import get_fomc_dates
from src.data.events.cpi import get_cpi_dates
from src.data.events.nfp import get_nfp_dates
from src.data.events.earnings import get_all_earnings
from src.data.events.unified import build_unified_calendar


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect all events data")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--tickers",
        type=str,
        default="SPY,AAPL,NVDA,MSFT",
        help="Comma-separated tickers for earnings",
    )
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]

    print("=" * 60)
    print("BSBlindspots — Events Data Collection")
    print("=" * 60)
    if args.start or args.end:
        print(f"Date range: {args.start or '(no limit)'} to {args.end or '(no limit)'}")
    print(f"Tickers for earnings: {tickers}")
    print(f"Force refresh: {args.force}")
    print()

    print("[1/4] FOMC meeting dates...")
    fomc_df = get_fomc_dates(start=args.start, end=args.end, force=args.force)
    print(f"       {len(fomc_df)} FOMC dates collected")

    print("[2/4] CPI release dates...")
    cpi_df = get_cpi_dates(start=args.start, end=args.end, force=args.force)
    print(f"       {len(cpi_df)} CPI dates collected")

    print("[3/4] NFP release dates...")
    nfp_df = get_nfp_dates(start=args.start, end=args.end, force=args.force)
    print(f"       {len(nfp_df)} NFP dates collected")

    print("[4/4] Earnings dates...")
    earnings_df = get_all_earnings(
        tickers=tickers, start=args.start, end=args.end, force=args.force
    )
    print(f"       {len(earnings_df)} earnings dates across {earnings_df['ticker'].nunique()} tickers")

    print()
    print("Building unified calendar...")
    unified = build_unified_calendar(
        tickers=tickers, start=args.start, end=args.end, force=args.force
    )
    print(f"Unified calendar: {len(unified)} total events")
    print()
    print("Event type breakdown:")
    for etype, count in unified["event_type"].value_counts().items():
        print(f"  {etype}: {count}")
    print()
    print(f"Date range: {unified['date'].min()} to {unified['date'].max()}")
    print()

    output_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "events"
    print(f"All data saved to: {output_dir}")
    print()

    files = sorted(output_dir.glob("*.csv"))
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:30s} ({size_kb:.1f} KB)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
