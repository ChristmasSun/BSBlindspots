"""
Unified event calendar — merges all event sources into one timeline
and provides feature-computation helpers for the ML pipeline.

This module is the main interface your feature builder should call.
It combines FOMC, CPI, NFP, and earnings into a single sorted calendar
and exposes helpers that return all macro-event features for a given date.
"""

import datetime
from pathlib import Path

import pandas as pd

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
from src.data.events.earnings import (
    get_all_earnings,
    days_to_next_earnings,
    days_since_last_earnings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "events"


def build_unified_calendar(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Build a single sorted calendar of all event types.

    Args:
        tickers: For earnings. Defaults to ["SPY", "AAPL", "NVDA", "MSFT"].
        start:   Date filter lower bound (YYYY-MM-DD).
        end:     Date filter upper bound (YYYY-MM-DD).
        force:   Re-fetch/regenerate all sources.

    Returns:
        DataFrame with columns: [date, event_type, ticker]
        - ticker is NaN for macro events (fomc, cpi, nfp).
        - Sorted by date ascending.
    """
    fomc_df = get_fomc_dates(start=start, end=end, force=force)
    fomc_df["ticker"] = None

    cpi_df = get_cpi_dates(start=start, end=end, force=force)
    cpi_df["ticker"] = None

    nfp_df = get_nfp_dates(start=start, end=end, force=force)
    nfp_df["ticker"] = None

    earnings_df = get_all_earnings(tickers=tickers, start=start, end=end, force=force)

    frames = [fomc_df, cpi_df, nfp_df]
    if not earnings_df.empty:
        frames.append(earnings_df)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
    combined = combined.sort_values("date").reset_index(drop=True)

    cache_path = RAW_DIR / "unified_calendar.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path, index=False)

    return combined


def get_macro_event_features(
    date: datetime.date,
    fomc_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    nfp_df: pd.DataFrame,
    fomc_window: int = 3,
    cpi_window: int = 2,
    nfp_window: int = 2,
) -> dict:
    """Compute all macro-event distance features for a single date.

    This is the function the feature builder should call per-row.

    Args:
        date: The trade date.
        fomc_df: Output of get_fomc_dates().
        cpi_df:  Output of get_cpi_dates().
        nfp_df:  Output of get_nfp_dates().
        fomc_window: Days around FOMC to flag as "in window".
        cpi_window:  Days around CPI to flag.
        nfp_window:  Days around NFP to flag.

    Returns:
        Dict with keys:
            days_to_fomc, days_since_fomc, in_fomc_window,
            days_to_cpi, days_since_cpi, in_cpi_window,
            days_to_nfp, days_since_nfp, in_nfp_window,
            in_any_macro_window
    """
    d_to_fomc = days_to_next_fomc(date, fomc_df)
    d_since_fomc = days_since_last_fomc(date, fomc_df)
    fomc_win = 1 if in_fomc_window(date, fomc_df, window=fomc_window) else 0

    d_to_cpi = days_to_next_cpi(date, cpi_df)
    d_since_cpi = days_since_last_cpi(date, cpi_df)
    cpi_win = 1 if in_cpi_window(date, cpi_df, window=cpi_window) else 0

    d_to_nfp = days_to_next_nfp(date, nfp_df)
    d_since_nfp = days_since_last_nfp(date, nfp_df)
    nfp_win = 1 if in_nfp_window(date, nfp_df, window=nfp_window) else 0

    in_any = 1 if (fomc_win or cpi_win or nfp_win) else 0

    return {
        "days_to_fomc": d_to_fomc,
        "days_since_fomc": d_since_fomc,
        "in_fomc_window": fomc_win,
        "days_to_cpi": d_to_cpi,
        "days_since_cpi": d_since_cpi,
        "in_cpi_window": cpi_win,
        "days_to_nfp": d_to_nfp,
        "days_since_nfp": d_since_nfp,
        "in_nfp_window": nfp_win,
        "in_any_macro_window": in_any,
    }


def get_nearest_macro_event(
    date: datetime.date,
    fomc_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    nfp_df: pd.DataFrame,
) -> dict:
    """Find the nearest upcoming macro event of any type.

    Returns:
        Dict with keys: nearest_macro_event_type, days_to_nearest_macro
    """
    candidates = []

    d = days_to_next_fomc(date, fomc_df)
    if d is not None:
        candidates.append(("fomc", d))

    d = days_to_next_cpi(date, cpi_df)
    if d is not None:
        candidates.append(("cpi", d))

    d = days_to_next_nfp(date, nfp_df)
    if d is not None:
        candidates.append(("nfp", d))

    if not candidates:
        return {"nearest_macro_event_type": None, "days_to_nearest_macro": None}

    best = min(candidates, key=lambda x: x[1])
    return {"nearest_macro_event_type": best[0], "days_to_nearest_macro": best[1]}


if __name__ == "__main__":
    print("Building unified event calendar...")
    cal = build_unified_calendar(force=True)
    print(f"Total events: {len(cal)}")
    print(f"\nEvent type counts:")
    print(cal["event_type"].value_counts().to_string())
    print(f"\nDate range: {cal['date'].min()} to {cal['date'].max()}")
    print(f"\nSample (first 15 rows):")
    print(cal.head(15).to_string(index=False))

    print("\n\nExample feature computation for 2024-03-19 (day before FOMC):")
    fomc_df = get_fomc_dates()
    cpi_df = get_cpi_dates()
    nfp_df = get_nfp_dates()
    features = get_macro_event_features(
        datetime.date(2024, 3, 19), fomc_df, cpi_df, nfp_df
    )
    for k, v in features.items():
        print(f"  {k}: {v}")

    nearest = get_nearest_macro_event(
        datetime.date(2024, 3, 19), fomc_df, cpi_df, nfp_df
    )
    print(f"  nearest_macro_event_type: {nearest['nearest_macro_event_type']}")
    print(f"  days_to_nearest_macro: {nearest['days_to_nearest_macro']}")
