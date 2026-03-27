"""
FOMC (Federal Open Market Committee) meeting dates.

Source: Federal Reserve published schedules at federalreserve.gov.
Includes all statement release dates (the decision day of each meeting).
Two-day meetings list the second day (announcement day).

Coverage: 2020-01-01 through 2026-12-31.
"""

import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "events"

FOMC_DATES: list[str] = [
    # 2020
    "2020-01-29",
    "2020-03-03",
    "2020-03-15",  # emergency cut
    "2020-04-29",
    "2020-06-10",
    "2020-07-29",
    "2020-09-16",
    "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27",
    "2021-03-17",
    "2021-04-28",
    "2021-06-16",
    "2021-07-28",
    "2021-09-22",
    "2021-11-03",
    "2021-12-15",
    # 2022
    "2022-01-26",
    "2022-03-16",
    "2022-05-04",
    "2022-06-15",
    "2022-07-27",
    "2022-09-21",
    "2022-11-02",
    "2022-12-14",
    # 2023
    "2023-02-01",
    "2023-03-22",
    "2023-05-03",
    "2023-06-14",
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13",
    # 2024
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    # 2025
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-17",
    # 2026 (scheduled)
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-16",
]


def get_fomc_dates(
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of FOMC meeting/announcement dates.

    Args:
        start: Filter to dates on or after this (YYYY-MM-DD). None = no lower bound.
        end:   Filter to dates on or before this (YYYY-MM-DD). None = no upper bound.
        force: If True, regenerate even if cached file exists.

    Returns:
        DataFrame with columns: [date, event_type]
        date is datetime64, event_type is always "fomc".
    """
    cache_path = RAW_DIR / "fomc_dates.csv"

    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        for d in FOMC_DATES:
            rows.append({"date": pd.Timestamp(d), "event_type": "fomc"})
        df = pd.DataFrame(rows)
        df.to_csv(cache_path, index=False)

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    return df.reset_index(drop=True)


def days_to_next_fomc(date: datetime.date, fomc_df: pd.DataFrame) -> int | None:
    """Calendar days from `date` to the next FOMC announcement."""
    target = pd.Timestamp(date).tz_localize(None)
    future = []
    for _, row in fomc_df.iterrows():
        dt = pd.Timestamp(row["date"]).tz_localize(None)
        if dt >= target:
            future.append(dt)
    if not future:
        return None
    return int((min(future) - target).days)


def days_since_last_fomc(date: datetime.date, fomc_df: pd.DataFrame) -> int | None:
    """Calendar days from the most recent past FOMC announcement to `date`."""
    target = pd.Timestamp(date).tz_localize(None)
    past = []
    for _, row in fomc_df.iterrows():
        dt = pd.Timestamp(row["date"]).tz_localize(None)
        if dt <= target:
            past.append(dt)
    if not past:
        return None
    return int((target - max(past)).days)


def in_fomc_window(date: datetime.date, fomc_df: pd.DataFrame, window: int = 3) -> bool:
    """True if `date` is within `window` calendar days of any FOMC date."""
    to_next = days_to_next_fomc(date, fomc_df)
    since_last = days_since_last_fomc(date, fomc_df)
    if to_next is not None and to_next <= window:
        return True
    if since_last is not None and since_last <= window:
        return True
    return False


if __name__ == "__main__":
    df = get_fomc_dates(force=True)
    print(f"FOMC dates: {len(df)} meetings")
    print(df.head(10))
    print("...")
    print(df.tail(10))
