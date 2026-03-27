"""
CPI (Consumer Price Index) release dates.

Source: Bureau of Labor Statistics release schedule.
These are the dates when the monthly CPI report is published,
typically around the 10th-14th of each month for the prior month's data.

Coverage: 2020-01-01 through 2026-12-31.
"""

import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "events"

CPI_DATES: list[str] = [
    # 2020
    "2020-01-14",
    "2020-02-13",
    "2020-03-11",
    "2020-04-10",
    "2020-05-12",
    "2020-06-10",
    "2020-07-14",
    "2020-08-12",
    "2020-09-11",
    "2020-10-13",
    "2020-11-12",
    "2020-12-10",
    # 2021
    "2021-01-13",
    "2021-02-10",
    "2021-03-10",
    "2021-04-13",
    "2021-05-12",
    "2021-06-10",
    "2021-07-13",
    "2021-08-11",
    "2021-09-14",
    "2021-10-13",
    "2021-11-10",
    "2021-12-10",
    # 2022
    "2022-01-12",
    "2022-02-10",
    "2022-03-10",
    "2022-04-12",
    "2022-05-11",
    "2022-06-10",
    "2022-07-13",
    "2022-08-10",
    "2022-09-13",
    "2022-10-13",
    "2022-11-10",
    "2022-12-13",
    # 2023
    "2023-01-12",
    "2023-02-14",
    "2023-03-14",
    "2023-04-12",
    "2023-05-10",
    "2023-06-13",
    "2023-07-12",
    "2023-08-10",
    "2023-09-13",
    "2023-10-12",
    "2023-11-14",
    "2023-12-12",
    # 2024
    "2024-01-11",
    "2024-02-13",
    "2024-03-12",
    "2024-04-10",
    "2024-05-15",
    "2024-06-12",
    "2024-07-11",
    "2024-08-14",
    "2024-09-11",
    "2024-10-10",
    "2024-11-13",
    "2024-12-11",
    # 2025
    "2025-01-15",
    "2025-02-12",
    "2025-03-12",
    "2025-04-10",
    "2025-05-13",
    "2025-06-11",
    "2025-07-15",
    "2025-08-12",
    "2025-09-10",
    "2025-10-14",
    "2025-11-12",
    "2025-12-10",
    # 2026 (projected — BLS typically publishes ~12 months ahead)
    "2026-01-14",
    "2026-02-11",
    "2026-03-11",
    "2026-04-14",
    "2026-05-12",
    "2026-06-10",
    "2026-07-14",
    "2026-08-12",
    "2026-09-11",
    "2026-10-13",
    "2026-11-12",
    "2026-12-09",
]


def get_cpi_dates(
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of CPI release dates.

    Args:
        start: Filter to dates on or after this (YYYY-MM-DD). None = no lower bound.
        end:   Filter to dates on or before this (YYYY-MM-DD). None = no upper bound.
        force: If True, regenerate even if cached file exists.

    Returns:
        DataFrame with columns: [date, event_type]
        date is datetime64, event_type is always "cpi".
    """
    cache_path = RAW_DIR / "cpi_dates.csv"

    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        for d in CPI_DATES:
            rows.append({"date": pd.Timestamp(d), "event_type": "cpi"})
        df = pd.DataFrame(rows)
        df.to_csv(cache_path, index=False)

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    return df.reset_index(drop=True)


def days_to_next_cpi(date: datetime.date, cpi_df: pd.DataFrame) -> int | None:
    """Calendar days from `date` to the next CPI release."""
    target = pd.Timestamp(date).tz_localize(None)
    future = []
    for _, row in cpi_df.iterrows():
        dt = pd.Timestamp(row["date"]).tz_localize(None)
        if dt >= target:
            future.append(dt)
    if not future:
        return None
    return int((min(future) - target).days)


def days_since_last_cpi(date: datetime.date, cpi_df: pd.DataFrame) -> int | None:
    """Calendar days from the most recent CPI release to `date`."""
    target = pd.Timestamp(date).tz_localize(None)
    past = []
    for _, row in cpi_df.iterrows():
        dt = pd.Timestamp(row["date"]).tz_localize(None)
        if dt <= target:
            past.append(dt)
    if not past:
        return None
    return int((target - max(past)).days)


def in_cpi_window(date: datetime.date, cpi_df: pd.DataFrame, window: int = 2) -> bool:
    """True if `date` is within `window` calendar days of any CPI release."""
    to_next = days_to_next_cpi(date, cpi_df)
    since_last = days_since_last_cpi(date, cpi_df)
    if to_next is not None and to_next <= window:
        return True
    if since_last is not None and since_last <= window:
        return True
    return False


if __name__ == "__main__":
    df = get_cpi_dates(force=True)
    print(f"CPI release dates: {len(df)} releases")
    print(df.head(10))
    print("...")
    print(df.tail(10))
