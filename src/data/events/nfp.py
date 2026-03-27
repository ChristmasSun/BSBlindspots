"""
NFP (Non-Farm Payrolls) / Jobs Report release dates.

Source: Bureau of Labor Statistics release schedule.
The Employment Situation report is typically released on the first Friday
of each month at 8:30 AM ET. Exceptions occur when the first Friday falls
on a holiday or when BLS shifts the schedule.

Coverage: 2020-01-01 through 2026-12-31.
"""

import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "events"

NFP_DATES: list[str] = [
    # 2020
    "2020-01-10",
    "2020-02-07",
    "2020-03-06",
    "2020-04-03",
    "2020-05-08",
    "2020-06-05",
    "2020-07-02",
    "2020-08-07",
    "2020-09-04",
    "2020-10-02",
    "2020-11-06",
    "2020-12-04",
    # 2021
    "2021-01-08",
    "2021-02-05",
    "2021-03-05",
    "2021-04-02",
    "2021-05-07",
    "2021-06-04",
    "2021-07-02",
    "2021-08-06",
    "2021-09-03",
    "2021-10-08",
    "2021-11-05",
    "2021-12-03",
    # 2022
    "2022-01-07",
    "2022-02-04",
    "2022-03-04",
    "2022-04-01",
    "2022-05-06",
    "2022-06-03",
    "2022-07-08",
    "2022-08-05",
    "2022-09-02",
    "2022-10-07",
    "2022-11-04",
    "2022-12-02",
    # 2023
    "2023-01-06",
    "2023-02-03",
    "2023-03-10",
    "2023-04-07",
    "2023-05-05",
    "2023-06-02",
    "2023-07-07",
    "2023-08-04",
    "2023-09-01",
    "2023-10-06",
    "2023-11-03",
    "2023-12-08",
    # 2024
    "2024-01-05",
    "2024-02-02",
    "2024-03-08",
    "2024-04-05",
    "2024-05-03",
    "2024-06-07",
    "2024-07-05",
    "2024-08-02",
    "2024-09-06",
    "2024-10-04",
    "2024-11-01",
    "2024-12-06",
    # 2025
    "2025-01-10",
    "2025-02-07",
    "2025-03-07",
    "2025-04-04",
    "2025-05-02",
    "2025-06-06",
    "2025-07-03",
    "2025-08-01",
    "2025-09-05",
    "2025-10-03",
    "2025-11-07",
    "2025-12-05",
    # 2026 (projected — first Friday rule with typical BLS adjustments)
    "2026-01-09",
    "2026-02-06",
    "2026-03-06",
    "2026-04-03",
    "2026-05-08",
    "2026-06-05",
    "2026-07-02",
    "2026-08-07",
    "2026-09-04",
    "2026-10-02",
    "2026-11-06",
    "2026-12-04",
]


def get_nfp_dates(
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of NFP / Jobs Report release dates.

    Args:
        start: Filter to dates on or after this (YYYY-MM-DD). None = no lower bound.
        end:   Filter to dates on or before this (YYYY-MM-DD). None = no upper bound.
        force: If True, regenerate even if cached file exists.

    Returns:
        DataFrame with columns: [date, event_type]
        date is datetime64, event_type is always "nfp".
    """
    cache_path = RAW_DIR / "nfp_dates.csv"

    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        for d in NFP_DATES:
            rows.append({"date": pd.Timestamp(d), "event_type": "nfp"})
        df = pd.DataFrame(rows)
        df.to_csv(cache_path, index=False)

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    return df.reset_index(drop=True)


def days_to_next_nfp(date: datetime.date, nfp_df: pd.DataFrame) -> int | None:
    """Calendar days from `date` to the next NFP release."""
    target = pd.Timestamp(date).tz_localize(None)
    future = []
    for _, row in nfp_df.iterrows():
        dt = pd.Timestamp(row["date"]).tz_localize(None)
        if dt >= target:
            future.append(dt)
    if not future:
        return None
    return int((min(future) - target).days)


def days_since_last_nfp(date: datetime.date, nfp_df: pd.DataFrame) -> int | None:
    """Calendar days from the most recent NFP release to `date`."""
    target = pd.Timestamp(date).tz_localize(None)
    past = []
    for _, row in nfp_df.iterrows():
        dt = pd.Timestamp(row["date"]).tz_localize(None)
        if dt <= target:
            past.append(dt)
    if not past:
        return None
    return int((target - max(past)).days)


def in_nfp_window(date: datetime.date, nfp_df: pd.DataFrame, window: int = 2) -> bool:
    """True if `date` is within `window` calendar days of any NFP release."""
    to_next = days_to_next_nfp(date, nfp_df)
    since_last = days_since_last_nfp(date, nfp_df)
    if to_next is not None and to_next <= window:
        return True
    if since_last is not None and since_last <= window:
        return True
    return False


if __name__ == "__main__":
    df = get_nfp_dates(force=True)
    print(f"NFP release dates: {len(df)} releases")
    print(df.head(10))
    print("...")
    print(df.tail(10))
