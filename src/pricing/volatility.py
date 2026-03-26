import math
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

ANNUALIZATION_FACTOR = math.sqrt(252)


def rolling_historical_vol(log_returns: pd.Series, window: int = 20) -> pd.Series:
    rolling_std = log_returns.rolling(window=window).std()
    annualized = rolling_std * ANNUALIZATION_FACTOR
    return annualized


def compute_vol_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    df = prices_df.copy()

    if "log_return" not in df.columns:
        raise ValueError("DataFrame must contain a 'log_return' column")

    df["hist_vol_20"] = rolling_historical_vol(df["log_return"], window=20)
    df["hist_vol_60"] = rolling_historical_vol(df["log_return"], window=60)
    df["vol_ratio"] = df["hist_vol_20"] / df["hist_vol_60"]

    return df


def compute_vol_of_vol(vix_df: pd.DataFrame, window: int = 20) -> pd.Series:
    vix_changes = vix_df["Close"].diff()
    vol_of_vol = vix_changes.rolling(window=window).std()
    return vol_of_vol


if __name__ == "__main__":
    from src.data.fetch_stocks import fetch_ticker, DEFAULT_TICKERS

    for ticker in DEFAULT_TICKERS[:2]:
        cache_file = RAW_DIR / f"{ticker}_daily.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            print(f"No cached data for {ticker}, fetching...")
            df = fetch_ticker(ticker)

        df = compute_vol_features(df)
        print(f"\n{ticker} vol features:")
        print(f"  Rows: {len(df)}")
        print(f"  hist_vol_20 mean: {df['hist_vol_20'].mean():.4f}")
        print(f"  hist_vol_60 mean: {df['hist_vol_60'].mean():.4f}")
        print(f"  vol_ratio mean:   {df['vol_ratio'].mean():.4f}")
        print(f"  NaN counts:")
        print(f"    hist_vol_20: {df['hist_vol_20'].isna().sum()}")
        print(f"    hist_vol_60: {df['hist_vol_60'].isna().sum()}")
        print(df[["Close", "log_return", "hist_vol_20", "hist_vol_60", "vol_ratio"]].tail(10))

    vix_file = RAW_DIR / "vix.csv"
    if vix_file.exists():
        vix_df = pd.read_csv(vix_file, parse_dates=["Date"])
        vov = compute_vol_of_vol(vix_df)
        print(f"\nVol of Vol (VIX):")
        print(f"  Mean: {vov.mean():.4f}")
        print(f"  NaN count: {vov.isna().sum()}")
        print(vov.tail(10))
    else:
        print("\nNo cached VIX data found, skipping vol-of-vol check.")
