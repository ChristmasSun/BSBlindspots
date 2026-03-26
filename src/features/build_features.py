import math
from pathlib import Path
from datetime import datetime

import pandas as pd

from src.data.fetch_stocks import fetch_ticker
from src.data.fetch_options import fetch_current_chains, fetch_historical_options
from src.data.fetch_rates import load_treasury_rates
from src.data.fetch_events import (
    fetch_earnings_dates,
    fetch_vix,
    get_days_to_next_earnings,
    get_days_since_last_earnings,
    get_vix_regime,
)
from src.pricing.black_scholes import (
    bs_price,
    abs_error,
    signed_error,
    rel_error,
    sq_error,
)
from src.pricing.volatility import compute_vol_features, compute_vol_of_vol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

EARNINGS_WINDOW_DAYS = 5

VIX_REGIME_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def _load_options(ticker: str) -> pd.DataFrame:
    try:
        df = fetch_current_chains(ticker)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    try:
        df = fetch_historical_options(ticker)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    raise ValueError(f"No options data available for {ticker}")


def _build_stock_lookup(stock_df: pd.DataFrame) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for idx in stock_df.index:
        date_key = pd.Timestamp(idx).strftime("%Y-%m-%d")
        row = stock_df.loc[idx]
        lookup[date_key] = {
            "close": float(row["Close"]),
            "hist_vol_20": float(row["hist_vol_20"]) if not pd.isna(row["hist_vol_20"]) else None,
            "hist_vol_60": float(row["hist_vol_60"]) if not pd.isna(row["hist_vol_60"]) else None,
            "vol_ratio": float(row["vol_ratio"]) if not pd.isna(row["vol_ratio"]) else None,
        }
    return lookup


def _build_vix_lookup(vix_df: pd.DataFrame, vol_of_vol: pd.Series) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for i in range(len(vix_df)):
        row = vix_df.iloc[i]
        date_key = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
        vix_close = float(row["Close"]) if not pd.isna(row["Close"]) else None
        vov = float(vol_of_vol.iloc[i]) if not pd.isna(vol_of_vol.iloc[i]) else None
        lookup[date_key] = {
            "vix": vix_close,
            "vol_of_vol": vov,
        }
    return lookup


def _build_rate_lookup(rates_df: pd.DataFrame) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for idx in rates_df.index:
        date_key = pd.Timestamp(idx).strftime("%Y-%m-%d")
        lookup[date_key] = float(rates_df.loc[idx, "risk_free_rate"])
    return lookup


def _get_nearest_rate(date_str: str, rate_lookup: dict[str, float], rates_df: pd.DataFrame) -> float:
    if date_str in rate_lookup:
        return rate_lookup[date_str]

    target = pd.Timestamp(date_str)
    mask = rates_df.index <= target
    if not mask.any():
        return float(rates_df.iloc[0]["risk_free_rate"])
    return float(rates_df.loc[mask].iloc[-1]["risk_free_rate"])


def _get_nearest_stock(date_str: str, stock_lookup: dict[str, dict], stock_df: pd.DataFrame) -> dict | None:
    if date_str in stock_lookup:
        return stock_lookup[date_str]

    target = pd.Timestamp(date_str)
    mask = stock_df.index <= target
    if not mask.any():
        return None
    nearest_idx = stock_df.loc[mask].index[-1]
    nearest_key = pd.Timestamp(nearest_idx).strftime("%Y-%m-%d")
    if nearest_key in stock_lookup:
        return stock_lookup[nearest_key]
    return None


def _get_nearest_vix(date_str: str, vix_lookup: dict[str, dict], vix_df: pd.DataFrame) -> dict | None:
    if date_str in vix_lookup:
        return vix_lookup[date_str]

    target = pd.Timestamp(date_str)
    vix_dates = pd.to_datetime(vix_df["Date"])
    mask = vix_dates <= target
    if not mask.any():
        return None
    nearest_idx = mask[mask].index[-1]
    nearest_key = pd.Timestamp(vix_df.iloc[nearest_idx]["Date"]).strftime("%Y-%m-%d")
    if nearest_key in vix_lookup:
        return vix_lookup[nearest_key]
    return None


def build_feature_matrix(ticker: str) -> pd.DataFrame:
    print(f"Building feature matrix for {ticker}...")

    print("  Loading options data...")
    options_df = _load_options(ticker)
    print(f"  {len(options_df)} option contracts loaded")

    print("  Loading stock prices and computing vol features...")
    stock_df = fetch_ticker(ticker)
    stock_df = compute_vol_features(stock_df)
    stock_lookup = _build_stock_lookup(stock_df)

    print("  Loading VIX data...")
    vix_df = fetch_vix()
    vol_of_vol = compute_vol_of_vol(vix_df)
    vix_lookup = _build_vix_lookup(vix_df, vol_of_vol)

    print("  Loading earnings dates...")
    earnings_df = fetch_earnings_dates(ticker)

    print("  Loading risk-free rates...")
    rates_df = load_treasury_rates()
    rate_lookup = _build_rate_lookup(rates_df)

    rows: list[dict] = []
    skipped = 0

    for i in range(len(options_df)):
        opt_row = options_df.iloc[i]

        date_str = str(opt_row["date"])
        strike = float(opt_row["strike"])
        option_type = str(opt_row["option_type"])
        expiration = str(opt_row["expiration"])
        bid = float(opt_row["bid"])
        ask = float(opt_row["ask"])
        mid_price = float(opt_row["mid_price"])
        volume = float(opt_row["volume"])
        open_interest = float(opt_row["open_interest"])

        stock_data = _get_nearest_stock(date_str, stock_lookup, stock_df)
        if stock_data is None:
            skipped += 1
            continue

        S = stock_data["close"]
        hist_vol_20 = stock_data["hist_vol_20"]
        hist_vol_60 = stock_data["hist_vol_60"]
        vol_ratio = stock_data["vol_ratio"]

        if hist_vol_20 is None or hist_vol_20 <= 0:
            skipped += 1
            continue

        if strike <= 0 or S <= 0:
            skipped += 1
            continue

        moneyness = S / strike
        log_moneyness = math.log(S / strike)

        trade_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        exp_date = datetime.strptime(expiration[:10], "%Y-%m-%d").date()
        dte = (exp_date - trade_date).days
        if dte <= 0:
            skipped += 1
            continue
        time_to_maturity = dte / 252.0

        option_type_binary = 1 if option_type == "call" else 0

        vix_data = _get_nearest_vix(date_str, vix_lookup, vix_df)
        vix_close = None
        vov_value = None
        vix_regime_encoded = None
        if vix_data is not None:
            vix_close = vix_data["vix"]
            vov_value = vix_data["vol_of_vol"]
            if vix_close is not None:
                regime_str = get_vix_regime(vix_close)
                vix_regime_encoded = VIX_REGIME_MAP[regime_str]

        days_to_earn = get_days_to_next_earnings(trade_date, ticker, earnings_df)
        days_since_earn = get_days_since_last_earnings(trade_date, ticker, earnings_df)

        in_earnings_window = 0
        if days_to_earn is not None and days_to_earn <= EARNINGS_WINDOW_DAYS:
            in_earnings_window = 1
        if days_since_earn is not None and days_since_earn <= EARNINGS_WINDOW_DAYS:
            in_earnings_window = 1

        bid_ask_spread = ask - bid
        mid_for_rel = mid_price if mid_price > 0 else 1.0
        bid_ask_rel = bid_ask_spread / mid_for_rel
        log_volume = math.log(volume + 1)
        log_open_interest = math.log(open_interest + 1)

        r = _get_nearest_rate(date_str, rate_lookup, rates_df)

        bs_theo = bs_price(S, strike, r, hist_vol_20, time_to_maturity, option_type)

        abs_err = abs_error(mid_price, bs_theo)
        signed_err = signed_error(mid_price, bs_theo)
        rel_err = rel_error(mid_price, bs_theo)
        sq_err = sq_error(mid_price, bs_theo)

        feature_row = {
            "date": date_str,
            "ticker": ticker,
            "expiration": expiration,
            "strike": strike,
            "option_type": option_type,
            "option_type_binary": option_type_binary,
            "underlying_price": S,
            "mid_price": mid_price,
            "moneyness": moneyness,
            "log_moneyness": log_moneyness,
            "dte": dte,
            "time_to_maturity": time_to_maturity,
            "hist_vol_20": hist_vol_20,
            "hist_vol_60": hist_vol_60,
            "vol_ratio": vol_ratio,
            "vix": vix_close,
            "vol_of_vol": vov_value,
            "days_to_earnings": days_to_earn,
            "days_since_earnings": days_since_earn,
            "in_earnings_window": in_earnings_window,
            "bid_ask_spread": bid_ask_spread,
            "bid_ask_rel": bid_ask_rel,
            "log_volume": log_volume,
            "log_open_interest": log_open_interest,
            "vix_regime": vix_regime_encoded,
            "bs_price": bs_theo,
            "abs_error": abs_err,
            "signed_error": signed_err,
            "rel_error": rel_err,
            "sq_error": sq_err,
        }
        rows.append(feature_row)

    print(f"  Built {len(rows)} feature rows, skipped {skipped}")

    result_df = pd.DataFrame(rows)

    critical_cols = [
        "moneyness",
        "dte",
        "hist_vol_20",
        "bs_price",
        "rel_error",
    ]
    before_drop = len(result_df)
    result_df = result_df.dropna(subset=critical_cols)
    after_drop = len(result_df)
    if before_drop != after_drop:
        print(f"  Dropped {before_drop - after_drop} rows with NaN in critical columns")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FEATURES_DIR / f"{ticker}_features.csv"
    result_df.to_csv(cache_path, index=False)
    print(f"  Cached to {cache_path}")
    print(f"  Final feature matrix: {result_df.shape[0]} rows x {result_df.shape[1]} columns")

    return result_df


if __name__ == "__main__":
    for t in ["SPY", "AAPL"]:
        df = build_feature_matrix(t)
        print(f"\n{t} summary:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        if not df.empty:
            print(f"  rel_error mean: {df['rel_error'].mean():.4f}")
            print(f"  rel_error std:  {df['rel_error'].std():.4f}")
            print(f"  moneyness range: [{df['moneyness'].min():.4f}, {df['moneyness'].max():.4f}]")
            print(f"  dte range: [{df['dte'].min()}, {df['dte'].max()}]")
        print()
