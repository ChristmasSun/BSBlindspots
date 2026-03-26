import os
import time
import math
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "options"

COLUMNS = [
    "date",
    "ticker",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid_price",
    "volume",
    "open_interest",
    "underlying_price",
]

MIN_DTE = 7
MAX_DTE = 180
MIN_MONEYNESS = 0.8
MAX_MONEYNESS = 1.2
MIN_MID_PRICE = 0.10
MIN_OI_FALLBACK = 100


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(ticker: str, suffix: str) -> Path:
    return RAW_DIR / f"{ticker}_{suffix}.csv"


def _load_cache(ticker: str, suffix: str) -> pd.DataFrame | None:
    path = _cache_path(ticker, suffix)
    if path.exists():
        return pd.read_csv(path)
    return None


def _save_cache(df: pd.DataFrame, ticker: str, suffix: str) -> None:
    _ensure_dirs()
    path = _cache_path(ticker, suffix)
    df.to_csv(path, index=False)


def _passes_filters(
    row_bid: float,
    row_ask: float,
    row_volume: float,
    row_oi: float,
    strike: float,
    underlying_price: float,
    dte: int,
) -> bool:
    mid = (row_bid + row_ask) / 2.0
    if mid <= MIN_MID_PRICE:
        return False
    if dte < MIN_DTE or dte > MAX_DTE:
        return False
    if strike <= 0:
        return False
    moneyness = underlying_price / strike
    if moneyness < MIN_MONEYNESS or moneyness > MAX_MONEYNESS:
        return False
    vol = row_volume if not math.isnan(row_volume) else 0
    oi = row_oi if not math.isnan(row_oi) else 0
    if vol <= 0 and oi < MIN_OI_FALLBACK:
        return False
    return True


def fetch_current_chains(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    if use_cache:
        cached = _load_cache(ticker, "current")
        if cached is not None:
            return cached

    stock = yf.Ticker(ticker)
    underlying_price = stock.fast_info.get("lastPrice", None)
    if underlying_price is None:
        hist = stock.history(period="1d")
        underlying_price = float(hist["Close"].iloc[-1])

    expirations = stock.options
    today = datetime.now().date()

    rows = []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte < MIN_DTE or dte > MAX_DTE:
            continue

        chain = stock.option_chain(exp_str)

        for opt_type, df_chain in [("call", chain.calls), ("put", chain.puts)]:
            for idx in range(len(df_chain)):
                row = df_chain.iloc[idx]
                bid = float(row.get("bid", 0))
                ask = float(row.get("ask", 0))
                volume = float(row.get("volume", 0)) if not pd.isna(row.get("volume")) else 0
                oi = float(row.get("openInterest", 0)) if not pd.isna(row.get("openInterest")) else 0
                strike = float(row["strike"])

                if not _passes_filters(bid, ask, volume, oi, strike, underlying_price, dte):
                    continue

                mid = (bid + ask) / 2.0
                rows.append({
                    "date": today.isoformat(),
                    "ticker": ticker,
                    "expiration": exp_str,
                    "strike": strike,
                    "option_type": opt_type,
                    "bid": bid,
                    "ask": ask,
                    "mid_price": round(mid, 4),
                    "volume": int(volume),
                    "open_interest": int(oi),
                    "underlying_price": underlying_price,
                })

    result = pd.DataFrame(rows, columns=COLUMNS)
    _save_cache(result, ticker, "current")
    return result


def fetch_historical_options(
    ticker: str,
    start_date: str = "",
    end_date: str = "",
    use_cache: bool = True,
) -> pd.DataFrame:
    if use_cache:
        cached = _load_cache(ticker, "historical")
        if cached is not None:
            return cached

    from polygon import RESTClient

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY not found in .env")

    client = RESTClient(api_key)

    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    stock_aggs = []
    for agg in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=start_date,
        to=end_date,
        limit=50000,
    ):
        stock_aggs.append(agg)
    time.sleep(12)

    price_map: dict[str, float] = {}
    for agg in stock_aggs:
        dt = datetime.fromtimestamp(agg.timestamp / 1000).strftime("%Y-%m-%d")
        price_map[dt] = agg.close

    contracts = []
    for contract in client.list_options_contracts(
        underlying_ticker=ticker,
        expired=True,
        limit=1000,
    ):
        contracts.append(contract)
    time.sleep(12)

    rows = []
    request_count = 0

    for contract in contracts:
        contract_ticker = contract.ticker
        strike = contract.strike_price
        exp_str = contract.expiration_date
        opt_type = "call" if contract.contract_type == "call" else "put"

        try:
            opt_aggs = []
            for agg in client.list_aggs(
                ticker=contract_ticker,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date,
                limit=50000,
            ):
                opt_aggs.append(agg)
        except Exception:
            continue

        request_count += 1
        if request_count % 4 == 0:
            time.sleep(12)

        for agg in opt_aggs:
            trade_date = datetime.fromtimestamp(agg.timestamp / 1000).strftime("%Y-%m-%d")
            underlying_price = price_map.get(trade_date)
            if underlying_price is None:
                continue

            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            trade_dt = datetime.strptime(trade_date, "%Y-%m-%d").date()
            dte = (exp_date - trade_dt).days

            bid = agg.low if agg.low else 0.0
            ask = agg.high if agg.high else 0.0
            volume = agg.volume if agg.volume else 0
            oi = 0

            if not _passes_filters(bid, ask, volume, oi, strike, underlying_price, dte):
                continue

            mid = (bid + ask) / 2.0
            rows.append({
                "date": trade_date,
                "ticker": ticker,
                "expiration": exp_str,
                "strike": strike,
                "option_type": opt_type,
                "bid": bid,
                "ask": ask,
                "mid_price": round(mid, 4),
                "volume": int(volume),
                "open_interest": int(oi),
                "underlying_price": underlying_price,
            })

    result = pd.DataFrame(rows, columns=COLUMNS)
    _save_cache(result, ticker, "historical")
    return result


def main() -> None:
    tickers = ["SPY", "AAPL"]

    for ticker in tickers:
        print(f"Fetching current options chains for {ticker}...")
        current_df = fetch_current_chains(ticker)
        print(f"  {len(current_df)} contracts fetched")
        print(f"  Cached to {_cache_path(ticker, 'current')}")

    polygon_key = os.getenv("POLYGON_API_KEY")
    if polygon_key:
        for ticker in tickers:
            print(f"Fetching historical options for {ticker} via Polygon...")
            hist_df = fetch_historical_options(ticker)
            print(f"  {len(hist_df)} records fetched")
            print(f"  Cached to {_cache_path(ticker, 'historical')}")
    else:
        print("POLYGON_API_KEY not set, skipping historical fetch")


if __name__ == "__main__":
    main()
