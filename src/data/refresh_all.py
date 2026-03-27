import argparse
import os
import time
from datetime import datetime, timedelta

from src.data.fetch_events import fetch_earnings_dates, fetch_vix
from src.data.fetch_options import fetch_current_chains, fetch_historical_options
from src.data.fetch_rates import load_treasury_rates
from src.data.fetch_stocks import fetch_ticker
from src.data.universe import build_and_save_large_universe
from src.features.build_features import build_feature_matrix


DEFAULT_TICKERS = ["SPY", "AAPL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh raw and processed data for BS Blindspots.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Tickers to refresh.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache where supported and refetch data.",
    )
    parser.add_argument(
        "--polygon-history",
        action="store_true",
        help="Fetch a small recent Polygon historical options backfill.",
    )
    parser.add_argument(
        "--historical-days",
        type=int,
        default=14,
        help="Days of Polygon history to request when --polygon-history is used.",
    )
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=12,
        help="Maximum Polygon option contracts to backfill per ticker.",
    )
    parser.add_argument(
        "--large-universe",
        action="store_true",
        help="Resolve and refresh the large daily universe of stocks, ETFs, and indexes.",
    )
    parser.add_argument(
        "--top-stock-count",
        type=int,
        default=200,
        help="Number of large-cap stocks to include when --large-universe is used.",
    )
    parser.add_argument(
        "--top-etf-count",
        type=int,
        default=20,
        help="Number of large ETFs to include when --large-universe is used.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between option-chain requests.",
    )
    return parser.parse_args()


def _resolve_universe(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    if not args.large_universe:
        return args.tickers, args.tickers, []

    universe = build_and_save_large_universe(
        top_stock_count=args.top_stock_count,
        top_etf_count=args.top_etf_count,
    )
    underlying_tickers = list(universe["underlying_tickers"])
    option_tickers = list(universe["option_tickers"])
    index_tickers = list(universe["index_tickers"])
    return underlying_tickers, option_tickers, index_tickers


def main() -> None:
    args = parse_args()
    underlying_tickers, option_tickers, index_tickers = _resolve_universe(args)

    print(
        f"Refreshing universe with {len(underlying_tickers)} underlyings, "
        f"{len(option_tickers)} optionable tickers, and {len(index_tickers)} indexes..."
    )

    print("Refreshing stock price history...")
    for ticker in underlying_tickers:
        try:
            df = fetch_ticker(ticker, force=args.force)
            print(f"  {ticker}: {len(df)} rows")
        except Exception as exc:
            print(f"  {ticker}: stock fetch failed: {exc}")

    print("Refreshing macro and event inputs...")
    rates_df = load_treasury_rates(force_refresh=args.force)
    print(f"  rates: {len(rates_df)} rows")

    vix_df = fetch_vix(force=args.force)
    print(f"  vix: {len(vix_df)} rows")

    for ticker in option_tickers:
        try:
            earnings_df = fetch_earnings_dates(ticker, limit=25, force=args.force)
            print(f"  {ticker} earnings: {len(earnings_df)} rows")
        except Exception as exc:
            print(f"  {ticker} earnings fetch failed: {exc}")

    print("Refreshing current option chains...")
    for ticker in option_tickers:
        try:
            current_df = fetch_current_chains(ticker, use_cache=not args.force)
            print(f"  {ticker} current options: {len(current_df)} rows")
        except Exception as exc:
            print(f"  {ticker} current options failed: {exc}")
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if args.polygon_history:
        polygon_key = os.getenv("POLYGON_API_KEY")
        if polygon_key:
            print("Refreshing recent Polygon historical options...")
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=args.historical_days)).strftime(
                "%Y-%m-%d"
            )
            for ticker in option_tickers:
                try:
                    hist_df = fetch_historical_options(
                        ticker,
                        start_date=start_date,
                        end_date=end_date,
                        use_cache=not args.force,
                        max_contracts=args.max_contracts,
                    )
                    print(f"  {ticker} polygon historical: {len(hist_df)} rows")
                except Exception as exc:
                    print(f"  {ticker} polygon historical failed: {exc}")
        else:
            print("POLYGON_API_KEY not set, skipping Polygon historical refresh")

    print("Building processed feature datasets...")
    for ticker in option_tickers:
        try:
            features_df = build_feature_matrix(ticker)
            print(f"  {ticker} processed rows: {len(features_df)}")
        except Exception as exc:
            print(f"  {ticker} processed dataset failed: {exc}")


if __name__ == "__main__":
    main()
