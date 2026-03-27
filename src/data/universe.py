import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_DIR = PROJECT_ROOT / "data" / "raw" / "universe"
LARGE_UNIVERSE_PATH = UNIVERSE_DIR / "large_universe.json"

US_STOCKS_URL_TEMPLATE = (
    "https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/?page={page}"
)
US_STOCKS_PAGE_ONE_URL = "https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/"
US_ETFS_URL = "https://companiesmarketcap.com/usa-etfs/largest-us-etfs-by-marketcap/"

MAJOR_INDEX_TICKERS = [
    "^GSPC",
    "^IXIC",
    "^DJI",
    "^NDX",
    "^RUT",
    "^VIX",
    "^SOX",
    "^NYA",
    "^XAX",
    "^W5000",
]

YAHOO_TICKER_OVERRIDES = {"BRK.B": "BRK-B", "BF.B": "BF-B"}


def _request_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def _extract_tickers_from_html(html: str, limit: int) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    tickers: list[str] = []
    seen: set[str] = set()

    rows = soup.select("table tbody tr")
    for row in rows:
        code_div = row.select_one("div.company-code")
        if code_div is None:
            continue
        ticker = code_div.get_text(strip=True)
        if ticker in YAHOO_TICKER_OVERRIDES:
            ticker = YAHOO_TICKER_OVERRIDES[ticker]
        if ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
        if len(tickers) >= limit:
            break

    return tickers


def fetch_top_us_stocks(limit: int = 200) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    page = 1

    while len(tickers) < limit:
        if page == 1:
            url = US_STOCKS_PAGE_ONE_URL
        else:
            url = US_STOCKS_URL_TEMPLATE.format(page=page)

        html = _request_html(url)
        page_tickers = _extract_tickers_from_html(html, limit=100)
        if not page_tickers:
            break

        for ticker in page_tickers:
            if ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)
            if len(tickers) >= limit:
                break

        page += 1

    return tickers[:limit]


def fetch_top_us_etfs(limit: int = 20) -> list[str]:
    html = _request_html(US_ETFS_URL)
    raw_tickers = _extract_tickers_from_html(html, limit=limit * 3)
    filtered_tickers: list[str] = []
    for ticker in raw_tickers:
        if "." in ticker:
            continue
        filtered_tickers.append(ticker)
        if len(filtered_tickers) >= limit:
            break
    return filtered_tickers


def build_large_universe(
    top_stock_count: int = 200,
    top_etf_count: int = 20,
) -> dict[str, list[str] | str | dict[str, str]]:
    stock_tickers = fetch_top_us_stocks(limit=top_stock_count)
    etf_tickers = fetch_top_us_etfs(limit=top_etf_count)

    option_tickers: list[str] = []
    seen: set[str] = set()
    for ticker in stock_tickers + etf_tickers:
        if ticker in seen:
            continue
        seen.add(ticker)
        option_tickers.append(ticker)

    underlying_tickers: list[str] = []
    seen_underlyings: set[str] = set()
    for ticker in option_tickers + MAJOR_INDEX_TICKERS:
        if ticker in seen_underlyings:
            continue
        seen_underlyings.add(ticker)
        underlying_tickers.append(ticker)

    universe = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sources": {
            "stocks": US_STOCKS_PAGE_ONE_URL,
            "etfs": US_ETFS_URL,
        },
        "stock_tickers": stock_tickers,
        "etf_tickers": etf_tickers,
        "index_tickers": MAJOR_INDEX_TICKERS,
        "option_tickers": option_tickers,
        "underlying_tickers": underlying_tickers,
    }
    return universe


def save_large_universe(universe: dict[str, list[str] | str | dict[str, str]]) -> None:
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LARGE_UNIVERSE_PATH, "w", encoding="utf-8") as handle:
        json.dump(universe, handle, indent=2)


def load_large_universe() -> dict[str, list[str] | str | dict[str, str]]:
    with open(LARGE_UNIVERSE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_and_save_large_universe(
    top_stock_count: int = 200,
    top_etf_count: int = 20,
) -> dict[str, list[str] | str | dict[str, str]]:
    universe = build_large_universe(
        top_stock_count=top_stock_count,
        top_etf_count=top_etf_count,
    )
    save_large_universe(universe)
    return universe


if __name__ == "__main__":
    universe = build_and_save_large_universe()
    print(f"stock_tickers={len(universe['stock_tickers'])}")
    print(f"etf_tickers={len(universe['etf_tickers'])}")
    print(f"index_tickers={len(universe['index_tickers'])}")
    print(f"option_tickers={len(universe['option_tickers'])}")
    print(f"underlying_tickers={len(universe['underlying_tickers'])}")
