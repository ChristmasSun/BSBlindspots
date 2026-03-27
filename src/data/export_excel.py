from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPORT_DIR = DATA_DIR / "exports"
WORKBOOK_PATH = EXPORT_DIR / "bs_blindspots_data.xlsx"
ACTUAL_DATA_WORKBOOK_PATH = EXPORT_DIR / "bs_blindspots_actual_data.xlsx"
MAX_EXCEL_ROWS = 1_000_000


def _combine_csvs(paths: list[Path], ticker_from_filename: str | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in paths:
        df = pd.read_csv(path)
        if ticker_from_filename is not None:
            df.insert(0, ticker_from_filename, path.stem.replace("_daily", ""))
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_universe_sheet() -> pd.DataFrame:
    path = RAW_DIR / "universe" / "large_universe.json"
    if not path.exists():
        return pd.DataFrame()

    payload = json.loads(path.read_text())
    rows: list[dict[str, str]] = []

    for group_name in ["stock_tickers", "etf_tickers", "index_tickers", "option_tickers"]:
        tickers = payload.get(group_name, [])
        for ticker in tickers:
            rows.append({"group": group_name, "ticker": ticker})

    return pd.DataFrame(rows)


def load_summary_sheet() -> pd.DataFrame:
    rows = []
    rows.append({"dataset": "generated_at", "value": datetime.now().isoformat()})

    paths = {
        "raw_stock_files": list(RAW_DIR.glob("*_daily.csv")),
        "raw_option_current_files": list((RAW_DIR / "options").glob("*_current.csv")),
        "raw_option_historical_files": list((RAW_DIR / "options").glob("*_historical.csv")),
        "earnings_files": list((RAW_DIR / "earnings").glob("*_earnings.csv")),
        "polygon_daily_files": list((RAW_DIR / "polygon_daily").glob("*_polygon_daily.csv")),
        "processed_files": list(PROCESSED_DIR.glob("*_pricing_features.csv")),
        "feature_files": list((DATA_DIR / "features").glob("*_features.csv")),
    }

    for name, files in paths.items():
        rows.append({"dataset": name, "value": len(files)})

    return pd.DataFrame(rows)


def load_raw_stocks_sheet() -> pd.DataFrame:
    paths = sorted(RAW_DIR.glob("*_daily.csv"))
    return _combine_csvs(paths, ticker_from_filename="ticker")


def load_raw_options_current_sheet() -> pd.DataFrame:
    paths = sorted((RAW_DIR / "options").glob("*_current.csv"))
    return _combine_csvs(paths)


def load_raw_options_historical_sheet() -> pd.DataFrame:
    paths = sorted((RAW_DIR / "options").glob("*_historical.csv"))
    return _combine_csvs(paths)


def load_earnings_sheet() -> pd.DataFrame:
    paths = sorted((RAW_DIR / "earnings").glob("*_earnings.csv"))
    return _combine_csvs(paths)


def load_processed_sheet() -> pd.DataFrame:
    paths = sorted(PROCESSED_DIR.glob("*_pricing_features.csv"))
    return _combine_csvs(paths)


def load_polygon_daily_sheet() -> pd.DataFrame:
    paths = sorted((RAW_DIR / "polygon_daily").glob("*_polygon_daily.csv"))
    return _combine_csvs(paths)


def load_rates_sheet() -> pd.DataFrame:
    path = RAW_DIR / "treasury_3mo.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_vix_sheet() -> pd.DataFrame:
    path = RAW_DIR / "vix.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_dataframe(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    if df.empty:
        pd.DataFrame({"note": ["no data"]}).to_excel(writer, sheet_name=sheet_name, index=False)
        return

    if len(df) <= MAX_EXCEL_ROWS:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        return

    chunk_index = 1
    start = 0
    while start < len(df):
        end = min(start + MAX_EXCEL_ROWS, len(df))
        chunk = df.iloc[start:end].copy()
        chunk.to_excel(writer, sheet_name=f"{sheet_name}_{chunk_index}", index=False)
        start = end
        chunk_index += 1


def export_workbook(output_path: Path = WORKBOOK_PATH) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    sheets = [
        ("processed", load_processed_sheet()),
        ("raw_options_current", load_raw_options_current_sheet()),
        ("raw_stocks", load_raw_stocks_sheet()),
        ("earnings", load_earnings_sheet()),
        ("polygon_daily", load_polygon_daily_sheet()),
        ("rates", load_rates_sheet()),
        ("vix", load_vix_sheet()),
        ("raw_options_historical", load_raw_options_historical_sheet()),
        ("universe", load_universe_sheet()),
        ("summary", load_summary_sheet()),
    ]

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets:
            _write_dataframe(writer, sheet_name, df)

    return output_path


def export_actual_data_workbook(output_path: Path = ACTUAL_DATA_WORKBOOK_PATH) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    sheets = [
        ("processed", load_processed_sheet()),
        ("raw_options_current", load_raw_options_current_sheet()),
        ("raw_stocks", load_raw_stocks_sheet()),
        ("earnings", load_earnings_sheet()),
        ("polygon_daily", load_polygon_daily_sheet()),
        ("rates", load_rates_sheet()),
        ("vix", load_vix_sheet()),
        ("raw_options_historical", load_raw_options_historical_sheet()),
    ]

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets:
            _write_dataframe(writer, sheet_name, df)

    return output_path


if __name__ == "__main__":
    workbook = export_workbook()
    actual_workbook = export_actual_data_workbook()
    print(workbook)
    print(actual_workbook)
