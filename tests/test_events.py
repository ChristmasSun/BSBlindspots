"""Tests for the events data pipeline (FOMC, CPI, NFP, earnings, unified)."""

import datetime

import pandas as pd
import pytest


class TestFOMC:
    def test_get_fomc_dates_returns_dataframe(self):
        from src.data.events.fomc import get_fomc_dates

        df = get_fomc_dates()
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "event_type" in df.columns

    def test_fomc_all_labeled_correctly(self):
        from src.data.events.fomc import get_fomc_dates

        df = get_fomc_dates()
        unique_types = df["event_type"].unique()
        assert len(unique_types) == 1
        assert unique_types[0] == "fomc"

    def test_fomc_no_duplicates(self):
        from src.data.events.fomc import get_fomc_dates

        df = get_fomc_dates()
        assert df["date"].is_unique

    def test_fomc_date_range_filter(self):
        from src.data.events.fomc import get_fomc_dates

        df = get_fomc_dates(start="2024-01-01", end="2024-12-31")
        assert len(df) == 8
        assert df["date"].min() >= pd.Timestamp("2024-01-01")
        assert df["date"].max() <= pd.Timestamp("2024-12-31")

    def test_fomc_known_date_present(self):
        from src.data.events.fomc import get_fomc_dates

        df = get_fomc_dates()
        dates_str = df["date"].dt.strftime("%Y-%m-%d").tolist()
        assert "2024-03-20" in dates_str

    def test_days_to_next_fomc(self):
        from src.data.events.fomc import get_fomc_dates, days_to_next_fomc

        df = get_fomc_dates()
        result = days_to_next_fomc(datetime.date(2024, 3, 19), df)
        assert result == 1

    def test_days_since_last_fomc(self):
        from src.data.events.fomc import get_fomc_dates, days_since_last_fomc

        df = get_fomc_dates()
        result = days_since_last_fomc(datetime.date(2024, 3, 21), df)
        assert result == 1

    def test_in_fomc_window_true(self):
        from src.data.events.fomc import get_fomc_dates, in_fomc_window

        df = get_fomc_dates()
        assert in_fomc_window(datetime.date(2024, 3, 19), df, window=3) is True

    def test_in_fomc_window_false(self):
        from src.data.events.fomc import get_fomc_dates, in_fomc_window

        df = get_fomc_dates()
        assert in_fomc_window(datetime.date(2024, 3, 10), df, window=3) is False

    def test_days_to_fomc_none_after_last(self):
        from src.data.events.fomc import get_fomc_dates, days_to_next_fomc

        df = get_fomc_dates()
        result = days_to_next_fomc(datetime.date(2030, 1, 1), df)
        assert result is None


class TestCPI:
    def test_get_cpi_dates_returns_dataframe(self):
        from src.data.events.cpi import get_cpi_dates

        df = get_cpi_dates()
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "event_type" in df.columns

    def test_cpi_12_per_year(self):
        from src.data.events.cpi import get_cpi_dates

        df = get_cpi_dates(start="2023-01-01", end="2023-12-31")
        assert len(df) == 12

    def test_cpi_no_duplicates(self):
        from src.data.events.cpi import get_cpi_dates

        df = get_cpi_dates()
        assert df["date"].is_unique

    def test_days_to_next_cpi(self):
        from src.data.events.cpi import get_cpi_dates, days_to_next_cpi

        df = get_cpi_dates()
        result = days_to_next_cpi(datetime.date(2024, 3, 12), df)
        assert result == 0

    def test_in_cpi_window_on_release_day(self):
        from src.data.events.cpi import get_cpi_dates, in_cpi_window

        df = get_cpi_dates()
        assert in_cpi_window(datetime.date(2024, 3, 12), df, window=2) is True


class TestNFP:
    def test_get_nfp_dates_returns_dataframe(self):
        from src.data.events.nfp import get_nfp_dates

        df = get_nfp_dates()
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "event_type" in df.columns

    def test_nfp_12_per_year(self):
        from src.data.events.nfp import get_nfp_dates

        df = get_nfp_dates(start="2023-01-01", end="2023-12-31")
        assert len(df) == 12

    def test_nfp_no_duplicates(self):
        from src.data.events.nfp import get_nfp_dates

        df = get_nfp_dates()
        assert df["date"].is_unique

    def test_days_to_next_nfp(self):
        from src.data.events.nfp import get_nfp_dates, days_to_next_nfp

        df = get_nfp_dates()
        result = days_to_next_nfp(datetime.date(2024, 3, 7), df)
        assert result == 1

    def test_in_nfp_window_false_far_away(self):
        from src.data.events.nfp import get_nfp_dates, in_nfp_window

        df = get_nfp_dates()
        assert in_nfp_window(datetime.date(2024, 3, 20), df, window=2) is False


class TestEarnings:
    def test_get_earnings_dates_returns_dataframe(self):
        from src.data.events.earnings import get_earnings_dates

        df = get_earnings_dates("AAPL")
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "ticker" in df.columns
        assert "event_type" in df.columns

    def test_earnings_ticker_column_correct(self):
        from src.data.events.earnings import get_earnings_dates

        df = get_earnings_dates("AAPL")
        if not df.empty:
            assert (df["ticker"] == "AAPL").all()
            assert (df["event_type"] == "earnings").all()

    def test_get_all_earnings_combines_tickers(self):
        from src.data.events.earnings import get_all_earnings

        df = get_all_earnings(tickers=["AAPL", "MSFT"])
        if not df.empty:
            assert set(df["ticker"].unique()).issubset({"AAPL", "MSFT"})

    def test_earnings_no_duplicates_per_ticker(self):
        from src.data.events.earnings import get_earnings_dates

        df = get_earnings_dates("AAPL")
        if not df.empty:
            assert df["date"].is_unique

    def test_days_to_next_earnings(self):
        from src.data.events.earnings import get_earnings_dates, days_to_next_earnings

        df = get_earnings_dates("AAPL")
        if not df.empty:
            first_date = pd.Timestamp(df.iloc[0]["date"]).date()
            before = first_date - datetime.timedelta(days=5)
            result = days_to_next_earnings(before, "AAPL", df)
            assert result is not None
            assert result >= 0

    def test_spy_returns_empty(self):
        from src.data.events.earnings import get_earnings_dates

        df = get_earnings_dates("SPY")
        assert df.empty


class TestUnified:
    def test_build_unified_calendar(self):
        from src.data.events.unified import build_unified_calendar

        cal = build_unified_calendar()
        assert isinstance(cal, pd.DataFrame)
        assert "date" in cal.columns
        assert "event_type" in cal.columns

    def test_unified_contains_all_types(self):
        from src.data.events.unified import build_unified_calendar

        cal = build_unified_calendar()
        types = set(cal["event_type"].unique())
        assert "fomc" in types
        assert "cpi" in types
        assert "nfp" in types
        assert "earnings" in types

    def test_unified_sorted_by_date(self):
        from src.data.events.unified import build_unified_calendar

        cal = build_unified_calendar()
        dates = cal["date"].tolist()
        assert dates == sorted(dates)

    def test_get_macro_event_features_returns_all_keys(self):
        from src.data.events.fomc import get_fomc_dates
        from src.data.events.cpi import get_cpi_dates
        from src.data.events.nfp import get_nfp_dates
        from src.data.events.unified import get_macro_event_features

        fomc_df = get_fomc_dates()
        cpi_df = get_cpi_dates()
        nfp_df = get_nfp_dates()

        features = get_macro_event_features(
            datetime.date(2024, 6, 1), fomc_df, cpi_df, nfp_df
        )

        expected_keys = {
            "days_to_fomc", "days_since_fomc", "in_fomc_window",
            "days_to_cpi", "days_since_cpi", "in_cpi_window",
            "days_to_nfp", "days_since_nfp", "in_nfp_window",
            "in_any_macro_window",
        }
        assert set(features.keys()) == expected_keys

    def test_macro_features_fomc_day(self):
        from src.data.events.fomc import get_fomc_dates
        from src.data.events.cpi import get_cpi_dates
        from src.data.events.nfp import get_nfp_dates
        from src.data.events.unified import get_macro_event_features

        fomc_df = get_fomc_dates()
        cpi_df = get_cpi_dates()
        nfp_df = get_nfp_dates()

        features = get_macro_event_features(
            datetime.date(2024, 3, 20), fomc_df, cpi_df, nfp_df
        )
        assert features["days_to_fomc"] == 0
        assert features["in_fomc_window"] == 1
        assert features["in_any_macro_window"] == 1

    def test_get_nearest_macro_event(self):
        from src.data.events.fomc import get_fomc_dates
        from src.data.events.cpi import get_cpi_dates
        from src.data.events.nfp import get_nfp_dates
        from src.data.events.unified import get_nearest_macro_event

        fomc_df = get_fomc_dates()
        cpi_df = get_cpi_dates()
        nfp_df = get_nfp_dates()

        result = get_nearest_macro_event(
            datetime.date(2024, 3, 19), fomc_df, cpi_df, nfp_df
        )
        assert result["nearest_macro_event_type"] == "fomc"
        assert result["days_to_nearest_macro"] == 1


class TestFetchEventsFacade:
    def test_facade_exports_macro_functions(self):
        from src.data.fetch_events import (
            get_fomc_dates,
            get_cpi_dates,
            get_nfp_dates,
            get_macro_event_features,
            get_nearest_macro_event,
        )

        assert callable(get_fomc_dates)
        assert callable(get_cpi_dates)
        assert callable(get_nfp_dates)
        assert callable(get_macro_event_features)
        assert callable(get_nearest_macro_event)

    def test_facade_exports_original_functions(self):
        from src.data.fetch_events import (
            fetch_earnings_dates,
            fetch_vix,
            get_days_to_next_earnings,
            get_days_since_last_earnings,
            get_earnings_direction,
            get_vix_regime,
        )

        assert callable(fetch_earnings_dates)
        assert callable(fetch_vix)
        assert callable(get_days_to_next_earnings)
        assert callable(get_days_since_last_earnings)
        assert callable(get_earnings_direction)
        assert callable(get_vix_regime)


class TestDataIntegrity:
    def test_csv_files_exist(self):
        from pathlib import Path

        events_dir = Path("data/raw/events")
        assert (events_dir / "fomc_dates.csv").exists()
        assert (events_dir / "cpi_dates.csv").exists()
        assert (events_dir / "nfp_dates.csv").exists()
        assert (events_dir / "unified_calendar.csv").exists()

    def test_csv_schemas_correct(self):
        fomc = pd.read_csv("data/raw/events/fomc_dates.csv")
        assert list(fomc.columns) == ["date", "event_type"]

        cpi = pd.read_csv("data/raw/events/cpi_dates.csv")
        assert list(cpi.columns) == ["date", "event_type"]

        nfp = pd.read_csv("data/raw/events/nfp_dates.csv")
        assert list(nfp.columns) == ["date", "event_type"]

        unified = pd.read_csv("data/raw/events/unified_calendar.csv")
        assert set(unified.columns) == {"date", "event_type", "ticker"}

    def test_unified_row_count_matches_sum(self):
        fomc = pd.read_csv("data/raw/events/fomc_dates.csv")
        cpi = pd.read_csv("data/raw/events/cpi_dates.csv")
        nfp = pd.read_csv("data/raw/events/nfp_dates.csv")
        unified = pd.read_csv("data/raw/events/unified_calendar.csv")

        earnings_count = len(unified[unified["event_type"] == "earnings"])
        expected = len(fomc) + len(cpi) + len(nfp) + earnings_count
        assert len(unified) == expected

    def test_dates_parseable(self):
        for fname in ["fomc_dates.csv", "cpi_dates.csv", "nfp_dates.csv"]:
            df = pd.read_csv(f"data/raw/events/{fname}")
            parsed = pd.to_datetime(df["date"], errors="coerce")
            assert parsed.isna().sum() == 0, f"Unparseable dates in {fname}"
