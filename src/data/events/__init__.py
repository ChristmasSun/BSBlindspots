"""
Events data package for BSBlindspots.

Provides modular collection of macro event calendars:
  - FOMC meeting dates (Federal Reserve interest rate decisions)
  - CPI release dates (Bureau of Labor Statistics inflation data)
  - NFP release dates (Bureau of Labor Statistics jobs reports)
  - Earnings dates (per-ticker, via yfinance)

Usage:
    from src.data.events import fomc, cpi, nfp, earnings, unified

    # Get individual event DataFrames
    fomc_df = fomc.get_fomc_dates()
    cpi_df = cpi.get_cpi_dates()
    nfp_df = nfp.get_nfp_dates()
    earnings_df = earnings.get_earnings_dates("AAPL")

    # Get unified calendar with all events merged
    calendar_df = unified.build_unified_calendar()

    # Compute distance features for a given date
    from src.data.events.unified import get_macro_event_features
    features = get_macro_event_features(date, fomc_df, cpi_df, nfp_df)
"""

from src.data.events import fomc, cpi, nfp, earnings, unified

__all__ = ["fomc", "cpi", "nfp", "earnings", "unified"]
