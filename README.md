# BS Blindspots

Investigating when and why Black-Scholes misprices options using real market data and ML.

**Core question:** When does the market know something Black-Scholes doesn't?

## Setup

```bash
# Install dependencies with uv
uv sync

# Copy and fill in your API keys
cp .env.example .env

# Run tests
uv run pytest
```

## Refresh Data

```bash
# Refresh the latest raw and processed datasets
uv run python -m src.data.refresh_all --force

# Optionally include a small recent Polygon historical backfill
uv run python -m src.data.refresh_all --force --polygon-history

# Refresh the broad daily universe
uv run python -m src.data.refresh_all --force --large-universe --top-stock-count 200 --top-etf-count 20

# Export the current raw and processed data to Excel
uv run python -m src.data.export_excel
```

## Daily Yahoo Collector

```bash
# Manual run
./scripts/run_yahoo_daily_refresh.sh
```

The repo also includes a macOS LaunchAgent plist at `ops/launchd/com.andyque.bsblindspots.yahoo-refresh.plist`
that can run the Yahoo-based refresh automatically every weekday at `1:15 PM` Pacific.

There is also a separate Polygon backfill agent at `ops/launchd/com.andyque.bsblindspots.polygon-refresh.plist`
that runs a small recurring historical options backfill every weekday at `2:00 PM` Pacific.

## Project Structure

- `src/data/` - Data fetching (yfinance, FRED)
- `src/pricing/` - Black-Scholes implementation, volatility calculations
- `src/features/` - Feature engineering for ML
- `src/models/` - XGBoost/LightGBM training, evaluation, SHAP analysis
- `src/viz/` - Plotting utilities
- `notebooks/` - Research notebooks (EDA, analysis, interpretation)
- `tests/` - Pipeline tests
- `data/` - Raw, processed, and feature data (git-ignored)

See `AGENTS.md` for full project specification.
