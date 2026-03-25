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
