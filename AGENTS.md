# Black-Scholes Mispricing Research Project

## Project Overview

We are building a quantitative research project that investigates when and why Black-Scholes misprices options using real market data and ML. The core research question is: **"When does the market know something Black-Scholes doesn't?"**

The angle is event-conditioned mispricing — options markets price in forward-looking information (earnings, Fed meetings, VIX spikes) that BS structurally cannot capture. We use ML + SHAP not just to outperform BS, but to identify what information BS is blind to and when.

This is NOT a "we implemented BS and compared prices" project. It is an empirical research pipeline that uses ML as a lens for understanding market microstructure and model failure modes.

## Research Framing

**Thesis:** Black-Scholes mispricing is systematic, not random. The pattern of errors reveals what information the options market is pricing in that BS cannot — specifically around earnings events, volatility regime shifts, and macro announcements. We use gradient boosting + SHAP to make this interpretable.

**Key deliverable:** A finding like *"proximity to earnings and VIX regime together explain ~X% of systematic BS mispricing, with the model learning to predict larger underpricing of OTM puts in high-vol regimes"* — economically meaningful, not just a benchmark number.

## Tech Stack

**Package management:** uv ([docs.astral.sh/uv](https://docs.astral.sh/uv/)). Use `uv sync` to install deps, `uv run` to execute scripts/tests (e.g. `uv run pytest`). Dependencies are defined in `pyproject.toml`.

- Python 3.11+
- pandas, numpy, scipy
- yfinance (stock prices, options chains, VIX)
- alpha_vantage (risk-free rate via Treasury Yield endpoint)
- polygon-api-client (historical options data)
- xgboost, lightgbm
- shap
- scikit-learn
- plotly, matplotlib
- streamlit (optional dashboard)
- jupyter notebooks for research/EDA
- pytest for pipeline tests

## Data Sources

| Data | Source | Notes |
|------|--------|-------|
| Stock prices (OHLCV) | yfinance | Free, no key needed |
| Options chains (current) | yfinance | Current chains only; collect daily via script |
| Options chains (historical) | Polygon.io (now Massive.com) | Free tier: 2yr EOD, 5 req/min. Starter ($29/mo) adds flat files with bid/ask. Use `polygon-api-client` SDK. |
| VIX | yfinance ticker `^VIX` | Vol regime feature |
| Risk-free rate | Alpha Vantage `TREASURY_YIELD` (3month maturity) | Match to option maturity. Free tier: 25 req/day. |
| Earnings dates | yfinance `.calendar` | For event flagging |

**API keys needed:** Alpha Vantage (free at alphavantage.co), Polygon.io (free at polygon.io). Store in `.env` as `ALPHA_VANTAGE_API_KEY` and `POLYGON_API_KEY`.

## Target Universe

- **Tickers:** SPY, AAPL, NVDA, MSFT (start with SPY + AAPL)
- **Maturities:** 7–180 days to expiry
- **Moneyness:** 0.8 ≤ S/K ≤ 1.2 (filter deep ITM/OTM noise)
- **Price filter:** option mid-price > $0.10
- **Use mid-price** = (bid + ask) / 2 as market price
- Both calls and puts
- **Filter:** volume > 0 or open_interest > 100

## Project Structure

```
/
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── src/
│   ├── data/
│   │   ├── fetch_stocks.py
│   │   ├── fetch_options.py
│   │   ├── fetch_rates.py
│   │   └── fetch_events.py
│   ├── pricing/
│   │   ├── black_scholes.py
│   │   ├── volatility.py
│   │   └── implied_vol.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── explain.py
│   └── viz/
│       └── plots.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_bs_error_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_shap_interpretation.ipynb
├── tests/
├── .env
└── README.md
```

## Black-Scholes Implementation

Standard BS for European options:

```
d1 = (ln(S/K) + (r + 0.5σ²)T) / (σ√T)
d2 = d1 - σ√T
Call = S·N(d1) - K·e^(-rT)·N(d2)
Put = K·e^(-rT)·N(-d2) - S·N(-d1)
```

**Volatility input:** 20-day rolling historical vol from log returns of closing prices. This is intentional — we want to test BS using ex-ante info only, so errors are real and meaningful.

## Error Metrics

For each contract compute:

- `abs_error = |market_mid - bs_price|`
- `signed_error = market_mid - bs_price` (positive = BS underprices)
- `rel_error = (market_mid - bs_price) / market_mid`
- `sq_error = (market_mid - bs_price)²`

## Feature Engineering

Build these features per contract for the ML model:

### Option features

- `moneyness = S/K`
- `log_moneyness = ln(S/K)`
- `dte = days to expiry`
- `time_to_maturity = dte / 252`
- `option_type = call/put` (binary)

### Volatility features

- `hist_vol_20` = 20-day realized vol
- `hist_vol_60` = 60-day realized vol
- `vix` = VIX level on that date
- `vol_ratio = hist_vol_20 / hist_vol_60` (vol momentum)
- `vol_of_vol` = rolling std of daily VIX changes (20-day)

### Event features

- `days_to_earnings` = calendar days until next earnings
- `days_since_earnings` = calendar days since last earnings
- `in_earnings_window` = binary, within 5 days of earnings
- `earnings_direction` = post-event price move direction

### Liquidity/market features

- `bid_ask_spread = ask - bid`
- `bid_ask_rel = (ask - bid) / mid`
- `log_volume = log(volume + 1)`
- `log_open_interest = log(open_interest + 1)`

### Vol regime

- `vix_regime` = low/medium/high (VIX < 15, 15–25, > 25)

## ML Model

- **Target variable:** `rel_error` (relative BS mispricing)
- **Model:** XGBoost regressor as primary, LightGBM as comparison

### Training setup

- Time-based train/test split (no lookahead) — train on earlier dates, test on later
- Never random shuffle across time
- Cross-validate with `TimeSeriesSplit`
- Tune with optuna if time allows

**Evaluation:** MAE, RMSE, R² on test set vs baseline (BS alone = 0 prediction).

## SHAP Analysis (the most important part)

After training, use SHAP to answer:

1. Which features drive BS mispricing most?
2. Does `days_to_earnings` show up as a top feature?
3. Does VIX regime interact with moneyness in a meaningful way?
4. Are errors for puts vs calls driven by different features?

**Generate:** SHAP summary plot, SHAP dependence plots for top 5 features, SHAP interaction plot for moneyness x vix_regime.

## Key Research Hypotheses to Test

1. BS mispricing is larger within earnings windows
2. High VIX regime → BS systematically underprices OTM puts
3. Short-dated options (DTE < 14) have larger relative errors
4. Liquidity (bid-ask spread) correlates with mispricing magnitude
5. ML-corrected price = BS + model_prediction outperforms vanilla BS on held-out data

## Coding Conventions

- No numpy for scalar ops in pricing functions — use `math` and plain Python
- No list comprehensions or single-line for loops — explicit multi-line with `.append()`
- Minimal comments
- Type hints on all functions
- All data pulls cache to `data/raw/` so we don't re-hit APIs constantly
- `.env` for all API keys, never hardcoded

## Current Status

Full pipeline implemented. All modules built:

**Data layer:**
- `src/data/fetch_stocks.py` — yfinance OHLCV + log returns, 12hr cache
- `src/data/fetch_rates.py` — Alpha Vantage 3mo Treasury yield, date lookup helper
- `src/data/fetch_options.py` — yfinance current chains + Polygon.io historical, all filters applied
- `src/data/fetch_events.py` — earnings dates + VIX history + regime helper

**Pricing layer:**
- `src/pricing/black_scholes.py` — BS pricer, greeks, error metrics (abs, signed, rel, sq)
- `src/pricing/volatility.py` — rolling historical vol (20d, 60d), vol ratio, vol of vol

**Feature engineering:**
- `src/features/build_features.py` — assembles full feature matrix from all data + pricing modules

**Model layer:**
- `src/models/train.py` — XGBoost + LightGBM with time-based splits, TimeSeriesSplit CV
- `src/models/evaluate.py` — metrics, baseline comparison, regime/moneyness breakdowns
- `src/models/explain.py` — SHAP analysis: summary, dependence, interaction, calls vs puts

See `LIMITATIONS.md` for known data limitations (Polygon free tier approximations).
