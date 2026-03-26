# Known Limitations & Future Upgrades

## Polygon.io Free Tier (Current)

### Bid/Ask are approximated from OHLCV
- `fetch_options.py` lines 228-231: `low` is used as bid proxy, `high` as ask proxy
- The free tier aggregates endpoint only returns OHLCV, not actual bid/ask quotes
- This means `mid_price`, `bid_ask_spread`, and `bid_ask_rel` features are approximations
- **Fix:** Upgrade to Starter ($29/mo) for flat file access with real bid/ask quotes

### Open interest is unavailable
- `open_interest` is hardcoded to 0 for all Polygon historical data
- The aggregates endpoint doesn't include OI
- The OI filter (`volume > 0 or open_interest >= 100`) effectively becomes just `volume > 0`
- `log_open_interest` feature will be meaningless for historical data
- **Fix:** Upgrade to Starter for flat files, or Advanced ($199/mo) for REST quotes with OI

### Rate limits
- Free tier: 5 requests/minute
- Current workaround: 12-second sleep every 4 requests
- Fetching historical options for many tickers is slow
- **Fix:** Any paid tier removes rate limits

### Historical depth
- Free tier: 2 years of data
- **Fix:** Developer ($79/mo) = 4 years, Advanced ($199/mo) = 5+ years

## Alpha Vantage Free Tier (Current)

### Request limits
- 25 requests/day per key (we have 2 keys = 50/day)
- Treasury yield only needs 1 call to get full history, so this is fine for rates
- Could be a bottleneck if we add more Alpha Vantage endpoints later

## yfinance

### Options chains are current-only
- No historical options chains available through yfinance
- `fetch_current_chains()` captures a single snapshot in time
- To build a historical dataset via yfinance, you'd need to run a daily cron job
- Historical data comes from Polygon instead

### Earnings dates are limited
- `get_earnings_dates(limit=20)` returns ~5 years of quarterly earnings
- Sufficient for our 2-year analysis window

## Impact on Research

### Features affected by Polygon free tier limitations
| Feature | Impact | Severity |
|---------|--------|----------|
| `bid_ask_spread` | Approximated (high - low instead of ask - bid) | Medium |
| `bid_ask_rel` | Approximated | Medium |
| `mid_price` | Approximated (affects all error metrics) | High |
| `log_open_interest` | Always 0 for historical data | High — exclude from model or use current-only data |

### Recommended approach
1. Start with current yfinance chains (real bid/ask) for model development
2. Use Polygon historical with approximations for backtesting
3. Upgrade to Starter when ready for publication-quality results
