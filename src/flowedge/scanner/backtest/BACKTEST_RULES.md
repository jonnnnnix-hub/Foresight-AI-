# Backtest & Regression Rules

## Rule 1: No Simulated Options Data — EVER

All option pricing in backtests and regression tests **must** use real market data.

**Prohibited:**
- Black-Scholes estimated option prices
- Synthetic/approximated option premiums
- Delta-approximated P&L
- Any form of simulated or modeled option pricing

**Required:**
- Real OPRA minute bars (from Massive S3 / Polygon flat files)
- Real Polygon REST API options snapshots
- Real ORATS historical IV and pricing data

If no real options data exists for a given date, that date must be **skipped** — never filled with synthetic data.

## Data Sources (in priority order)

1. **Massive S3 OPRA flat files** — Bulk historical, cached locally at `data/flat_files_s3/{TICKER}/options_1min/`
2. **Polygon REST API** — On-demand via `polygon_intraday.py` for missing dates
3. **ORATS API** — Historical IV surfaces and option pricing

## How to Download Options Data

```bash
# Trident (SPY, QQQ, IWM):
.venv/bin/python scripts/download_trident_opra.py

# All tickers:
.venv/bin/python scripts/download_opra.py

# Check what's cached:
.venv/bin/python scripts/download_trident_opra.py --check-only
```

## Applying This Rule

Every backtester must:
1. Check for OPRA data existence before processing a date
2. Skip dates with no real options data
3. Log how many days were skipped vs processed
4. Report `days_with_opra` and `days_skipped_no_opra` in results

The `OptionsMatcher` class enforces this — it only returns contracts with real OPRA minute bars.
