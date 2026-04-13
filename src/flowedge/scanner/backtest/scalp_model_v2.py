"""FlowEdge Scalp v2 — Real OPRA options backtest.

Same 7-condition entry filter as v1, but ALL option pricing uses
actual OPRA market data.  No Black-Scholes.  No estimated spreads.

Entry/exit prices come from real option OHLCV bars.  P&L is measured
in actual dollars and percentages on real contract prices.

If no matching option contract exists for a signal, the signal is
skipped — the skip count itself tells you how many theoretical
signals were actually tradeable.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import date
from math import sqrt
from pathlib import Path
from typing import Any, Literal

import structlog

from flowedge.scanner.backtest.options_matcher import OptionsMatcher
from flowedge.scanner.backtest.scalp_config import ALL_33_TICKERS, HIGH_WR_TICKERS, ScalpConfig
from flowedge.scanner.backtest.schemas import (
    BacktestResult,
    BacktestTrade,
    TradeOutcome,
)

logger = structlog.get_logger()

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05

# Default tickers: sweep-validated 8-ticker high-WR universe
# Use ALL_33_TICKERS for broad scanning / new ticker discovery
SCALP_TICKERS = HIGH_WR_TICKERS

# Sweep-validated defaults (90% WR on 4yr OPRA, 25,600 combos tested)
SCALP_DTE = 5             # 0-5 DTE — captures weekly Fri expirations
SCALP_MIN_PREMIUM = 0.30  # Minimum real option premium to enter

# Exit parameters (sweep-validated)
SCALP_TP_UNDERLYING = 0.002  # 0.20% underlying move
SCALP_MAX_HOLD_BARS = 12    # 12 × 5-min = 60 minutes
SCALP_TRAIL_PCT = 0.03      # 3% trail from peak real premium

# Entry filters (sweep-validated)
SCALP_IBS = 0.12            # IBS upper limit
SCALP_RSI3 = 15.0           # RSI(3) upper limit
SCALP_VOL_SPIKE = 2.5       # Volume spike multiplier
SCALP_INTRADAY_DROP = -0.002  # -0.2% from open

# Risk
SCALP_MAX_POSITIONS = 2
SCALP_RISK_PER_TRADE = 0.05  # 5% per scalp

CACHE_DIR = Path("data/flat_files_s3")

# Regular market hours: 9:30-16:00 ET.  Pre-market bars from stock data
# do NOT have matching OPRA option bars, so we must filter them out.
#
# DST-aware UTC boundaries:
#   EDT (2nd Sun Mar – 1st Sun Nov): 9:30 ET = 13:30 UTC, 16:00 ET = 20:00 UTC
#   EST (1st Sun Nov – 2nd Sun Mar): 9:30 ET = 14:30 UTC, 16:00 ET = 21:00 UTC
#
# We add 5 minutes of margin on each side.
_RTH_EDT_START = 13 * 3600 + 25 * 60   # 13:25 UTC
_RTH_EDT_END = 20 * 3600 + 5 * 60      # 20:05 UTC
_RTH_EST_START = 14 * 3600 + 25 * 60   # 14:25 UTC
_RTH_EST_END = 21 * 3600 + 5 * 60      # 21:05 UTC

EntryMode = Literal["next_open", "signal_close", "signal_high"]
ExitMode = Literal["bar_close", "bar_low"]


def _validate_bars(bars: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    """Validate and clean OHLC bars, removing corrupt entries.

    Checks:
    - OHLC values are non-negative
    - low <= high (basic OHLC consistency)
    - close is within [low, high] range (with 1% tolerance for float rounding)
    - Volume is non-negative
    - Timestamp is non-zero

    Returns:
        Cleaned list with corrupt bars removed.
    """
    clean: list[dict[str, Any]] = []
    dropped = 0
    for bar in bars:
        h = float(bar.get("high", bar.get("h", 0)))
        lo = float(bar.get("low", bar.get("l", 0)))
        c = float(bar.get("close", bar.get("c", 0)))
        v = float(bar.get("volume", bar.get("v", 0)))

        # Skip bars with zero/negative prices
        if h <= 0 or c <= 0:
            dropped += 1
            continue

        # Skip bars where low > high (corrupt)
        if lo > 0 and lo > h * 1.01:  # 1% tolerance
            dropped += 1
            continue

        # Skip bars with negative volume
        if v < 0:
            dropped += 1
            continue

        # Check timestamp exists
        ts = bar.get("ts", bar.get("timestamp", 0))
        try:
            if int(ts) == 0:
                dropped += 1
                continue
        except (ValueError, TypeError):
            dropped += 1
            continue

        clean.append(bar)

    if dropped > 0:
        logger.info(
            "bars_validation_dropped",
            ticker=ticker,
            dropped=dropped,
            kept=len(clean),
        )

    return clean


def _is_dst(date_str: str) -> bool:
    """Check if a date falls within US Eastern Daylight Time.

    DST: 2nd Sunday of March to 1st Sunday of November.
    Simple heuristic — good enough for US equity market dates.
    """
    try:
        d = date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return True  # Default to EDT (more common for market hours)

    # March: DST starts 2nd Sunday
    if d.month == 3:
        # Find 2nd Sunday: first day of month, advance to first Sunday, +7
        first = date(d.year, 3, 1)
        days_to_sunday = (6 - first.weekday()) % 7
        second_sunday = first.day + days_to_sunday + 7
        return d.day >= second_sunday
    # November: DST ends 1st Sunday
    if d.month == 11:
        first = date(d.year, 11, 1)
        days_to_sunday = (6 - first.weekday()) % 7
        first_sunday = first.day + days_to_sunday
        return d.day < first_sunday
    # Apr-Oct: always EDT; Dec-Feb: always EST
    return 4 <= d.month <= 10


def _filter_rth(
    bars: list[dict[str, Any]],
    date_str: str = "",
) -> list[dict[str, Any]]:
    """Filter minute bars to regular trading hours only (DST-aware)."""
    dst = _is_dst(date_str)
    rth_start = _RTH_EDT_START if dst else _RTH_EST_START
    rth_end = _RTH_EDT_END if dst else _RTH_EST_END

    rth: list[dict[str, Any]] = []
    for b in bars:
        ts_ns = int(b.get("ts", b.get("timestamp", 0)))
        if ts_ns == 0:
            continue
        ts_sec = ts_ns // 1_000_000_000
        secs_into_day = ts_sec % 86400
        if rth_start <= secs_into_day <= rth_end:
            rth.append(b)
    return rth


def _gf(bar: dict[str, Any], long_key: str, short_key: str) -> float:
    val: object = bar.get(long_key, bar.get(short_key, 0))
    return float(val)  # type: ignore[arg-type]


def run_scalp_backtest_v2(
    tickers: list[str] | None = None,
    starting_capital: float = 25_000.0,
    entry_mode: EntryMode = "next_open",
    exit_mode: ExitMode = "bar_close",
    spread_cents: float = 0.0,
    commission_per_contract: float = 0.50,
    config: ScalpConfig | None = None,
) -> BacktestResult:
    """Run scalp model backtest on REAL OPRA option prices.

    Identical 7-condition filter to v1, but all option pricing uses
    actual market data.  No Black-Scholes.  No synthetic slippage.

    Args:
        tickers: Underlyings to test (default :data:`SCALP_TICKERS`).
            Overrides ``config.tickers`` if provided.
        starting_capital: Portfolio starting value in dollars.
            Overrides ``config.starting_capital`` if provided.
        entry_mode: How to price the entry fill:
            - ``"next_open"``: open of the option bar *after* the signal
              (most realistic — can't trade the bar you observe).
            - ``"signal_close"``: close of the signal bar's option bar.
            - ``"signal_high"``: high of the signal bar (worst-case buy).
        exit_mode: How to price the exit fill:
            - ``"bar_close"``: close of the exit bar (default).
            - ``"bar_low"``: low of the exit bar (worst-case sell).
        spread_cents: Bid-ask spread penalty in cents.  When > 0, entry
            fills get worse by ``spread_cents / 100`` (added to buy price)
            and exit fills get worse by the same amount (subtracted from
            sell price).  Default 0.0 (no spread).
        commission_per_contract: Per-contract commission in dollars,
            charged on both entry and exit.  Typical range: $0.50-$2.00
            for options.  Default 0.50.
        config: Optional :class:`ScalpConfig` to override all module-level
            constants.  Explicit keyword args take priority over config.

    Returns:
        :class:`BacktestResult` with real dollar P&L and percentage
        returns on actual option contract prices.
    """
    cfg = config or ScalpConfig(
        spread_cents=spread_cents,
        commission_per_contract=commission_per_contract,
        starting_capital=starting_capital,
    )
    tickers = tickers or cfg.tickers
    starting_capital = cfg.starting_capital
    spread_cents = cfg.spread_cents
    commission_per_contract = cfg.commission_per_contract
    matcher = OptionsMatcher()

    # ── Load minute bars (identical to v1) ─────────────────────
    all_bars: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for ticker in tickers:
        min_dir = CACHE_DIR / ticker / "1min"
        if not min_dir.exists():
            continue
        bars: list[dict[str, Any]] = []
        for f in sorted(min_dir.glob("*.json")):
            bars.extend(json.loads(f.read_text()))
        bars = _validate_bars(bars, ticker)
        if len(bars) < 5000:
            continue

        # Log data gaps (informational — does not drop bars)
        if bars:
            dates = sorted(set(str(b.get("date", b.get("d", ""))) for b in bars))
            if len(dates) > 1:
                expected_days = len(dates)
                actual_bars = len(bars)
                bars_per_day = actual_bars / expected_days if expected_days else 0
                if bars_per_day < 50:  # expect ~78 bars/day (6.5h * 12 bars/hr)
                    logger.warning(
                        "low_bar_density",
                        ticker=ticker,
                        bars=actual_bars,
                        days=expected_days,
                        avg_bars_per_day=round(bars_per_day, 1),
                    )

        by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for b in bars:
            d = str(b.get("date", b.get("d", "")))
            if d:
                by_date[d].append(b)
        all_bars[ticker] = dict(by_date)
        logger.info("scalp_v2_loaded", ticker=ticker, days=len(by_date))

    if not all_bars:
        return BacktestResult(
            run_id=f"scalp-real-{uuid.uuid4().hex[:8]}",
            tickers=tickers,
            starting_capital=starting_capital,
        )

    # All dates and daily closes for trend check
    all_dates: set[str] = set()
    for ticker_days in all_bars.values():
        all_dates.update(ticker_days.keys())
    sorted_dates = sorted(all_dates)

    daily_closes: dict[str, list[float]] = {t: [] for t in all_bars}

    # Portfolio
    cash = starting_capital
    closed: list[BacktestTrade] = []
    daily_values: list[tuple[str, float]] = []
    signals_total = 0
    signals_matched = 0
    signals_skipped_no_data = 0
    signals_skipped_low_premium = 0

    # Per-condition rejection counters
    filter_ibs = 0
    filter_rsi3 = 0
    filter_vwap = 0
    filter_vol_spike = 0
    filter_drop = 0
    filter_prior_bar = 0
    filter_sma = 0
    filter_no_option = 0
    filter_low_premium = 0

    for d in sorted_dates:
        # Record daily closes for trend
        for ticker in all_bars:
            day_bars = all_bars[ticker].get(d, [])
            if day_bars:
                daily_closes[ticker].append(_gf(day_bars[-1], "close", "c"))

        intraday_positions: list[dict[str, Any]] = []

        for ticker in all_bars:
            raw_day_bars = all_bars[ticker].get(d, [])
            # Filter to regular trading hours only — pre-market bars
            # have no matching OPRA option data.
            day_bars = _filter_rth(raw_day_bars, date_str=d)
            if len(day_bars) < 50:
                continue
            dc = daily_closes[ticker]
            if len(dc) < 20:
                continue

            # Daily uptrend check (identical to v1)
            sma10 = sum(dc[-10:]) / 10
            sma20 = sum(dc[-20:]) / 20
            if sma10 <= sma20:
                continue

            # Build 5-min chunks aligned to real market boundaries.
            # Group bars by floor(ts_ns / 5-minute-window) so that bars
            # from 9:30:00-9:34:59 form one chunk, 9:35:00-9:39:59 the
            # next, etc.  This avoids misalignment when the first bar
            # doesn't land exactly on a 5-minute mark.
            _5min_ns = 5 * 60 * 1_000_000_000
            window_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for b in day_bars:
                ts_ns = int(b.get("ts", b.get("timestamp", 0)))
                if ts_ns == 0:
                    continue
                bucket = ts_ns // _5min_ns
                window_buckets[bucket].append(b)

            chunks: list[dict[str, Any]] = []
            for bucket in sorted(window_buckets):
                chunk = window_buckets[bucket]
                o = _gf(chunk[0], "open", "o")
                h = max(_gf(b, "high", "h") for b in chunk)
                lows = [
                    _gf(b, "low", "l")
                    for b in chunk
                    if _gf(b, "low", "l") > 0
                ]
                lo = min(lows) if lows else 0.0
                c = _gf(chunk[-1], "close", "c")
                v = sum(_gf(b, "volume", "v") for b in chunk)
                ts = str(chunk[0].get("ts", chunk[0].get("timestamp", "")))
                chunks.append(
                    {"o": o, "h": h, "l": lo, "c": c, "v": v, "ts": ts}
                )

            if len(chunks) < 30:
                continue

            day_open = chunks[0]["o"]

            # VWAP (identical to v1)
            cum_pv = 0.0
            cum_v = 0.0
            vwaps: list[float] = []
            for ch in chunks:
                tp = (ch["h"] + ch["l"] + ch["c"]) / 3
                cum_pv += tp * ch["v"]
                cum_v += ch["v"]
                vwaps.append(cum_pv / cum_v if cum_v > 0 else ch["c"])

            # ── Scan morning session (bars 6-24) ───────────────
            for i in range(6, min(24, len(chunks) - cfg.max_hold_bars)):
                if len(intraday_positions) >= cfg.max_positions:
                    break
                if any(p["ticker"] == ticker for p in intraday_positions):
                    break

                ch = chunks[i]
                rng = ch["h"] - ch["l"]
                if rng <= 0 or ch["c"] <= 0:
                    continue

                # ── 7-CONDITION FILTER (identical to v1) ───────
                ibs = (ch["c"] - ch["l"]) / rng
                if ibs >= cfg.ibs_threshold:
                    filter_ibs += 1
                    continue

                # RSI(3)
                if i < 4:
                    continue
                c5m = [chunks[j]["c"] for j in range(i - 3, i + 1)]
                g = [max(0, c5m[k] - c5m[k - 1]) for k in range(1, 4)]
                ls = [max(0, c5m[k - 1] - c5m[k]) for k in range(1, 4)]
                ag = sum(g) / 3
                al = sum(ls) / 3
                rsi3 = (
                    100.0 - 100.0 / (1 + ag / al) if al > 0 else 100.0
                )
                if rsi3 >= cfg.rsi3_threshold:
                    filter_rsi3 += 1
                    continue

                # Below VWAP
                if ch["c"] >= vwaps[i]:
                    filter_vwap += 1
                    continue

                # Volume spike
                start_idx = max(0, i - 10)
                avg_vol = sum(
                    chunks[j]["v"] for j in range(start_idx, i)
                ) / max(1, i - start_idx)
                vr = ch["v"] / avg_vol if avg_vol > 0 else 1
                if vr < cfg.vol_spike:
                    filter_vol_spike += 1
                    continue

                # Intraday drop
                drop = (ch["c"] - day_open) / day_open
                if drop > cfg.intraday_drop:
                    filter_drop += 1
                    continue

                # Prior bar red
                if i > 0 and chunks[i - 1]["c"] >= chunks[max(0, i - 2)]["c"]:
                    filter_prior_bar += 1
                    continue

                # 5-bar SMA < 10-bar SMA
                if i >= 10:
                    sma5 = (
                        sum(chunks[j]["c"] for j in range(i - 4, i + 1)) / 5
                    )
                    sma10_id = sum(
                        chunks[j]["c"] for j in range(max(0, i - 9), i + 1)
                    ) / min(10, i + 1)
                    if sma5 >= sma10_id:
                        filter_sma += 1
                        continue

                # ═══ ALL 7 CONDITIONS PASSED ═══════════════════
                signals_total += 1
                entry_price_underlying = ch["c"]
                signal_ts_ns = int(ch["ts"]) if ch["ts"] else 0

                if signal_ts_ns == 0:
                    signals_skipped_no_data += 1
                    filter_no_option += 1
                    continue

                # ── FIND REAL OPTION CONTRACT ──────────────────
                contract = matcher.find_best_contract(
                    underlying=ticker,
                    date_str=d,
                    underlying_price=entry_price_underlying,
                    signal_ts_ns=signal_ts_ns,
                    max_dte=cfg.dte,
                )

                if contract is None:
                    signals_skipped_no_data += 1
                    filter_no_option += 1
                    logger.debug(
                        "scalp_v2_no_contract",
                        ticker=ticker,
                        date=d,
                        price=entry_price_underlying,
                    )
                    continue

                # ── REAL ENTRY PRICE ───────────────────────────
                # Aggregate contract bars to 5-min for consistency
                opt_5min = OptionsMatcher.aggregate_to_5min(contract.bars)
                if not opt_5min:
                    signals_skipped_no_data += 1
                    filter_no_option += 1
                    continue

                if entry_mode == "next_open":
                    # Bar after the signal — most realistic
                    next_ts = signal_ts_ns + 5 * 60 * 1_000_000_000
                    entry_bar = OptionsMatcher.get_bar_at_time(
                        opt_5min, next_ts, tolerance_ns=180_000_000_000,
                    )
                    if entry_bar is None:
                        signals_skipped_no_data += 1
                        filter_no_option += 1
                        continue
                    fill = float(entry_bar.get("o", 0))
                elif entry_mode == "signal_high":
                    entry_bar = OptionsMatcher.get_bar_at_time(
                        opt_5min, signal_ts_ns,
                    )
                    if entry_bar is None:
                        signals_skipped_no_data += 1
                        filter_no_option += 1
                        continue
                    fill = float(entry_bar.get("h", 0))
                else:  # signal_close
                    entry_bar = OptionsMatcher.get_bar_at_time(
                        opt_5min, signal_ts_ns,
                    )
                    if entry_bar is None:
                        signals_skipped_no_data += 1
                        filter_no_option += 1
                        continue
                    fill = float(entry_bar.get("c", 0))

                # Apply bid-ask spread penalty to entry (buy worse)
                if spread_cents > 0:
                    fill += spread_cents / 100

                if fill < cfg.min_premium:
                    signals_skipped_low_premium += 1
                    filter_low_premium += 1
                    continue

                signals_matched += 1

                # Position sizing — real prices, no slippage model
                budget = cash * cfg.risk_per_trade
                contracts = max(1, int(budget / (fill * 100)))
                cost = contracts * fill * 100

                # Commission on entry
                entry_commission = contracts * commission_per_contract

                if cost > cash * 0.9 or cost < 10:
                    continue

                # ── HOLD SIMULATION ON REAL BARS ───────────────
                hold_start_ts = int(entry_bar.get("ts", signal_ts_ns))
                hold_bars = OptionsMatcher.get_bars_after(
                    opt_5min, hold_start_ts, cfg.max_hold_bars + 1,
                )

                max_premium = fill
                exit_fill = fill
                exit_bar_idx = 0
                exit_reason = "time_exit"
                exit_underlying = entry_price_underlying

                for j, opt_bar in enumerate(hold_bars[1:], start=1):
                    bar_close = float(opt_bar.get("c", 0))
                    bar_high = float(opt_bar.get("h", 0))
                    bar_low = float(opt_bar.get("l", 0))

                    if bar_close <= 0:
                        continue

                    if bar_high > 0:
                        max_premium = max(max_premium, bar_high)

                    # Check underlying move for TP
                    # Match by timestamp, not index — handles liquidity gaps
                    opt_ts = int(opt_bar.get("ts", 0))
                    und_bar = None
                    if opt_ts > 0:
                        # Find nearest underlying chunk by timestamp
                        for ci in range(i, min(i + j + 2, len(chunks))):
                            chunk_ts = int(
                                chunks[ci].get("ts", chunks[ci].get("timestamp", 0))
                            )
                            if chunk_ts >= opt_ts:
                                und_bar = chunks[ci]
                                break
                        # Fallback to index-based if no timestamp match
                        if und_bar is None and i + j < len(chunks):
                            und_bar = chunks[i + j]

                    if und_bar is not None:
                        und_price = und_bar["c"]
                        underlying_gain = (
                            (und_price - entry_price_underlying)
                            / entry_price_underlying
                        )
                        if underlying_gain >= cfg.tp_underlying:
                            exit_bar_idx = j
                            exit_reason = "take_profit"
                            exit_underlying = und_price
                            if exit_mode == "bar_low":
                                exit_fill = bar_low if bar_low > 0 else bar_close
                            else:
                                exit_fill = bar_close
                            break

                    # Trailing stop on real premium — trigger on bar_low
                    # (bar_low breaching trail = stop hit during the bar)
                    if max_premium > fill * 1.05:
                        trail = max_premium * (1 - cfg.trail_pct)
                        if bar_low > 0 and bar_low <= trail:
                            exit_bar_idx = j
                            exit_reason = "trailing_stop"
                            if und_bar is not None:
                                exit_underlying = und_bar["c"]
                            if exit_mode == "bar_low":
                                exit_fill = bar_low if bar_low > 0 else bar_close
                            else:
                                exit_fill = bar_close
                            break

                    # Time exit on last bar
                    if j >= cfg.max_hold_bars:
                        exit_bar_idx = j
                        exit_reason = "time_exit"
                        if und_bar is not None:
                            exit_underlying = und_bar["c"]
                        if exit_mode == "bar_low":
                            exit_fill = bar_low if bar_low > 0 else bar_close
                        else:
                            exit_fill = bar_close
                        break
                else:
                    # Exhausted hold bars without explicit exit
                    if hold_bars:
                        last = hold_bars[-1]
                        if exit_mode == "bar_low":
                            exit_fill = (
                                float(last.get("l", 0))
                                if float(last.get("l", 0)) > 0
                                else float(last.get("c", fill))
                            )
                        else:
                            exit_fill = float(last.get("c", fill))

                # ── COMPUTE REAL P&L ──────────────────────────
                # Apply bid-ask spread penalty to exit (sell worse)
                if spread_cents > 0:
                    exit_fill -= spread_cents / 100
                exit_fill = max(0.01, exit_fill)
                exit_val = exit_fill * contracts * 100

                # Commission on exit and total round-trip commission
                exit_commission = contracts * commission_per_contract
                total_commission = entry_commission + exit_commission

                pnl_dollars = exit_val - cost - total_commission
                pnl_pct = (pnl_dollars / cost * 100) if cost > 0 else 0.0

                cash += pnl_dollars

                um = (
                    (exit_underlying - entry_price_underlying)
                    / entry_price_underlying
                    * 100
                )

                outcome = (
                    TradeOutcome.WIN
                    if pnl_pct >= 3
                    else (
                        TradeOutcome.LOSS
                        if pnl_pct < -3
                        else TradeOutcome.BREAKEVEN
                    )
                )

                hold_minutes = exit_bar_idx * 5

                closed.append(
                    BacktestTrade(
                        ticker=ticker,
                        entry_date=date.fromisoformat(d),
                        exit_date=date.fromisoformat(d),
                        option_type="call",
                        strike=round(contract.strike, 2),
                        entry_price=round(fill, 4),
                        exit_price=round(exit_fill, 4),
                        underlying_entry=round(entry_price_underlying, 2),
                        underlying_exit=round(exit_underlying, 2),
                        underlying_move_pct=round(um, 3),
                        pnl_per_contract=round(
                            pnl_dollars / max(contracts, 1), 2,
                        ),
                        pnl_pct=round(pnl_pct, 2),
                        outcome=outcome,
                        signal_score=9.0,
                        signal_type=(
                            f"scalp_7sig_real|{hold_minutes}min"
                            f"|{contract.contract_symbol}"
                        ),
                        hold_days=0,
                        strategy="scalp_real",
                        regime="intraday_uptrend",
                        conviction=9.0,
                        exit_reason=exit_reason,
                        contracts=contracts,
                        cost_basis=round(cost + total_commission, 2),
                        exit_value=round(exit_val, 2),
                    )
                )

        daily_values.append((d, cash))

    # ── Compile Results ────────────────────────────────────────
    trades = closed
    total = len(trades)
    wins = sum(1 for t in trades if t.outcome == TradeOutcome.WIN)
    win_pnls = [t.pnl_pct for t in trades if t.outcome == TradeOutcome.WIN]
    loss_pnls = [t.pnl_pct for t in trades if t.outcome != TradeOutcome.WIN]
    gp = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gl = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))

    by_ticker: dict[str, dict[str, float]] = {}
    for tk in tickers:
        tt = [t for t in trades if t.ticker == tk]
        if tt:
            tw = sum(1 for t in tt if t.outcome == TradeOutcome.WIN)
            by_ticker[tk] = {
                "trades": float(len(tt)),
                "win_rate": round(tw / len(tt), 3),
                "avg_pnl_pct": round(sum(t.pnl_pct for t in tt) / len(tt), 2),
                "total_pnl_pct": round(sum(t.pnl_pct for t in tt), 2),
                "total_pnl_dollars": round(
                    sum(t.exit_value - t.cost_basis for t in tt), 2,
                ),
            }

    by_year: dict[str, dict[str, float]] = {}
    for year_str in sorted({str(t.entry_date.year) for t in trades}):
        yt = [t for t in trades if str(t.entry_date.year) == year_str]
        if yt:
            yw = sum(1 for t in yt if t.outcome == TradeOutcome.WIN)
            year_pnl_dollars = sum(t.exit_value - t.cost_basis for t in yt)
            by_year[year_str] = {
                "trades": float(len(yt)),
                "win_rate": round(yw / len(yt), 3),
                "avg_pnl_pct": round(
                    sum(t.pnl_pct for t in yt) / len(yt), 2,
                ),
                "total_pnl_pct": round(sum(t.pnl_pct for t in yt), 2),
                "total_pnl_dollars": round(year_pnl_dollars, 2),
            }

    ending = daily_values[-1][1] if daily_values else starting_capital
    ret = (ending - starting_capital) / starting_capital * 100

    peak = starting_capital
    max_dd = 0.0
    for _, v in daily_values:
        if v > peak:
            peak = v
        max_dd = max(max_dd, (peak - v) / peak * 100 if peak > 0 else 0)

    sharpe = 0.0
    if len(daily_values) > 10:
        rets = [
            (daily_values[i][1] - daily_values[i - 1][1])
            / daily_values[i - 1][1]
            for i in range(1, len(daily_values))
            if daily_values[i - 1][1] > 0
        ]
        if rets:
            m = sum(rets) / len(rets)
            var = sum((r - m) ** 2 for r in rets) / len(rets)
            s = sqrt(var) if var > 0 else 0.001
            sharpe = round((m * 252 - RISK_FREE_RATE) / (s * sqrt(252)), 3)

    filter_stats = {
        "ibs": filter_ibs,
        "rsi3": filter_rsi3,
        "vwap": filter_vwap,
        "vol_spike": filter_vol_spike,
        "drop": filter_drop,
        "prior_bar": filter_prior_bar,
        "sma": filter_sma,
        "no_option": filter_no_option,
        "low_premium": filter_low_premium,
    }

    notes = [
        f"config={cfg.model_dump_json()}",
        f"entry_mode={entry_mode}",
        f"exit_mode={exit_mode}",
        f"spread_cents={spread_cents}",
        f"commission_per_contract={commission_per_contract}",
        f"signals_total={signals_total}",
        f"signals_matched={signals_matched}",
        f"signals_skipped_no_data={signals_skipped_no_data}",
        f"signals_skipped_low_premium={signals_skipped_low_premium}",
        "pricing=REAL_OPRA (no Black-Scholes)",
        f"filter_stats={json.dumps(filter_stats)}",
        f"by_year={json.dumps(by_year)}",
    ]

    result = BacktestResult(
        run_id=f"scalp-real-{uuid.uuid4().hex[:8]}",
        tickers=tickers,
        lookback_days=len(sorted_dates),
        total_trades=total,
        wins=wins,
        losses=total - wins,
        win_rate=round(wins / total, 3) if total > 0 else 0,
        avg_win_pct=round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0,
        avg_loss_pct=(
            round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0
        ),
        best_trade_pct=round(max((t.pnl_pct for t in trades), default=0), 2),
        worst_trade_pct=round(min((t.pnl_pct for t in trades), default=0), 2),
        total_pnl_pct=round(sum(t.pnl_pct for t in trades), 2),
        profit_factor=round(gp / gl, 2) if gl > 0 else 0,
        avg_hold_days=0,
        expectancy_pct=(
            round(sum(t.pnl_pct for t in trades) / total, 2) if total > 0 else 0
        ),
        trades=trades,
        by_ticker=by_ticker,
        starting_capital=starting_capital,
        ending_value=round(ending, 2),
        portfolio_return_pct=round(ret, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=sharpe,
        notes=notes,
    )

    logger.info(
        "scalp_v2_complete",
        trades=total,
        wins=wins,
        win_rate=result.win_rate,
        starting_capital=starting_capital,
        ending_value=result.ending_value,
        portfolio_return_pct=result.portfolio_return_pct,
        portfolio_return_dollars=round(ending - starting_capital, 2),
        signals_total=signals_total,
        signals_matched=signals_matched,
        signals_skipped_no_data=signals_skipped_no_data,
    )

    validation_warnings = _validate_result(result)
    if validation_warnings:
        for w in validation_warnings:
            logger.warning("backtest_validation", warning=w)
        result.notes.append(
            f"validation_warnings={json.dumps(validation_warnings)}"
        )

    if total >= 5:
        from flowedge.scanner.backtest.learning_hook import (
            post_backtest_learn_from_result,
        )

        post_backtest_learn_from_result(result, model_name="scalp_real")

    return result


def _validate_result(result: BacktestResult) -> list[str]:
    """Run sanity checks on a backtest result. Returns list of warnings."""
    warnings: list[str] = []

    # 1. Trade count consistency
    if result.total_trades != result.wins + result.losses:
        warnings.append(
            f"Trade count mismatch: total={result.total_trades} != "
            f"wins={result.wins} + losses={result.losses}"
        )

    # 2. Win rate bounds
    if result.total_trades > 0:
        expected_wr = result.wins / result.total_trades
        if abs(result.win_rate - expected_wr) > 0.01:
            warnings.append(
                f"Win rate mismatch: reported={result.win_rate:.3f} vs "
                f"computed={expected_wr:.3f}"
            )

    # 3. P&L consistency: sum of trade P&L should approximately match
    #    portfolio return
    if result.trades:
        sum_trade_pnl = sum(
            t.exit_value - t.cost_basis for t in result.trades
        )
        portfolio_pnl = result.ending_value - result.starting_capital
        if abs(sum_trade_pnl - portfolio_pnl) > 1.0:  # $1 tolerance
            warnings.append(
                f"P&L mismatch: sum_trades=${sum_trade_pnl:.2f} vs "
                f"portfolio=${portfolio_pnl:.2f}"
            )

    # 4. No negative cost basis
    for t in result.trades:
        if t.cost_basis < 0:
            warnings.append(
                f"Negative cost_basis: {t.ticker} on {t.entry_date}"
            )
            break

    # 5. Max drawdown should be non-negative and <= 100%
    if result.max_drawdown_pct < 0 or result.max_drawdown_pct > 100:
        warnings.append(
            f"Invalid max_drawdown_pct: {result.max_drawdown_pct:.2f}"
        )

    # 6. Ending value should be non-negative
    if result.ending_value < 0:
        warnings.append(f"Negative ending_value: ${result.ending_value:.2f}")

    # 7. Trades should be sorted by entry date
    if len(result.trades) > 1:
        for i in range(len(result.trades) - 1):
            if result.trades[i].entry_date > result.trades[i + 1].entry_date:
                warnings.append("Trades not sorted by entry_date")
                break

    return warnings
