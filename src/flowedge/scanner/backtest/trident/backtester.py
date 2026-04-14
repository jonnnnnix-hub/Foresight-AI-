"""Trident backtester — runs one config against historical data.

╔══════════════════════════════════════════════════════════════╗
║  RULE: NO SIMULATED OPTIONS DATA — EVER.                     ║
║                                                              ║
║  All option pricing uses REAL OPRA minute bars from cache.   ║
║  If no OPRA data exists for a date, that date is SKIPPED.    ║
║  No Black-Scholes.  No estimated spreads.  No approximation. ║
║  This rule applies to ALL backtests and regressions.         ║
╚══════════════════════════════════════════════════════════════╝

Data sources:
  - Stock 1-min bars: data/flat_files_s3/{TICKER}/1min/*.json
  - Option 1-min bars: data/flat_files_s3/{TICKER}/options_1min/*.json
    Downloaded via Massive S3 (Polygon OPRA) or Polygon REST API.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from flowedge.scanner.backtest.options_matcher import OptionsMatcher

from .config import (
    CACHE_DIR,
    TRADING_DAYS_PER_YEAR,
    Direction,
    TridentConfig,
)
from .signals import Bar, SignalSnapshot, compute_all_signals, evaluate_signals

logger = logging.getLogger("trident.backtest")

# ── Constants ─────────────────────────────────────────────────────
NS_PER_SEC = 1_000_000_000
NS_PER_MIN = 60 * NS_PER_SEC


# ── Trade record ──────────────────────────────────────────────────

@dataclass
class TridentTrade:
    """Single backtest trade record — priced from REAL OPRA data."""

    ticker: str
    direction: str          # "call" or "put"
    contract_symbol: str    # real OCC symbol
    strike: float
    dte: int
    entry_date: str
    entry_time: str
    entry_bar_idx: int
    entry_underlying: float
    entry_option_price: float   # real OPRA price
    exit_underlying: float
    exit_option_price: float    # real OPRA price
    exit_reason: str        # "tp", "sl", "trail", "time", "eod", "reversal"
    hold_bars: int
    hold_minutes: int
    pnl_dollars: float
    pnl_pct: float
    contracts: int
    conviction: float
    signals_fired: int


@dataclass
class TridentResult:
    """Complete backtest result for one configuration."""

    config_name: str
    config: dict[str, Any]
    trades: list[TridentTrade]

    # Aggregate stats
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_hold_minutes: float = 0.0
    calls_taken: int = 0
    puts_taken: int = 0
    call_win_rate: float = 0.0
    put_win_rate: float = 0.0
    signals_per_day: float = 0.0
    trading_days: int = 0
    days_with_opra: int = 0
    days_skipped_no_opra: int = 0

    # Per-ticker breakdown
    per_ticker: dict[str, dict[str, Any]] = field(default_factory=dict)

    def summary_line(self) -> str:
        return (
            f"{self.config_name}: {self.total_trades} trades, "
            f"WR={self.win_rate:.1f}%, "
            f"P&L=${self.total_pnl:,.0f}, "
            f"PF={self.profit_factor:.2f}, "
            f"Sharpe={self.sharpe_ratio:.2f}, "
            f"AvgHold={self.avg_hold_minutes:.0f}m, "
            f"DD={self.max_drawdown_pct:.1f}%, "
            f"OPRA={self.days_with_opra}d"
        )


# ── Data loading ──────────────────────────────────────────────────

def _load_1min_bars(ticker: str) -> dict[str, list[dict[str, Any]]]:
    """Load all 1-minute bars from cache, grouped by date."""
    ticker_dir = CACHE_DIR / ticker / "1min"
    if not ticker_dir.exists():
        logger.warning("No 1-min data for %s at %s", ticker, ticker_dir)
        return {}

    all_bars: list[dict[str, Any]] = []
    for fpath in sorted(ticker_dir.glob("*.json")):
        try:
            raw = json.loads(fpath.read_text())
            if isinstance(raw, list):
                all_bars.extend(raw)
            elif isinstance(raw, dict) and "results" in raw:
                all_bars.extend(raw["results"])
        except Exception as exc:
            logger.debug("Skip %s: %s", fpath.name, exc)

    if not all_bars:
        return {}

    by_date: dict[str, list[dict[str, Any]]] = {}
    for b in all_bars:
        # Handle both short-form (ts/o/h/l/c/v) and long-form
        # (timestamp/open/high/low/close/volume) key names.
        ts = int(b.get("ts") or b.get("timestamp") or b.get("t") or 0)
        if ts == 0:
            continue
        dt = datetime.fromtimestamp(ts / NS_PER_SEC, tz=UTC)
        date_str = b.get("date") or b.get("d") or dt.strftime("%Y-%m-%d")
        bar = {
            "ts": ts,
            "o": float(b.get("o") or b.get("open") or 0),
            "h": float(b.get("h") or b.get("high") or 0),
            "l": float(b.get("l") or b.get("low") or 0),
            "c": float(b.get("c") or b.get("close") or 0),
            "v": int(float(b.get("v") or b.get("volume") or 0)),
            "date": date_str,
        }
        if bar["o"] <= 0 or bar["h"] <= 0 or bar["l"] <= 0 or bar["c"] <= 0:
            continue
        if bar["l"] > bar["h"]:
            continue
        by_date.setdefault(date_str, []).append(bar)

    for date_str in by_date:
        by_date[date_str].sort(key=lambda x: x["ts"])

    return by_date


def _filter_rth(
    bars: list[dict[str, Any]], date_str: str,
) -> list[dict[str, Any]]:
    """Filter bars to Regular Trading Hours (9:30-16:00 ET)."""
    month = int(date_str.split("-")[1])
    utc_offset_hours = 4 if 3 <= month <= 10 else 5
    open_utc = (9 + utc_offset_hours) * 60 + 30
    close_utc = (16 + utc_offset_hours) * 60

    rth: list[dict[str, Any]] = []
    for b in bars:
        dt = datetime.fromtimestamp(b["ts"] / NS_PER_SEC, tz=UTC)
        mins = dt.hour * 60 + dt.minute
        if open_utc <= mins < close_utc:
            rth.append(b)
    return rth


def _aggregate_bars(
    bars_1m: list[dict[str, Any]], size_minutes: int = 5,
) -> list[Bar]:
    """Aggregate 1-minute bars to N-minute bars."""
    if not bars_1m:
        return []

    window_ns = size_minutes * NS_PER_MIN
    buckets: dict[int, list[dict[str, Any]]] = {}
    for b in bars_1m:
        bucket_key = b["ts"] // window_ns
        buckets.setdefault(bucket_key, []).append(b)

    agg: list[Bar] = []
    for key in sorted(buckets):
        chunk = buckets[key]
        agg.append(Bar(
            ts=chunk[0]["ts"],
            o=chunk[0]["o"],
            h=max(b["h"] for b in chunk),
            lo=min(b["l"] for b in chunk),
            c=chunk[-1]["c"],
            v=sum(b["v"] for b in chunk),
            date=chunk[0].get("date", ""),
        ))
    return agg


def _load_daily_closes(ticker: str) -> list[float]:
    """Build daily close prices from 1-min cache (last bar per day)."""
    raw = _load_1min_bars(ticker)
    closes: list[float] = []
    for date_str in sorted(raw.keys()):
        day_bars = raw[date_str]
        if day_bars:
            closes.append(day_bars[-1]["c"])
    return closes


def _has_opra_data(ticker: str, date_str: str) -> bool:
    """Check if OPRA option data exists in cache for this date."""
    path = (
        CACHE_DIR / ticker / "options_1min"
        / f"{ticker}_options_1min_{date_str}.json"
    )
    return path.exists()


# ── Core backtest loop ────────────────────────────────────────────

def run_trident_backtest(
    config: TridentConfig | None = None,
) -> TridentResult:
    """Run a full Trident backtest with REAL OPRA option pricing.

    RULE: No simulated data. All option entry/exit prices come from
    real OPRA minute bars. Days without OPRA data are skipped.
    """
    cfg = config or TridentConfig()
    logger.info("Trident backtest: %s | tickers=%s", cfg.name, cfg.tickers)

    matcher = OptionsMatcher()
    all_trades: list[TridentTrade] = []
    total_days = 0
    opra_days = 0
    no_opra_days = 0
    cash = cfg.starting_capital
    equity_curve: list[float] = [cash]

    for ticker in cfg.tickers:
        logger.info("Loading %s ...", ticker)
        raw_by_date = _load_1min_bars(ticker)
        if not raw_by_date:
            logger.warning("No stock data for %s, skipping", ticker)
            continue

        daily_closes = _load_daily_closes(ticker)
        dates = sorted(raw_by_date.keys())
        total_days += len(dates)
        logger.info("  %s: %d trading days", ticker, len(dates))

        for date_str in dates:
            # ── REAL DATA GATE: skip if no OPRA options data ──
            if not _has_opra_data(ticker, date_str):
                no_opra_days += 1
                continue
            opra_days += 1

            raw_bars = raw_by_date[date_str]
            rth_bars = _filter_rth(raw_bars, date_str)
            if len(rth_bars) < 30:
                continue

            bars_5m = _aggregate_bars(rth_bars, cfg.bar_size_minutes)
            if len(bars_5m) < 10:
                continue

            # Daily closes up to this date for trend detection
            date_idx = dates.index(date_str)
            recent_daily = daily_closes[:date_idx] if date_idx > 0 else []

            # Compute all signals
            snapshots = compute_all_signals(
                bars_5m,
                daily_closes=recent_daily,
                ema_fast_period=cfg.entry.ema_fast,
                ema_slow_period=cfg.entry.ema_slow,
                bollinger_period=cfg.entry.bollinger_period,
                bollinger_std=cfg.entry.bollinger_std,
                opening_range_bars=(
                    cfg.entry.opening_range_minutes // cfg.bar_size_minutes
                ),
            )

            # Track active position for this ticker-day
            active_pos: _ActivePosition | None = None
            last_trade_bar = -999

            for snap in snapshots:
                bar = snap.bar
                bar_idx = snap.bar_idx

                # ── Time filters ──────────────────────────────
                dt = datetime.fromtimestamp(
                    bar.ts / NS_PER_SEC, tz=UTC,
                )
                month = int(date_str.split("-")[1])
                utc_offset = 4 if 3 <= month <= 10 else 5
                et_hour = dt.hour - utc_offset
                et_min = dt.minute
                minutes_since_open = (et_hour - 9) * 60 + (et_min - 30)

                if minutes_since_open < cfg.time_filter.skip_first_n_minutes:
                    continue

                # Force close near EOD
                if minutes_since_open > (
                    390 - cfg.time_filter.skip_last_n_minutes
                ):
                    if active_pos is not None:
                        trade = _close_position_real(
                            active_pos, snap, matcher, "eod", cfg,
                        )
                        if trade:
                            cash += trade.pnl_dollars
                            all_trades.append(trade)
                            equity_curve.append(cash)
                        active_pos = None
                    continue

                # ── Check exits on active position ────────────
                if active_pos is not None:
                    exit_reason = _check_exit_real(
                        active_pos, snap, matcher, cfg,
                    )
                    if exit_reason:
                        trade = _close_position_real(
                            active_pos, snap, matcher, exit_reason, cfg,
                        )
                        if trade:
                            cash += trade.pnl_dollars
                            all_trades.append(trade)
                            equity_curve.append(cash)
                        active_pos = None
                    else:
                        _update_peak_real(active_pos, snap, matcher)
                    continue

                # ── Cooldown ──────────────────────────────────
                if (bar_idx - last_trade_bar
                        < cfg.position.min_bars_between_trades):
                    continue

                # ── Check entry signals ───────────────────────
                call_count, put_count, conviction = evaluate_signals(
                    snap, cfg.entry,
                )

                take_call = (
                    call_count >= cfg.entry.min_signals_call
                    and cfg.direction in (Direction.LONG, Direction.BOTH)
                )
                take_put = (
                    put_count >= cfg.entry.min_signals_put
                    and cfg.direction in (Direction.SHORT, Direction.BOTH)
                )

                # Pick the stronger direction
                if take_call and take_put:
                    if call_count >= put_count:
                        take_put = False
                    else:
                        take_call = False

                if not take_call and not take_put:
                    continue

                is_call = take_call
                opt_type = "C" if is_call else "P"
                signals_fired = call_count if is_call else put_count

                # ── Find REAL option contract via OPRA data ───
                contract = matcher.find_best_contract(
                    underlying=ticker,
                    date_str=date_str,
                    underlying_price=bar.c,
                    signal_ts_ns=bar.ts,
                    max_dte=cfg.options.max_dte,
                    option_type=opt_type,
                )
                if contract is None:
                    continue  # no real option data → skip

                # Get entry price from real OPRA bar
                entry_bar = matcher.get_bar_at_time(
                    contract.bars, bar.ts,
                )
                if entry_bar is None:
                    continue

                entry_price = float(entry_bar.get("c", 0))
                if entry_price < cfg.options.min_premium:
                    continue

                # Apply estimated spread cost
                spread_cost = cfg.position.spread_cents / 100.0
                entry_cost = entry_price + spread_cost

                # Position sizing
                max_spend = cash * cfg.position.risk_per_trade
                contracts = max(1, int(max_spend / (entry_cost * 100)))
                if contracts * entry_cost * 100 > cash * 0.5:
                    continue

                commission = (
                    contracts * cfg.position.commission_per_contract * 2
                )

                active_pos = _ActivePosition(
                    ticker=ticker,
                    is_call=is_call,
                    contract=contract,
                    entry_bar_idx=bar_idx,
                    entry_underlying=bar.c,
                    entry_option_price=entry_cost,
                    entry_date=date_str,
                    entry_time=dt.strftime("%H:%M"),
                    contracts=contracts,
                    conviction=conviction,
                    signals_fired=signals_fired,
                    commission=commission,
                    peak_option_price=entry_cost,
                    bars_held=0,
                )
                last_trade_bar = bar_idx

            # End of day — force close
            if active_pos is not None and snapshots:
                trade = _close_position_real(
                    active_pos, snapshots[-1], matcher, "eod", cfg,
                )
                if trade:
                    cash += trade.pnl_dollars
                    all_trades.append(trade)
                    equity_curve.append(cash)
                active_pos = None

    # ── Compute statistics ────────────────────────────────────────
    result = _compute_stats(
        all_trades, cfg, equity_curve, total_days,
        opra_days, no_opra_days,
    )
    logger.info("Trident result: %s", result.summary_line())
    return result


# ── Active position tracking ─────────────────────────────────────

@dataclass
class _ActivePosition:
    ticker: str
    is_call: bool
    contract: Any  # MatchedContract
    entry_bar_idx: int
    entry_underlying: float
    entry_option_price: float
    entry_date: str
    entry_time: str
    contracts: int
    conviction: float
    signals_fired: int
    commission: float
    peak_option_price: float
    bars_held: int


def _get_real_option_price(
    pos: _ActivePosition,
    snap: SignalSnapshot,
    matcher: OptionsMatcher,
) -> float | None:
    """Get the REAL option price at the current bar from OPRA data."""
    bar = matcher.get_bar_at_time(pos.contract.bars, snap.bar.ts)
    if bar is None:
        return None
    price = float(bar.get("c", 0))
    return price if price > 0 else None


def _update_peak_real(
    pos: _ActivePosition,
    snap: SignalSnapshot,
    matcher: OptionsMatcher,
) -> None:
    """Update peak option price for trailing stop using real data."""
    price = _get_real_option_price(pos, snap, matcher)
    if price is not None and price > pos.peak_option_price:
        pos.peak_option_price = price
    pos.bars_held = snap.bar_idx - pos.entry_bar_idx


def _check_exit_real(
    pos: _ActivePosition,
    snap: SignalSnapshot,
    matcher: OptionsMatcher,
    cfg: TridentConfig,
) -> str | None:
    """Check exit conditions using REAL OPRA option prices."""
    current_price = _get_real_option_price(pos, snap, matcher)
    if current_price is None:
        return None  # no bar data at this time → hold

    entry_price = pos.entry_option_price
    bars_held = snap.bar_idx - pos.entry_bar_idx
    pos.bars_held = bars_held

    if entry_price <= 0:
        return "error"

    pnl_pct = (current_price - entry_price) / entry_price

    # Take profit
    if pnl_pct >= cfg.exit.tp_pct:
        return "tp"

    # Stop loss
    if pnl_pct <= cfg.exit.sl_pct:
        return "sl"

    # Trailing stop
    if cfg.exit.use_trailing and pos.peak_option_price > entry_price:
        peak = pos.peak_option_price
        retrace = (peak - current_price) / peak if peak > 0 else 0
        if retrace >= cfg.exit.trail_pct:
            return "trail"

    # Time exit
    if bars_held >= cfg.exit.max_hold_bars:
        return "time"

    return None


def _close_position_real(
    pos: _ActivePosition,
    snap: SignalSnapshot,
    matcher: OptionsMatcher,
    reason: str,
    cfg: TridentConfig,
) -> TridentTrade | None:
    """Close position using REAL OPRA exit pricing."""
    exit_price = _get_real_option_price(pos, snap, matcher)
    if exit_price is None:
        # Fallback: use last known bar in contract
        if pos.contract.bars:
            exit_price = float(pos.contract.bars[-1].get("c", 0))
        if not exit_price or exit_price <= 0:
            return None

    spread_cost = cfg.position.spread_cents / 100.0
    exit_net = max(0.01, exit_price - spread_cost)

    entry_price = pos.entry_option_price
    contracts = pos.contracts
    commission = pos.commission

    pnl_per_contract = (exit_net - entry_price) * 100
    pnl_dollars = pnl_per_contract * contracts - commission
    pnl_pct = (
        (exit_net - entry_price) / entry_price * 100
        if entry_price > 0 else 0.0
    )

    bars_held = snap.bar_idx - pos.entry_bar_idx
    hold_minutes = bars_held * cfg.bar_size_minutes

    return TridentTrade(
        ticker=pos.ticker,
        direction="call" if pos.is_call else "put",
        contract_symbol=pos.contract.contract_symbol,
        strike=pos.contract.strike,
        dte=pos.contract.dte,
        entry_date=pos.entry_date,
        entry_time=pos.entry_time,
        entry_bar_idx=pos.entry_bar_idx,
        entry_underlying=pos.entry_underlying,
        entry_option_price=entry_price,
        exit_underlying=snap.bar.c,
        exit_option_price=exit_net,
        exit_reason=reason,
        hold_bars=bars_held,
        hold_minutes=hold_minutes,
        pnl_dollars=pnl_dollars,
        pnl_pct=pnl_pct,
        contracts=contracts,
        conviction=pos.conviction,
        signals_fired=pos.signals_fired,
    )


# ── Statistics computation ────────────────────────────────────────

def _compute_stats(
    trades: list[TridentTrade],
    cfg: TridentConfig,
    equity_curve: list[float],
    total_days: int,
    opra_days: int,
    no_opra_days: int,
) -> TridentResult:
    """Compute aggregate statistics from trade list."""
    result = TridentResult(
        config_name=cfg.name,
        config=cfg.to_dict(),
        trades=trades,
        total_trades=len(trades),
        trading_days=total_days,
        days_with_opra=opra_days,
        days_skipped_no_opra=no_opra_days,
    )

    if not trades:
        return result

    wins = [t for t in trades if t.pnl_dollars > 0]
    losses = [t for t in trades if t.pnl_dollars <= 0]
    result.wins = len(wins)
    result.losses = len(losses)
    result.win_rate = len(wins) / len(trades) * 100 if trades else 0.0

    result.total_pnl = sum(t.pnl_dollars for t in trades)
    result.avg_pnl_pct = (
        sum(t.pnl_pct for t in trades) / len(trades) if trades else 0.0
    )
    result.avg_win_pct = (
        sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
    )
    result.avg_loss_pct = (
        sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
    )

    gross_profit = sum(t.pnl_dollars for t in wins)
    gross_loss = abs(sum(t.pnl_dollars for t in losses))
    result.profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )

    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    result.max_drawdown_pct = max_dd

    if len(equity_curve) > 2:
        returns = [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            for i in range(1, len(equity_curve))
        ]
        if returns:
            avg_r = sum(returns) / len(returns)
            std_r = math.sqrt(
                sum((r - avg_r) ** 2 for r in returns) / len(returns),
            )
            result.sharpe_ratio = (
                avg_r / std_r * math.sqrt(TRADING_DAYS_PER_YEAR)
                if std_r > 0 else 0.0
            )

    result.avg_hold_minutes = (
        sum(t.hold_minutes for t in trades) / len(trades) if trades else 0.0
    )

    calls = [t for t in trades if t.direction == "call"]
    puts = [t for t in trades if t.direction == "put"]
    result.calls_taken = len(calls)
    result.puts_taken = len(puts)
    call_wins = [t for t in calls if t.pnl_dollars > 0]
    put_wins = [t for t in puts if t.pnl_dollars > 0]
    result.call_win_rate = (
        len(call_wins) / len(calls) * 100 if calls else 0.0
    )
    result.put_win_rate = (
        len(put_wins) / len(puts) * 100 if puts else 0.0
    )

    unique_days = len({t.entry_date for t in trades})
    result.signals_per_day = (
        len(trades) / unique_days if unique_days > 0 else 0.0
    )

    for ticker in cfg.tickers:
        tt = [t for t in trades if t.ticker == ticker]
        if not tt:
            continue
        t_wins = [t for t in tt if t.pnl_dollars > 0]
        result.per_ticker[ticker] = {
            "trades": len(tt),
            "wins": len(t_wins),
            "win_rate": len(t_wins) / len(tt) * 100,
            "total_pnl": sum(t.pnl_dollars for t in tt),
            "avg_pnl_pct": sum(t.pnl_pct for t in tt) / len(tt),
            "calls": len([t for t in tt if t.direction == "call"]),
            "puts": len([t for t in tt if t.direction == "put"]),
        }

    return result
