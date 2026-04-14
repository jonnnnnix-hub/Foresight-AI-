#!/usr/bin/env python3
"""Trident Lotto Research — Candlestick Pattern Scanner.

Tests candlestick reversal/continuation patterns on 1-min and 5-min bars
as entry signals for 0-30 min lotto plays on deep OTM options.

Patterns tested:
  - Hammer / Inverted Hammer (reversal)
  - Engulfing (bullish/bearish)
  - Three-bar reversal (3 reds then green, or vice versa)
  - Doji at support/resistance (indecision → breakout)
  - Large body candle after consolidation (expansion)
  - Pin bar (long wick rejection)

Targets deep OTM options (delta 0.10-0.30) for max leverage on lottos.
"""

import json
import logging
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.backtest.trident.backtester import (
    NS_PER_SEC,
    TridentResult,
    TridentTrade,
    _aggregate_bars,
    _filter_rth,
    _has_opra_data,
    _load_1min_bars,
    _load_daily_closes,
)
from flowedge.scanner.backtest.trident.config import CACHE_DIR, TridentConfig
from flowedge.scanner.backtest.trident.signals import Bar, _cumulative_vwap
from flowedge.scanner.backtest.options_matcher import OptionsMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trident.lotto.candle")
logging.getLogger("flowedge").setLevel(logging.WARNING)

TICKERS = ["SPY", "QQQ", "IWM"]


# ── Candlestick pattern detection ────────────────────────────────

def is_hammer(bar: Bar, prev_bars: list[Bar]) -> bool:
    """Hammer: small body at top, long lower wick (2x body)."""
    body = abs(bar.c - bar.o)
    lower_wick = min(bar.o, bar.c) - bar.lo
    upper_wick = bar.h - max(bar.o, bar.c)
    if body <= 0:
        return False
    return lower_wick >= 2 * body and upper_wick < body * 0.5


def is_inverted_hammer(bar: Bar, prev_bars: list[Bar]) -> bool:
    """Inverted hammer: small body at bottom, long upper wick."""
    body = abs(bar.c - bar.o)
    upper_wick = bar.h - max(bar.o, bar.c)
    lower_wick = min(bar.o, bar.c) - bar.lo
    if body <= 0:
        return False
    return upper_wick >= 2 * body and lower_wick < body * 0.5


def is_bullish_engulfing(bars: list[Bar], idx: int) -> bool:
    """Current green candle body fully engulfs prior red candle body."""
    if idx < 1:
        return False
    cur, prev = bars[idx], bars[idx - 1]
    if cur.c <= cur.o or prev.c >= prev.o:
        return False  # cur must be green, prev must be red
    return cur.o <= prev.c and cur.c >= prev.o


def is_bearish_engulfing(bars: list[Bar], idx: int) -> bool:
    """Current red candle body fully engulfs prior green candle body."""
    if idx < 1:
        return False
    cur, prev = bars[idx], bars[idx - 1]
    if cur.c >= cur.o or prev.c <= prev.o:
        return False  # cur must be red, prev must be green
    return cur.o >= prev.c and cur.c <= prev.o


def is_three_bar_reversal_bull(bars: list[Bar], idx: int) -> bool:
    """3 consecutive red bars followed by a green bar."""
    if idx < 3:
        return False
    return (
        bars[idx].c > bars[idx].o  # current green
        and bars[idx - 1].c < bars[idx - 1].o  # prev red
        and bars[idx - 2].c < bars[idx - 2].o
        and bars[idx - 3].c < bars[idx - 3].o
    )


def is_three_bar_reversal_bear(bars: list[Bar], idx: int) -> bool:
    """3 consecutive green bars followed by a red bar."""
    if idx < 3:
        return False
    return (
        bars[idx].c < bars[idx].o  # current red
        and bars[idx - 1].c > bars[idx - 1].o  # prev green
        and bars[idx - 2].c > bars[idx - 2].o
        and bars[idx - 3].c > bars[idx - 3].o
    )


def is_pin_bar(bar: Bar) -> tuple[bool, str]:
    """Pin bar: one long wick, tiny body. Returns (is_pin, direction)."""
    body = abs(bar.c - bar.o)
    upper_wick = bar.h - max(bar.o, bar.c)
    lower_wick = min(bar.o, bar.c) - bar.lo
    total_range = bar.h - bar.lo
    if total_range <= 0:
        return False, ""
    body_pct = body / total_range
    if body_pct > 0.25:
        return False, ""  # body too large
    if lower_wick > upper_wick * 2:
        return True, "bull"  # rejection of lows → bullish
    if upper_wick > lower_wick * 2:
        return True, "bear"  # rejection of highs → bearish
    return False, ""


def is_expansion_candle(bars: list[Bar], idx: int) -> tuple[bool, str]:
    """Large body candle after 3+ bars of tight consolidation."""
    if idx < 4:
        return False, ""
    # Check prior 3 bars for consolidation (small range)
    prior = bars[idx - 3 : idx]
    avg_range = sum(abs(b.h - b.lo) for b in prior) / len(prior)
    cur_range = abs(bars[idx].h - bars[idx].lo)
    cur_body = abs(bars[idx].c - bars[idx].o)
    if avg_range <= 0:
        return False, ""
    if cur_range < avg_range * 2.0 or cur_body < avg_range * 1.5:
        return False, ""
    direction = "bull" if bars[idx].c > bars[idx].o else "bear"
    return True, direction


# ── Backtest runner ───────────────────────────────────────────────

def run_candlestick_backtest(
    pattern_name: str,
    tp_pct: float = 0.50,
    sl_pct: float = -0.40,
    max_hold_bars: int = 6,  # 30 min at 5-min bars
    dte_max: int = 2,
    delta_min: float = 0.10,
    delta_max: float = 0.30,
) -> dict[str, Any]:
    """Run a backtest for one candlestick pattern config."""
    matcher = OptionsMatcher()
    trades: list[dict[str, Any]] = []
    cash = 25_000.0
    spread_cents = 3.0

    for ticker in TICKERS:
        raw_by_date = _load_1min_bars(ticker)
        if not raw_by_date:
            continue
        dates = sorted(raw_by_date.keys())

        for date_str in dates:
            if not _has_opra_data(ticker, date_str):
                continue
            raw = raw_by_date[date_str]
            rth = _filter_rth(raw, date_str)
            if len(rth) < 30:
                continue
            bars_5m = _aggregate_bars(rth, 5)
            if len(bars_5m) < 10:
                continue

            # Compute VWAP for context
            vwap_vals = _cumulative_vwap(bars_5m)
            for i, b in enumerate(bars_5m):
                b.vwap = vwap_vals[i]

            active_pos = None

            for idx in range(4, len(bars_5m)):
                bar = bars_5m[idx]

                # Time filter: skip first 5 min, no entries in last 30 min
                dt = datetime.fromtimestamp(bar.ts / NS_PER_SEC, tz=UTC)
                month = int(date_str.split("-")[1])
                utc_off = 4 if 3 <= month <= 10 else 5
                mins = (dt.hour - utc_off - 9) * 60 + (dt.minute - 30)
                if mins < 5 or mins > 360:
                    if active_pos and mins > 375:
                        # Force close EOD
                        exit_bar = matcher.get_bar_at_time(
                            active_pos["contract"].bars, bar.ts,
                        )
                        if exit_bar:
                            exit_p = float(exit_bar.get("c", 0))
                            _close_trade(active_pos, exit_p, "eod", trades, cash, spread_cents)
                        active_pos = None
                    continue

                # Check exit on active position
                if active_pos is not None:
                    opt_bar = matcher.get_bar_at_time(
                        active_pos["contract"].bars, bar.ts,
                    )
                    if opt_bar:
                        cur_p = float(opt_bar.get("c", 0))
                        if cur_p > active_pos["peak"]:
                            active_pos["peak"] = cur_p
                        entry_p = active_pos["entry_price"]
                        active_pos["bars_held"] += 1
                        pnl_pct = (cur_p - entry_p) / entry_p if entry_p > 0 else 0

                        reason = None
                        if pnl_pct >= tp_pct:
                            reason = "tp"
                        elif pnl_pct <= sl_pct:
                            reason = "sl"
                        elif active_pos["bars_held"] >= max_hold_bars:
                            reason = "time"
                        # Trailing: 40% retrace from peak
                        elif active_pos["peak"] > entry_p:
                            retrace = (active_pos["peak"] - cur_p) / active_pos["peak"]
                            if retrace >= 0.40:
                                reason = "trail"

                        if reason:
                            cash += _close_trade(active_pos, cur_p, reason, trades, cash, spread_cents)
                            active_pos = None
                    continue

                # ── Pattern detection ─────────────────────────
                is_call = None
                matched = False

                if pattern_name == "hammer":
                    if is_hammer(bar, bars_5m[:idx]):
                        is_call, matched = True, True
                elif pattern_name == "inv_hammer":
                    if is_inverted_hammer(bar, bars_5m[:idx]):
                        is_call, matched = False, True
                elif pattern_name == "bull_engulf":
                    if is_bullish_engulfing(bars_5m, idx):
                        is_call, matched = True, True
                elif pattern_name == "bear_engulf":
                    if is_bearish_engulfing(bars_5m, idx):
                        is_call, matched = False, True
                elif pattern_name == "3bar_bull":
                    if is_three_bar_reversal_bull(bars_5m, idx):
                        is_call, matched = True, True
                elif pattern_name == "3bar_bear":
                    if is_three_bar_reversal_bear(bars_5m, idx):
                        is_call, matched = False, True
                elif pattern_name == "pin_bar":
                    is_pin, pin_dir = is_pin_bar(bar)
                    if is_pin:
                        is_call = pin_dir == "bull"
                        matched = True
                elif pattern_name == "expansion":
                    is_exp, exp_dir = is_expansion_candle(bars_5m, idx)
                    if is_exp:
                        is_call = exp_dir == "bull"
                        matched = True
                elif pattern_name == "all_reversal":
                    # Any reversal pattern
                    if is_hammer(bar, bars_5m[:idx]):
                        is_call, matched = True, True
                    elif is_bullish_engulfing(bars_5m, idx):
                        is_call, matched = True, True
                    elif is_three_bar_reversal_bull(bars_5m, idx):
                        is_call, matched = True, True
                    elif is_bearish_engulfing(bars_5m, idx):
                        is_call, matched = False, True
                    elif is_three_bar_reversal_bear(bars_5m, idx):
                        is_call, matched = False, True
                elif pattern_name == "all_patterns":
                    # Every pattern
                    is_pin, pin_dir = is_pin_bar(bar)
                    is_exp, exp_dir = is_expansion_candle(bars_5m, idx)
                    if is_hammer(bar, bars_5m[:idx]):
                        is_call, matched = True, True
                    elif is_bullish_engulfing(bars_5m, idx):
                        is_call, matched = True, True
                    elif is_three_bar_reversal_bull(bars_5m, idx):
                        is_call, matched = True, True
                    elif is_bearish_engulfing(bars_5m, idx):
                        is_call, matched = False, True
                    elif is_three_bar_reversal_bear(bars_5m, idx):
                        is_call, matched = False, True
                    elif is_pin:
                        is_call = pin_dir == "bull"
                        matched = True
                    elif is_exp:
                        is_call = exp_dir == "bull"
                        matched = True

                if not matched or is_call is None:
                    continue

                # Find deep OTM option for lotto play
                opt_type = "C" if is_call else "P"
                contract = matcher.find_best_contract(
                    ticker, date_str, bar.c, bar.ts,
                    max_dte=dte_max, option_type=opt_type,
                )
                if contract is None:
                    continue

                entry_bar = matcher.get_bar_at_time(contract.bars, bar.ts)
                if entry_bar is None:
                    continue
                entry_p = float(entry_bar.get("c", 0))
                if entry_p < 0.10:  # min $0.10 for lotto
                    continue
                entry_cost = entry_p + spread_cents / 100.0

                # Lotto sizing: 1-2% of equity, small contracts
                max_spend = cash * 0.02
                contracts_qty = max(1, int(max_spend / (entry_cost * 100)))

                active_pos = {
                    "ticker": ticker,
                    "is_call": is_call,
                    "contract": contract,
                    "entry_price": entry_cost,
                    "peak": entry_cost,
                    "contracts": contracts_qty,
                    "entry_date": date_str,
                    "bars_held": 0,
                    "pattern": pattern_name,
                }

            # EOD close
            if active_pos and bars_5m:
                last_bar = bars_5m[-1]
                opt_bar = matcher.get_bar_at_time(
                    active_pos["contract"].bars, last_bar.ts,
                )
                if opt_bar:
                    cash += _close_trade(
                        active_pos, float(opt_bar.get("c", 0)),
                        "eod", trades, cash, spread_cents,
                    )
                active_pos = None

    # Compute stats
    return _compute_results(pattern_name, trades, cash, tp_pct, sl_pct, max_hold_bars)


def _close_trade(pos, exit_price, reason, trades_list, cash, spread_cents):
    exit_net = max(0.01, exit_price - spread_cents / 100.0)
    entry_p = pos["entry_price"]
    pnl = (exit_net - entry_p) * 100 * pos["contracts"] - pos["contracts"] * 1.0
    trades_list.append({
        "ticker": pos["ticker"],
        "direction": "call" if pos["is_call"] else "put",
        "pattern": pos["pattern"],
        "entry_price": entry_p,
        "exit_price": exit_net,
        "pnl": pnl,
        "pnl_pct": (exit_net - entry_p) / entry_p * 100 if entry_p > 0 else 0,
        "reason": reason,
        "bars_held": pos["bars_held"],
        "date": pos["entry_date"],
    })
    return pnl


def _compute_results(name, trades, final_cash, tp, sl, hold):
    if not trades:
        return {"name": name, "trades": 0, "win_rate": 0, "pnl": 0, "pf": 0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = abs(sum(t["pnl"] for t in losses))
    return {
        "name": name,
        "trades": len(trades),
        "wins": len(wins),
        "win_rate": len(wins) / len(trades) * 100,
        "pnl": sum(t["pnl"] for t in trades),
        "avg_pnl_pct": sum(t["pnl_pct"] for t in trades) / len(trades),
        "pf": gross_w / gross_l if gross_l > 0 else float("inf"),
        "avg_hold": sum(t["bars_held"] for t in trades) / len(trades) * 5,
        "calls": len([t for t in trades if t["direction"] == "call"]),
        "puts": len([t for t in trades if t["direction"] == "put"]),
        "tp_hits": len([t for t in trades if t["reason"] == "tp"]),
        "sl_hits": len([t for t in trades if t["reason"] == "sl"]),
        "tp_pct": tp, "sl_pct": sl, "max_hold": hold * 5,
    }


def main():
    patterns = [
        "hammer", "inv_hammer", "bull_engulf", "bear_engulf",
        "3bar_bull", "3bar_bear", "pin_bar", "expansion",
        "all_reversal", "all_patterns",
    ]

    # Also test different exit profiles
    exit_profiles = [
        (0.50, -0.40, 6),   # 50% TP, 40% SL, 30 min hold — standard lotto
        (1.00, -0.50, 6),   # 100% TP, 50% SL — bigger swings
        (0.30, -0.30, 4),   # 30% TP, 30% SL, 20 min — quick scalp
        (2.00, -0.50, 6),   # 200% TP, 50% SL — moon shots
    ]

    results = []
    total = len(patterns) * len(exit_profiles)
    i = 0

    for pattern in patterns:
        for tp, sl, hold in exit_profiles:
            i += 1
            t0 = time.time()
            r = run_candlestick_backtest(pattern, tp, sl, hold)
            elapsed = time.time() - t0
            results.append(r)
            logger.info(
                "[%d/%d] %.0fs | %s (TP=%.0f%% SL=%.0f%% H=%dm): %d trades, WR=%.1f%%, P&L=$%.0f, PF=%.2f",
                i, total, elapsed, pattern, tp * 100, abs(sl) * 100, hold * 5,
                r["trades"], r["win_rate"], r["pnl"], r["pf"],
            )

    # Rank
    ranked = sorted(results, key=lambda r: r["pnl"] if r["trades"] >= 50 else -999999, reverse=True)

    logger.info("")
    logger.info("=" * 70)
    logger.info("CANDLESTICK PATTERN LOTTO RESULTS — TOP 15")
    logger.info("=" * 70)
    for i, r in enumerate(ranked[:15], 1):
        logger.info(
            "%2d. %s (TP=%.0f%% SL=%.0f%% H=%dm): %d trades, WR=%.1f%%, P&L=$%.0f, PF=%.2f, AvgHold=%.0fm, TP hits=%d",
            i, r["name"], r["tp_pct"] * 100, abs(r["sl_pct"]) * 100,
            r["max_hold"], r["trades"], r["win_rate"], r["pnl"],
            r["pf"], r.get("avg_hold", 0), r.get("tp_hits", 0),
        )

    # Save
    out = Path("data/trident_backtest_results")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"lotto_candlestick_{ts}.json").write_text(
        json.dumps({"results": ranked}, indent=2, default=str),
    )


if __name__ == "__main__":
    main()
