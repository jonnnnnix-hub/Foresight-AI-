#!/usr/bin/env python3
"""Trident Lotto Research — Volume/Flow Anomaly Scanner.

Tests volume-based anomaly signals for lotto plays:
  - Massive volume spike (3x, 5x, 10x average)
  - Volume climax (highest volume bar of the day)
  - Volume dry-up then expansion (squeeze → release)
  - Consecutive volume acceleration (each bar higher than prior)
  - Volume divergence (price down but volume surging = accumulation)
  - Opening volume burst (first 5 min > 3x avg daily first-5-min)

All priced with REAL OPRA data only. Deep OTM options for lotto sizing.
"""

import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.backtest.options_matcher import OptionsMatcher
from flowedge.scanner.backtest.trident.backtester import (
    NS_PER_SEC,
    _aggregate_bars,
    _filter_rth,
    _has_opra_data,
    _load_1min_bars,
)
from flowedge.scanner.backtest.trident.signals import Bar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trident.lotto.flow")
logging.getLogger("flowedge").setLevel(logging.WARNING)

TICKERS = ["SPY", "QQQ", "IWM"]


def detect_volume_signals(bars: list[Bar], idx: int) -> list[tuple[str, str]]:
    """Detect volume anomaly signals at bar index. Returns [(signal_name, direction)]."""
    if idx < 10:
        return []
    signals = []
    bar = bars[idx]
    prev_vols = [b.v for b in bars[max(0, idx - 10):idx]]
    avg_vol = sum(prev_vols) / len(prev_vols) if prev_vols else 1

    # 1. Volume spike levels
    if avg_vol > 0:
        ratio = bar.v / avg_vol
        if ratio >= 10:
            signals.append(("vol_spike_10x", "bull" if bar.c > bar.o else "bear"))
        elif ratio >= 5:
            signals.append(("vol_spike_5x", "bull" if bar.c > bar.o else "bear"))
        elif ratio >= 3:
            signals.append(("vol_spike_3x", "bull" if bar.c > bar.o else "bear"))

    # 2. Volume climax (highest of the day so far)
    day_vols = [b.v for b in bars[:idx + 1]]
    if bar.v == max(day_vols) and bar.v > avg_vol * 2 and idx > 5:
        signals.append(("vol_climax", "bull" if bar.c > bar.o else "bear"))

    # 3. Volume dry-up then expansion (squeeze)
    if idx >= 6:
        recent_3 = [b.v for b in bars[idx - 3:idx]]
        prior_3 = [b.v for b in bars[idx - 6:idx - 3]]
        avg_recent = sum(recent_3) / 3
        avg_prior = sum(prior_3) / 3
        if avg_prior > 0 and avg_recent < avg_prior * 0.5 and bar.v > avg_prior * 2:
            signals.append(("vol_squeeze_release", "bull" if bar.c > bar.o else "bear"))

    # 4. Consecutive volume acceleration (3 bars each higher)
    if idx >= 3 and bars[idx].v > bars[idx - 1].v > bars[idx - 2].v > bars[idx - 3].v:
        trend = "bull" if bars[idx].c > bars[idx - 2].c else "bear"
        signals.append(("vol_accel_3", trend))

    # 5. Volume divergence (price dropping but volume surging = accumulation)
    if idx >= 3:
        price_down = bars[idx].c < bars[idx - 2].c
        vol_up = bars[idx].v > bars[idx - 2].v * 1.5
        if price_down and vol_up:
            signals.append(("vol_divergence_bull", "bull"))
        price_up = bars[idx].c > bars[idx - 2].c
        vol_up2 = bars[idx].v > bars[idx - 2].v * 1.5
        if price_up and vol_up2 and bars[idx].c < bars[idx].o:
            # Price up but candle is red and volume surging = distribution
            signals.append(("vol_divergence_bear", "bear"))

    return signals


def run_flow_backtest(
    signal_filter: str,  # "all", specific signal name, or "combined_2"
    tp_pct: float = 0.50,
    sl_pct: float = -0.40,
    max_hold: int = 6,
    min_confluence: int = 1,
) -> dict[str, Any]:
    matcher = OptionsMatcher()
    trades: list[dict] = []
    cash = 25_000.0
    spread = 3.0

    for ticker in TICKERS:
        raw = _load_1min_bars(ticker)
        if not raw:
            continue
        for date_str in sorted(raw.keys()):
            if not _has_opra_data(ticker, date_str):
                continue
            rth = _filter_rth(raw[date_str], date_str)
            if len(rth) < 30:
                continue
            bars = _aggregate_bars(rth, 5)
            if len(bars) < 10:
                continue

            active = None
            for idx in range(5, len(bars)):
                bar = bars[idx]
                dt = datetime.fromtimestamp(bar.ts / NS_PER_SEC, tz=UTC)
                month = int(date_str.split("-")[1])
                off = 4 if 3 <= month <= 10 else 5
                mins = (dt.hour - off - 9) * 60 + (dt.minute - 30)
                if mins < 3 or mins > 360:
                    if active and mins > 375:
                        ob = OptionsMatcher.get_bar_at_time(active["contract"].bars, bar.ts)
                        if ob:
                            pnl = _close(active, float(ob.get("c", 0)), "eod", trades, spread)
                            cash += pnl
                        active = None
                    continue

                if active:
                    ob = OptionsMatcher.get_bar_at_time(active["contract"].bars, bar.ts)
                    if ob:
                        cp = float(ob.get("c", 0))
                        if cp > active["peak"]:
                            active["peak"] = cp
                        active["held"] += 1
                        ep = active["ep"]
                        pct = (cp - ep) / ep if ep > 0 else 0
                        reason = None
                        if pct >= tp_pct:
                            reason = "tp"
                        elif pct <= sl_pct:
                            reason = "sl"
                        elif active["held"] >= max_hold:
                            reason = "time"
                        elif active["peak"] > ep and (active["peak"] - cp) / active["peak"] >= 0.40:
                            reason = "trail"
                        if reason:
                            cash += _close(active, cp, reason, trades, spread)
                            active = None
                    continue

                sigs = detect_volume_signals(bars, idx)
                if not sigs:
                    continue

                # Filter
                if signal_filter == "all":
                    pass
                elif signal_filter == "combined_2":
                    if len(sigs) < min_confluence:
                        continue
                else:
                    sigs = [(n, d) for n, d in sigs if n == signal_filter]
                if not sigs:
                    continue

                # Direction: majority vote
                bull = sum(1 for _, d in sigs if d == "bull")
                bear = sum(1 for _, d in sigs if d == "bear")
                is_call = bull >= bear
                opt_type = "C" if is_call else "P"

                contract = matcher.find_best_contract(
                    ticker, date_str, bar.c, bar.ts, max_dte=2, option_type=opt_type,
                )
                if not contract:
                    continue
                eb = OptionsMatcher.get_bar_at_time(contract.bars, bar.ts)
                if not eb:
                    continue
                ep = float(eb.get("c", 0))
                if ep < 0.10:
                    continue
                ep += spread / 100

                qty = max(1, int(cash * 0.02 / (ep * 100)))
                active = {
                    "ticker": ticker, "is_call": is_call, "contract": contract,
                    "ep": ep, "peak": ep, "qty": qty, "date": date_str, "held": 0,
                    "sig": signal_filter,
                }

            if active and bars:
                ob = OptionsMatcher.get_bar_at_time(active["contract"].bars, bars[-1].ts)
                if ob:
                    cash += _close(active, float(ob.get("c", 0)), "eod", trades, spread)
                active = None

    return _stats(signal_filter, trades, tp_pct, sl_pct, max_hold)


def _close(pos, exit_p, reason, trades, spread):
    en = max(0.01, exit_p - spread / 100)
    pnl = (en - pos["ep"]) * 100 * pos["qty"] - pos["qty"] * 1.0
    trades.append({
        "ticker": pos["ticker"], "dir": "call" if pos["is_call"] else "put",
        "pnl": pnl, "pnl_pct": (en - pos["ep"]) / pos["ep"] * 100 if pos["ep"] > 0 else 0,
        "reason": reason, "held": pos["held"], "date": pos["date"], "sig": pos["sig"],
    })
    return pnl


def _stats(name, trades, tp, sl, hold):
    if not trades:
        return {"name": name, "trades": 0, "wr": 0, "pnl": 0, "pf": 0}
    w = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in w)
    gl = abs(sum(t["pnl"] for t in losses))
    return {
        "name": name, "trades": len(trades), "wins": len(w),
        "wr": len(w) / len(trades) * 100, "pnl": sum(t["pnl"] for t in trades),
        "avg_pnl_pct": sum(t["pnl_pct"] for t in trades) / len(trades),
        "pf": gw / gl if gl > 0 else float("inf"),
        "avg_hold": sum(t["held"] for t in trades) / len(trades) * 5,
        "tp_hits": len([t for t in trades if t["reason"] == "tp"]),
        "tp": tp, "sl": sl, "hold": hold * 5,
    }


def main():
    signal_types = [
        "vol_spike_3x", "vol_spike_5x", "vol_spike_10x",
        "vol_climax", "vol_squeeze_release", "vol_accel_3",
        "vol_divergence_bull", "vol_divergence_bear",
        "all", "combined_2",
    ]

    exits = [
        (0.50, -0.40, 6),   # standard lotto
        (1.00, -0.50, 6),   # big target
        (0.30, -0.30, 4),   # quick scalp
        (2.00, -0.60, 6),   # moon shot
    ]

    results = []
    total = len(signal_types) * len(exits)
    i = 0
    for sig in signal_types:
        for tp, sl, h in exits:
            i += 1
            t0 = time.time()
            minc = 2 if sig == "combined_2" else 1
            r = run_flow_backtest(sig, tp, sl, h, minc)
            elapsed = time.time() - t0
            results.append(r)
            logger.info(
                "[%d/%d] %.0fs | %s TP=%.0f%% SL=%.0f%%: %d trades, WR=%.1f%%, P&L=$%.0f, PF=%.2f",
                i, total, elapsed, sig, tp * 100, abs(sl) * 100,
                r["trades"], r["wr"], r["pnl"], r["pf"],
            )

    ranked = sorted(results, key=lambda r: r["pnl"] if r["trades"] >= 30 else -999999, reverse=True)
    logger.info("")
    logger.info("=" * 70)
    logger.info("VOLUME/FLOW LOTTO RESULTS — TOP 15")
    logger.info("=" * 70)
    for i, r in enumerate(ranked[:15], 1):
        logger.info(
            "%2d. %s (TP=%.0f%% SL=%.0f%% H=%dm): "
            "%d trades, WR=%.1f%%, P&L=$%.0f, PF=%.2f, AvgHold=%.0fm",
            i, r["name"], r.get("tp", 0) * 100,
            abs(r.get("sl", 0)) * 100,
            r.get("hold", 0), r["trades"], r["wr"],
            r["pnl"], r["pf"], r.get("avg_hold", 0),
        )

    out = Path("data/trident_backtest_results")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"lotto_volume_flow_{ts}.json").write_text(
        json.dumps({"results": ranked}, indent=2, default=str),
    )


if __name__ == "__main__":
    main()
