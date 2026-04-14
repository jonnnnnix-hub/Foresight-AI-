#!/usr/bin/env python3
"""Trident volume optimization — maximize trade count while maintaining WR.

Strategy:
  1. Test the champion signals (opening_range + macd + ema_cross) at 2-of-3
     with tighter exits to recover WR
  2. Test 4-5 signal combos at 2-of-N and 3-of-N (more entry opportunities)
  3. Sweep exit params on the best high-volume configs
  4. Sweep options params (DTE, delta) for more contract matches

Target: 500+ trades/4yr, WR >= 55%, PF >= 1.2, Sharpe >= 1.5
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flowedge.scanner.backtest.trident.backtester import run_trident_backtest
from flowedge.scanner.backtest.trident.config import (
    Direction,
    EntrySignals,
    ExitParams,
    OptionsParams,
    PositionParams,
    TimeFilter,
    TridentConfig,
)
from flowedge.scanner.backtest.trident.optimizer import score_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trident.vol_opt")

# Suppress noisy debug logs from structlog
logging.getLogger("flowedge").setLevel(logging.WARNING)


def build_configs():
    """Build all configs targeting higher volume."""
    configs = []

    # All signal flags off as base
    OFF = {
        "use_rsi3": False, "use_rsi14": False,
        "use_vwap_position": False, "use_vwap_distance": False,
        "use_ibs": False, "use_volume_spike": False,
        "use_intraday_move": False, "use_prior_bar_color": False,
        "use_ema_cross": False, "use_macd": False,
        "use_bollinger": False, "use_opening_range": False,
        "use_daily_trend": False, "use_sma_micro": False,
    }

    # ── GROUP 1: Champion combo with tighter exits ─────────────
    # 2-of-3 (opening_range + macd + ema_cross) = high volume
    # Tighten exits to recover WR from 54.9% toward 60%+
    champ_sigs = {**OFF, "use_opening_range": True, "use_macd": True, "use_ema_cross": True}

    for tp in [0.15, 0.20, 0.25, 0.30]:
        for sl in [-0.25, -0.30, -0.35]:
            for trail in [0.20, 0.25, 0.30]:
                for hold in [9, 12, 15, 18]:
                    entry = EntrySignals(**{**champ_sigs, "min_signals_call": 2, "min_signals_put": 2})
                    exit_p = ExitParams(tp_pct=tp, sl_pct=sl, trail_pct=trail,
                                        max_hold_bars=hold, use_trailing=True)
                    configs.append(TridentConfig(
                        name=f"champ2of3_tp{int(tp*100)}_sl{int(abs(sl)*100)}_tr{int(trail*100)}_h{hold*5}m",
                        entry=entry, exit=exit_p,
                    ))

    # ── GROUP 2: 4-signal combos at 3-of-4 (more entry points) ─
    four_signal_sets = [
        # Champion + each additional signal
        ("or+macd+ema+vwap", {**OFF, "use_opening_range": True, "use_macd": True,
         "use_ema_cross": True, "use_vwap_position": True}),
        ("or+macd+ema+vol", {**OFF, "use_opening_range": True, "use_macd": True,
         "use_ema_cross": True, "use_volume_spike": True}),
        ("or+macd+ema+rsi3", {**OFF, "use_opening_range": True, "use_macd": True,
         "use_ema_cross": True, "use_rsi3": True}),
        ("or+macd+ema+ibs", {**OFF, "use_opening_range": True, "use_macd": True,
         "use_ema_cross": True, "use_ibs": True}),
        ("or+macd+ema+trend", {**OFF, "use_opening_range": True, "use_macd": True,
         "use_ema_cross": True, "use_daily_trend": True}),
        ("or+macd+ema+intra", {**OFF, "use_opening_range": True, "use_macd": True,
         "use_ema_cross": True, "use_intraday_move": True}),
        ("or+macd+ema+boll", {**OFF, "use_opening_range": True, "use_macd": True,
         "use_ema_cross": True, "use_bollinger": True}),
        ("or+macd+ema+prior", {**OFF, "use_opening_range": True, "use_macd": True,
         "use_ema_cross": True, "use_prior_bar_color": True}),
        # Non-champion combos that might have volume
        ("vwap+rsi3+vol+ema", {**OFF, "use_vwap_position": True, "use_rsi3": True,
         "use_volume_spike": True, "use_ema_cross": True}),
        ("vwap+ibs+macd+trend", {**OFF, "use_vwap_position": True, "use_ibs": True,
         "use_macd": True, "use_daily_trend": True}),
        ("or+vwap+vol+macd", {**OFF, "use_opening_range": True, "use_vwap_position": True,
         "use_volume_spike": True, "use_macd": True}),
        ("ema+macd+trend+vol", {**OFF, "use_ema_cross": True, "use_macd": True,
         "use_daily_trend": True, "use_volume_spike": True}),
    ]

    for name, sigs in four_signal_sets:
        for min_s in [2, 3]:
            entry = EntrySignals(**{**sigs, "min_signals_call": min_s, "min_signals_put": min_s})
            configs.append(TridentConfig(name=f"4sig_{name}_min{min_s}", entry=entry))

    # ── GROUP 3: 5-signal combos at 3-of-5 ────────────────────
    five_signal_sets = [
        ("or+macd+ema+vwap+vol", {**OFF, "use_opening_range": True, "use_macd": True,
         "use_ema_cross": True, "use_vwap_position": True, "use_volume_spike": True}),
        ("or+macd+ema+rsi3+trend", {**OFF, "use_opening_range": True, "use_macd": True,
         "use_ema_cross": True, "use_rsi3": True, "use_daily_trend": True}),
        ("or+macd+ema+ibs+vol", {**OFF, "use_opening_range": True, "use_macd": True,
         "use_ema_cross": True, "use_ibs": True, "use_volume_spike": True}),
        ("or+ema+vwap+trend+vol", {**OFF, "use_opening_range": True, "use_ema_cross": True,
         "use_vwap_position": True, "use_daily_trend": True, "use_volume_spike": True}),
        ("macd+ema+vwap+rsi3+vol", {**OFF, "use_macd": True, "use_ema_cross": True,
         "use_vwap_position": True, "use_rsi3": True, "use_volume_spike": True}),
    ]

    for name, sigs in five_signal_sets:
        for min_s in [2, 3]:
            entry = EntrySignals(**{**sigs, "min_signals_call": min_s, "min_signals_put": min_s})
            configs.append(TridentConfig(name=f"5sig_{name}_min{min_s}", entry=entry))

    # ── GROUP 4: Options params sweep on champion 2-of-3 ──────
    champ_entry_2of3 = EntrySignals(**{**champ_sigs, "min_signals_call": 2, "min_signals_put": 2})
    for dte_max in [2, 3, 5, 7]:
        for d_min, d_max in [(0.15, 0.35), (0.20, 0.45), (0.30, 0.50), (0.10, 0.30)]:
            opts = OptionsParams(min_dte=0, max_dte=dte_max, min_delta=d_min, max_delta=d_max)
            configs.append(TridentConfig(
                name=f"opts_champ2of3_dte{dte_max}_d{int(d_min*100)}-{int(d_max*100)}",
                entry=champ_entry_2of3, options=opts,
            ))

    # ── GROUP 5: Position sizing sweep ────────────────────────
    for risk in [0.02, 0.03, 0.04, 0.05, 0.06]:
        for max_pos in [2, 3, 4]:
            pos = PositionParams(risk_per_trade=risk, max_positions=max_pos, min_bars_between_trades=2)
            configs.append(TridentConfig(
                name=f"pos_champ2of3_r{int(risk*100)}_p{max_pos}",
                entry=champ_entry_2of3, position=pos,
            ))

    # ── GROUP 6: Time window sweep ────────────────────────────
    for skip in [3, 5, 10]:
        for morn, mid, aft in [(True, True, True), (True, True, False), (True, False, True)]:
            tf = TimeFilter(skip_first_n_minutes=skip, use_morning=morn, use_midday=mid, use_afternoon=aft)
            wname = f"{'M' if morn else ''}{'D' if mid else ''}{'A' if aft else ''}"
            configs.append(TridentConfig(
                name=f"time_champ2of3_{wname}_skip{skip}",
                entry=champ_entry_2of3, time_filter=tf,
            ))

    return configs


def volume_score(r):
    """Score optimized for high volume + good WR + positive P&L."""
    if r.total_trades < 50:
        return -1000.0
    if r.win_rate < 50.0:
        return -500.0
    if r.profit_factor < 0.9:
        return -200.0

    # Volume bonus (sweet spot: 500-2000 trades over 4 years)
    vol_score = min(r.total_trades / 2000, 1.0) * 100

    # WR bonus (above 55% gets strong bonus)
    wr = min(r.win_rate / 100, 1.0)
    wr_score = wr * 100
    if r.win_rate >= 55:
        wr_score += 20
    if r.win_rate >= 60:
        wr_score += 20

    # P&L matters — needs to be meaningfully positive
    pnl_score = min(max(r.total_pnl / 50000, 0), 1.0) * 100

    # Profit factor
    pf_score = min(r.profit_factor / 2.0, 1.0) * 100

    # Sharpe
    sharpe_score = min(max(r.sharpe_ratio, 0) / 3.0, 1.0) * 80

    # Drawdown penalty (but more tolerant than default)
    dd_score = max(0, 100 - r.max_drawdown_pct)

    composite = (
        vol_score * 0.25     # volume is king
        + wr_score * 0.20    # but WR still matters
        + pnl_score * 0.20   # absolute P&L
        + pf_score * 0.15    # profit factor
        + sharpe_score * 0.10  # risk-adjusted
        + dd_score * 0.10    # drawdown tolerance
    )
    return composite


def main():
    configs = build_configs()
    logger.info("=" * 70)
    logger.info("TRIDENT VOLUME OPTIMIZATION")
    logger.info("Configs to test: %d", len(configs))
    logger.info("Target: 500+ trades, WR >= 55%%, PF >= 1.2")
    logger.info("=" * 70)

    results = []
    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        try:
            r = run_trident_backtest(cfg)
            results.append(r)
            elapsed = time.time() - t0
            vs = volume_score(r)
            logger.info(
                "[%d/%d] %.0fs | VS=%.1f | %s",
                i, len(configs), elapsed, vs, r.summary_line(),
            )
        except Exception as e:
            logger.error("[%d/%d] FAILED: %s — %s", i, len(configs), cfg.name, e)

    # Rank by volume score
    ranked = sorted(results, key=volume_score, reverse=True)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TOP 20 VOLUME-OPTIMIZED CONFIGS")
    logger.info("=" * 70)
    for i, r in enumerate(ranked[:20], 1):
        vs = volume_score(r)
        logger.info(
            "%2d. [VS=%.1f] %s", i, vs, r.summary_line(),
        )

    # Also show the best by different criteria
    logger.info("")
    logger.info("── Best by Win Rate (min 200 trades) ──")
    wr_ranked = sorted(
        [r for r in results if r.total_trades >= 200],
        key=lambda r: r.win_rate, reverse=True,
    )
    for r in wr_ranked[:5]:
        logger.info("  WR=%.1f%% | %s", r.win_rate, r.summary_line())

    logger.info("")
    logger.info("── Best by Profit Factor (min 200 trades) ──")
    pf_ranked = sorted(
        [r for r in results if r.total_trades >= 200],
        key=lambda r: r.profit_factor, reverse=True,
    )
    for r in pf_ranked[:5]:
        logger.info("  PF=%.2f | %s", r.profit_factor, r.summary_line())

    logger.info("")
    logger.info("── Best by Total P&L (min 200 trades) ──")
    pnl_ranked = sorted(
        [r for r in results if r.total_trades >= 200],
        key=lambda r: r.total_pnl, reverse=True,
    )
    for r in pnl_ranked[:5]:
        logger.info("  P&L=$%.0f | %s", r.total_pnl, r.summary_line())

    logger.info("")
    logger.info("── Best by Sharpe (min 200 trades) ──")
    sh_ranked = sorted(
        [r for r in results if r.total_trades >= 200],
        key=lambda r: r.sharpe_ratio, reverse=True,
    )
    for r in sh_ranked[:5]:
        logger.info("  Sharpe=%.2f | %s", r.sharpe_ratio, r.summary_line())

    # Save results
    import json
    from datetime import datetime
    out_dir = Path("data/trident_backtest_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"trident_vol_opt_{ts}.json"
    data = {
        "total_configs": len(results),
        "top_20": [
            {
                "rank": i + 1,
                "volume_score": volume_score(r),
                "config_name": r.config_name,
                "trades": r.total_trades,
                "win_rate": r.win_rate,
                "total_pnl": r.total_pnl,
                "profit_factor": r.profit_factor,
                "sharpe": r.sharpe_ratio,
                "max_dd": r.max_drawdown_pct,
                "avg_hold_min": r.avg_hold_minutes,
                "calls": r.calls_taken,
                "puts": r.puts_taken,
                "call_wr": r.call_win_rate,
                "put_wr": r.put_win_rate,
                "per_ticker": r.per_ticker,
                "config": r.config,
            }
            for i, r in enumerate(ranked[:20])
        ],
    }
    out_path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
