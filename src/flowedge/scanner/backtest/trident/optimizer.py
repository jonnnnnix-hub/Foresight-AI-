"""Trident optimizer — grid search across signal/exit/options parameter space.

Strategy:
  Phase 1: Test individual signals to find top performers
  Phase 2: Test combinations of top signals
  Phase 3: Optimize exit parameters on best signal combos
  Phase 4: Optimize options/position parameters

Outputs ranked configurations with full backtest statistics.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .backtester import TridentResult, run_trident_backtest
from .config import (
    Direction,
    EntrySignals,
    ExitParams,
    OptionsParams,
    PositionParams,
    TimeFilter,
    TridentConfig,
)

logger = logging.getLogger("trident.optimizer")

RESULTS_DIR = Path("data/trident_backtest_results")


@dataclass
class OptimizationRun:
    """A complete optimization run with all tested configs."""

    run_id: str
    started_at: str
    completed_at: str = ""
    total_configs: int = 0
    results: list[TridentResult] = field(default_factory=list)
    best_config: dict[str, Any] = field(default_factory=dict)
    phase: str = ""

    def save(self) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"trident_opt_{self.run_id}.json"
        data = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_configs": self.total_configs,
            "phase": self.phase,
            "best_config": self.best_config,
            "results": [
                {
                    "config_name": r.config_name,
                    "total_trades": r.total_trades,
                    "win_rate": r.win_rate,
                    "total_pnl": r.total_pnl,
                    "profit_factor": r.profit_factor,
                    "sharpe_ratio": r.sharpe_ratio,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "avg_hold_minutes": r.avg_hold_minutes,
                    "calls_taken": r.calls_taken,
                    "puts_taken": r.puts_taken,
                    "call_win_rate": r.call_win_rate,
                    "put_win_rate": r.put_win_rate,
                    "signals_per_day": r.signals_per_day,
                    "per_ticker": r.per_ticker,
                }
                for r in self.results
            ],
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved optimization results to %s", path)
        return path


# ── Scoring function ──────────────────────────────────────────────

def score_result(r: TridentResult) -> float:
    """Score a backtest result for ranking.

    Optimises for:
      - High win rate (weight 0.25)
      - High profit factor (weight 0.25)
      - High Sharpe ratio (weight 0.20)
      - Low max drawdown (weight 0.15)
      - Sufficient trade count (weight 0.10)
      - Balanced call/put usage (weight 0.05)

    Returns a composite score (higher = better).
    Minimum thresholds: >= 20 trades, >= 40% WR, PF > 0.8
    """
    # Hard filters — disqualify bad configs
    if r.total_trades < 20:
        return -1000.0
    if r.win_rate < 40.0:
        return -500.0
    if r.profit_factor < 0.8:
        return -200.0

    # Normalised components
    wr_score = min(r.win_rate / 100.0, 1.0) * 100
    pf_score = min(r.profit_factor / 3.0, 1.0) * 100
    sharpe_score = min(max(r.sharpe_ratio, 0) / 3.0, 1.0) * 100
    dd_score = max(0, 100 - r.max_drawdown_pct * 2)
    # Trade count: sweet spot 50-200/year, penalise extremes
    days = max(r.trading_days, 1)
    annualised_trades = r.total_trades / days * 252
    if annualised_trades < 20:
        trade_score = annualised_trades / 20 * 50
    elif annualised_trades > 500:
        trade_score = max(0, 100 - (annualised_trades - 500) / 10)
    else:
        trade_score = 80 + min(20, annualised_trades / 25)

    # Balance: penalise heavily one-sided
    total = r.calls_taken + r.puts_taken
    if total > 0:
        minority = min(r.calls_taken, r.puts_taken) / total
        balance_score = minority * 200  # 50/50 = 100, 80/20 = 40
    else:
        balance_score = 0

    composite = (
        wr_score * 0.25
        + pf_score * 0.25
        + sharpe_score * 0.20
        + dd_score * 0.15
        + trade_score * 0.10
        + balance_score * 0.05
    )

    return composite


# ── Phase 1: Individual signal testing ────────────────────────────

def _build_single_signal_configs() -> list[TridentConfig]:
    """Build configs that test each signal individually."""
    configs: list[TridentConfig] = []

    # Base: all signals off, then enable one at a time
    base: dict[str, Any] = {
        "use_rsi3": False, "use_rsi14": False,
        "use_vwap_position": False, "use_vwap_distance": False,
        "use_ibs": False, "use_volume_spike": False,
        "use_intraday_move": False, "use_prior_bar_color": False,
        "use_ema_cross": False, "use_macd": False,
        "use_bollinger": False, "use_opening_range": False,
        "use_daily_trend": False, "use_sma_micro": False,
    }

    signal_names = [
        "use_rsi3", "use_rsi14", "use_vwap_position",
        "use_ibs", "use_volume_spike", "use_intraday_move",
        "use_prior_bar_color", "use_ema_cross", "use_macd",
        "use_bollinger", "use_opening_range", "use_daily_trend",
        "use_sma_micro",
    ]

    for sig in signal_names:
        params = dict(base)
        params[sig] = True
        params["min_signals_call"] = 1
        params["min_signals_put"] = 1
        entry = EntrySignals(**params)
        cfg = TridentConfig(
            name=f"single_{sig}",
            direction=Direction.BOTH,
            entry=entry,
        )
        configs.append(cfg)

    return configs


# ── Phase 2: Signal combinations ──────────────────────────────────

def _build_combo_configs(
    top_signals: list[str],
    combo_size: int = 3,
) -> list[TridentConfig]:
    """Build configs testing N-of-M signal combinations."""
    configs: list[TridentConfig] = []

    base: dict[str, Any] = {
        "use_rsi3": False, "use_rsi14": False,
        "use_vwap_position": False, "use_vwap_distance": False,
        "use_ibs": False, "use_volume_spike": False,
        "use_intraday_move": False, "use_prior_bar_color": False,
        "use_ema_cross": False, "use_macd": False,
        "use_bollinger": False, "use_opening_range": False,
        "use_daily_trend": False, "use_sma_micro": False,
    }

    for combo in itertools.combinations(top_signals, combo_size):
        params = dict(base)
        for sig in combo:
            params[sig] = True

        # Test different confluence requirements
        for min_sigs in range(2, min(combo_size + 1, 4)):
            params["min_signals_call"] = min_sigs
            params["min_signals_put"] = min_sigs
            entry = EntrySignals(**params)

            name = "+".join(s.replace("use_", "") for s in combo)
            cfg = TridentConfig(
                name=f"combo_{name}_min{min_sigs}",
                direction=Direction.BOTH,
                entry=entry,
            )
            configs.append(cfg)

    return configs


# ── Phase 3: Exit parameter sweep ─────────────────────────────────

def _build_exit_configs(
    best_entry: EntrySignals,
) -> list[TridentConfig]:
    """Sweep exit parameters on the best entry signal config."""
    configs: list[TridentConfig] = []

    tp_values = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75]
    sl_values = [-0.20, -0.25, -0.30, -0.35, -0.40, -0.50]
    trail_values = [0.20, 0.25, 0.30, 0.40, 0.50]
    max_hold_values = [6, 9, 12, 15, 18]  # 30, 45, 60, 75, 90 min

    # Sample key combinations (full grid would be too large)
    for tp in tp_values:
        for sl in sl_values:
            for trail in trail_values:
                for hold in max_hold_values:
                    # Skip illogical combos
                    if abs(sl) < tp * 0.5:
                        continue  # SL too tight relative to TP
                    exit_params = ExitParams(
                        tp_pct=tp,
                        sl_pct=sl,
                        trail_pct=trail,
                        max_hold_bars=hold,
                        use_trailing=True,
                    )
                    cfg = TridentConfig(
                        name=(
                            f"exit_tp{int(tp*100)}"
                            f"_sl{int(abs(sl)*100)}"
                            f"_tr{int(trail*100)}"
                            f"_h{hold*5}m"
                        ),
                        direction=Direction.BOTH,
                        entry=best_entry,
                        exit=exit_params,
                    )
                    configs.append(cfg)

    return configs


# ── Phase 4: Options parameter sweep ──────────────────────────────

def _build_options_configs(
    best_entry: EntrySignals,
    best_exit: ExitParams,
) -> list[TridentConfig]:
    """Sweep options selection parameters."""
    configs: list[TridentConfig] = []

    dte_ranges = [(0, 0), (0, 1), (0, 2), (0, 5)]
    delta_ranges = [
        (0.15, 0.30),   # OTM — lottery
        (0.25, 0.40),   # slight OTM
        (0.30, 0.50),   # near ATM
        (0.40, 0.55),   # ATM
    ]
    risk_values = [0.02, 0.03, 0.04, 0.05]
    max_pos_values = [1, 2, 3]

    for dte_min, dte_max in dte_ranges:
        for d_min, d_max in delta_ranges:
            for risk in risk_values:
                for max_pos in max_pos_values:
                    opts = OptionsParams(
                        min_dte=dte_min,
                        max_dte=dte_max,
                        min_delta=d_min,
                        max_delta=d_max,
                    )
                    pos = PositionParams(
                        risk_per_trade=risk,
                        max_positions=max_pos,
                    )
                    cfg = TridentConfig(
                        name=(
                            f"opts_dte{dte_min}-{dte_max}"
                            f"_d{int(d_min*100)}-{int(d_max*100)}"
                            f"_r{int(risk*100)}"
                            f"_p{max_pos}"
                        ),
                        direction=Direction.BOTH,
                        entry=best_entry,
                        exit=best_exit,
                        options=opts,
                        position=pos,
                    )
                    configs.append(cfg)

    return configs


# ── Phase 5: Time window sweep ────────────────────────────────────

def _build_time_configs(
    best_entry: EntrySignals,
    best_exit: ExitParams,
    best_options: OptionsParams,
    best_position: PositionParams,
) -> list[TridentConfig]:
    """Sweep time-of-day windows."""
    configs: list[TridentConfig] = []

    windows = [
        ("all_day", True, True, True),
        ("morning_only", True, False, False),
        ("morning_afternoon", True, False, True),
        ("midday_afternoon", False, True, True),
        ("afternoon_only", False, False, True),
    ]
    skip_first = [3, 5, 10, 15]

    for wname, morn, mid, aft in windows:
        for skip in skip_first:
            tf = TimeFilter(
                skip_first_n_minutes=skip,
                use_morning=morn,
                use_midday=mid,
                use_afternoon=aft,
            )
            cfg = TridentConfig(
                name=f"time_{wname}_skip{skip}",
                direction=Direction.BOTH,
                entry=best_entry,
                exit=best_exit,
                options=best_options,
                position=best_position,
                time_filter=tf,
            )
            configs.append(cfg)

    return configs


# ── Main optimization runner ──────────────────────────────────────

def run_optimization(
    phases: list[str] | None = None,
    max_configs_per_phase: int = 200,
) -> OptimizationRun:
    """Run the full Trident optimization pipeline.

    Args:
        phases: Which phases to run. Default: all.
            Options: "signals", "combos", "exits", "options", "time"
        max_configs_per_phase: Cap configs per phase to control runtime.

    Returns:
        OptimizationRun with all results and the best config.
    """
    if phases is None:
        phases = ["signals", "combos", "exits", "options", "time"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = OptimizationRun(
        run_id=run_id,
        started_at=datetime.now().isoformat(),
    )

    best_entry = EntrySignals()
    best_exit = ExitParams()
    best_options = OptionsParams()
    best_position = PositionParams()

    # ── Phase 1: Individual signals ───────────────────────────
    if "signals" in phases:
        run.phase = "signals"
        logger.info("=" * 60)
        logger.info("PHASE 1: Individual Signal Testing")
        logger.info("=" * 60)

        configs = _build_single_signal_configs()
        results = _run_configs(configs[:max_configs_per_phase])
        run.results.extend(results)

        # Rank signals by score
        ranked = sorted(results, key=lambda r: score_result(r), reverse=True)
        top_signals: list[str] = []
        logger.info("\nIndividual signal rankings:")
        for i, r in enumerate(ranked):
            sc = score_result(r)
            logger.info(
                "  %2d. [%.1f] %s", i + 1, sc, r.summary_line(),
            )
            if sc > 0 and len(top_signals) < 8:
                # Extract signal name from config_name
                sig_name = r.config_name.replace("single_", "")
                top_signals.append(sig_name)

        if not top_signals:
            # Fallback: use defaults
            top_signals = [
                "use_rsi3", "use_vwap_position", "use_ibs",
                "use_volume_spike", "use_intraday_move",
                "use_prior_bar_color", "use_daily_trend",
            ]
            logger.info(
                "No signals passed threshold, using defaults: %s",
                top_signals,
            )

        logger.info("Top signals for combo testing: %s", top_signals)

    else:
        top_signals = [
            "use_rsi3", "use_vwap_position", "use_ibs",
            "use_volume_spike", "use_intraday_move",
            "use_prior_bar_color", "use_daily_trend",
        ]

    # ── Phase 2: Signal combinations ──────────────────────────
    if "combos" in phases:
        run.phase = "combos"
        logger.info("=" * 60)
        logger.info("PHASE 2: Signal Combination Testing")
        logger.info("=" * 60)

        configs = _build_combo_configs(top_signals, combo_size=3)
        # Also test 4-signal combos with top 6
        configs += _build_combo_configs(top_signals[:6], combo_size=4)

        # Cap if too many
        if len(configs) > max_configs_per_phase:
            logger.info(
                "Capping from %d to %d configs", len(configs),
                max_configs_per_phase,
            )
            configs = configs[:max_configs_per_phase]

        results = _run_configs(configs)
        run.results.extend(results)

        ranked = sorted(results, key=lambda r: score_result(r), reverse=True)
        logger.info("\nTop 10 signal combinations:")
        for i, r in enumerate(ranked[:10]):
            sc = score_result(r)
            logger.info("  %2d. [%.1f] %s", i + 1, sc, r.summary_line())

        if ranked and score_result(ranked[0]) > 0:
            best_entry = EntrySignals(**ranked[0].config["entry"])
        logger.info("Best entry config: %s", ranked[0].config_name if ranked else "default")

    # ── Phase 3: Exit parameter sweep ─────────────────────────
    if "exits" in phases:
        run.phase = "exits"
        logger.info("=" * 60)
        logger.info("PHASE 3: Exit Parameter Optimization")
        logger.info("=" * 60)

        configs = _build_exit_configs(best_entry)
        if len(configs) > max_configs_per_phase:
            # Sample uniformly
            step = len(configs) // max_configs_per_phase
            configs = configs[::max(1, step)][:max_configs_per_phase]

        results = _run_configs(configs)
        run.results.extend(results)

        ranked = sorted(results, key=lambda r: score_result(r), reverse=True)
        logger.info("\nTop 10 exit configurations:")
        for i, r in enumerate(ranked[:10]):
            sc = score_result(r)
            logger.info("  %2d. [%.1f] %s", i + 1, sc, r.summary_line())

        if ranked and score_result(ranked[0]) > 0:
            best_exit = ExitParams(**ranked[0].config["exit"])

    # ── Phase 4: Options parameters ───────────────────────────
    if "options" in phases:
        run.phase = "options"
        logger.info("=" * 60)
        logger.info("PHASE 4: Options Parameter Optimization")
        logger.info("=" * 60)

        configs = _build_options_configs(best_entry, best_exit)
        if len(configs) > max_configs_per_phase:
            step = len(configs) // max_configs_per_phase
            configs = configs[::max(1, step)][:max_configs_per_phase]

        results = _run_configs(configs)
        run.results.extend(results)

        ranked = sorted(results, key=lambda r: score_result(r), reverse=True)
        logger.info("\nTop 10 options configurations:")
        for i, r in enumerate(ranked[:10]):
            sc = score_result(r)
            logger.info("  %2d. [%.1f] %s", i + 1, sc, r.summary_line())

        if ranked and score_result(ranked[0]) > 0:
            best_options = OptionsParams(**ranked[0].config["options"])
            best_position = PositionParams(**ranked[0].config["position"])

    # ── Phase 5: Time windows ─────────────────────────────────
    if "time" in phases:
        run.phase = "time"
        logger.info("=" * 60)
        logger.info("PHASE 5: Time Window Optimization")
        logger.info("=" * 60)

        configs = _build_time_configs(
            best_entry, best_exit, best_options, best_position,
        )
        results = _run_configs(configs[:max_configs_per_phase])
        run.results.extend(results)

        ranked = sorted(results, key=lambda r: score_result(r), reverse=True)
        logger.info("\nTop 5 time configurations:")
        for i, r in enumerate(ranked[:5]):
            sc = score_result(r)
            logger.info("  %2d. [%.1f] %s", i + 1, sc, r.summary_line())

    # ── Final: build champion config ──────────────────────────
    all_ranked = sorted(
        run.results, key=lambda r: score_result(r), reverse=True,
    )

    if all_ranked:
        champion = all_ranked[0]
        run.best_config = champion.config
        logger.info("=" * 60)
        logger.info("CHAMPION CONFIG: %s", champion.config_name)
        logger.info("  %s", champion.summary_line())
        logger.info("  Score: %.1f", score_result(champion))
        logger.info("=" * 60)
    else:
        logger.warning("No valid results found")

    run.completed_at = datetime.now().isoformat()
    run.total_configs = len(run.results)
    run.save()

    return run


def _run_configs(configs: list[TridentConfig]) -> list[TridentResult]:
    """Run a batch of configs and return results."""
    results: list[TridentResult] = []
    total = len(configs)

    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        try:
            result = run_trident_backtest(cfg)
            results.append(result)
            elapsed = time.time() - t0
            logger.info(
                "[%d/%d] %s — %.1fs — %s",
                i, total, cfg.name, elapsed, result.summary_line(),
            )
        except Exception as exc:
            logger.error("[%d/%d] %s FAILED: %s", i, total, cfg.name, exc)

    return results


# ── Quick single-config test ──────────────────────────────────────

def run_single_test(config: TridentConfig | None = None) -> TridentResult:
    """Run a single backtest with default or custom config.

    Useful for quick validation before launching full optimization.
    """
    cfg = config or TridentConfig(name="trident_quick_test")
    logger.info("Running single test: %s", cfg.name)
    result = run_trident_backtest(cfg)
    logger.info("Result: %s", result.summary_line())

    # Save individual result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"trident_single_{ts}.json"
    data = {
        "config_name": result.config_name,
        "config": result.config,
        "summary": result.summary_line(),
        "total_trades": result.total_trades,
        "win_rate": result.win_rate,
        "total_pnl": result.total_pnl,
        "profit_factor": result.profit_factor,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "avg_hold_minutes": result.avg_hold_minutes,
        "signals_per_day": result.signals_per_day,
        "calls_taken": result.calls_taken,
        "puts_taken": result.puts_taken,
        "call_win_rate": result.call_win_rate,
        "put_win_rate": result.put_win_rate,
        "per_ticker": result.per_ticker,
        "trades": [
            {
                "ticker": t.ticker,
                "direction": t.direction,
                "entry_date": t.entry_date,
                "entry_time": t.entry_time,
                "entry_underlying": t.entry_underlying,
                "entry_option_price": t.entry_option_price,
                "exit_underlying": t.exit_underlying,
                "exit_option_price": t.exit_option_price,
                "exit_reason": t.exit_reason,
                "hold_minutes": t.hold_minutes,
                "pnl_dollars": t.pnl_dollars,
                "pnl_pct": t.pnl_pct,
                "conviction": t.conviction,
            }
            for t in result.trades[:500]  # cap for file size
        ],
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Saved to %s", path)

    return result
