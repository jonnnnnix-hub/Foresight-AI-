"""Per-ticker parameter grid search optimizer.

Coarse-to-fine grid search for shares_engine and scalp_real parameters.
Runs staged sweeps to avoid combinatorial explosion:
  Stage 1: IBS × RSI (entry signal tuning)
  Stage 2: TP% × max_hold (exit tuning, using best entry params)
  Stage 3: trail_trigger × trail_exit (trail tuning, using best entry+exit)

Each run tests a single ticker with a single parameter set.
Results ranked by composite score:
  0.4 × win_rate + 0.3 × norm(profit_factor) + 0.2 × norm(return) + 0.1 × (1 - norm(drawdown))
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from flowedge.scanner.backtest.schemas import BacktestResult
from flowedge.scanner.backtest.shares_engine import run_shares_backtest

logger = structlog.get_logger()

CACHE_DIR = Path("data/flat_files_s3")
OUTPUT_DIR = Path("data/optimizer")


# ── Schemas ──────────────────────────────────────────────────────────────────


@dataclass
class ParamRange:
    """A single parameter's search space."""

    name: str
    values: list[float]


@dataclass
class TrialResult:
    """Result of a single parameter combination trial."""

    params: dict[str, float]
    total_trades: int
    wins: int
    win_rate: float
    profit_factor: float
    total_pnl_pct: float
    portfolio_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_hold_days: float
    composite_score: float


@dataclass
class TickerOptResult:
    """Optimal parameters and metrics for a single ticker."""

    ticker: str
    best_params: dict[str, float]
    best_score: float
    baseline_win_rate: float
    optimized_win_rate: float
    baseline_return_pct: float
    optimized_return_pct: float
    baseline_profit_factor: float
    optimized_profit_factor: float
    total_trades: int
    trials_run: int
    years_of_data: float = 0.0
    avg_annual_return_pct: float = 0.0
    avg_annual_pnl_dollars: float = 0.0  # Based on $10K/year starting capital
    all_trials: list[TrialResult] = field(default_factory=list)


@dataclass
class GridSearchResult:
    """Aggregate results across all tickers."""

    engine: str
    tickers: list[TickerOptResult]
    total_trials: int
    elapsed_seconds: float


# ── Scoring ──────────────────────────────────────────────────────────────────


def _composite_score(result: BacktestResult) -> float:
    """Score a backtest result for optimization ranking.

    Composite = 0.4 × WR + 0.3 × norm(PF) + 0.2 × norm(return) + 0.1 × (1 - norm(DD))
    """
    wr = result.win_rate

    # Normalize profit factor: cap at 5.0, scale to 0-1
    pf = min(result.profit_factor, 5.0) / 5.0

    # Normalize return: sigmoid-ish mapping, cap at ±200%
    ret = result.portfolio_return_pct
    ret_norm = max(0.0, min(1.0, (ret + 50) / 150))  # -50%→0, +100%→1

    # Normalize drawdown: lower is better, cap at 30%
    dd = min(result.max_drawdown_pct, 30.0)
    dd_norm = dd / 30.0

    return 0.4 * wr + 0.3 * pf + 0.2 * ret_norm + 0.1 * (1.0 - dd_norm)


def _run_single_trial(
    ticker: str,
    params: dict[str, Any],
    mode: str = "precision_shares",
    starting_capital: float = 10_000.0,
) -> TrialResult | None:
    """Run a single shares backtest trial for one ticker with given params."""
    result = run_shares_backtest(
        mode=mode,
        starting_capital=starting_capital,
        tickers=[ticker],
        params=params,
    )
    if result.total_trades < 10:
        return None

    score = _composite_score(result)
    return TrialResult(
        params=dict(params),
        total_trades=result.total_trades,
        wins=result.wins,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        total_pnl_pct=result.total_pnl_pct,
        portfolio_return_pct=result.portfolio_return_pct,
        max_drawdown_pct=result.max_drawdown_pct,
        sharpe_ratio=result.sharpe_ratio,
        avg_hold_days=result.avg_hold_days,
        composite_score=score,
    )


# ── Shares Grid Search ──────────────────────────────────────────────────────


# Stage 1: Entry signal sweep
SHARES_IBS_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SHARES_RSI_VALUES = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]

# Stage 2: Exit sweep
SHARES_TP_VALUES = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
SHARES_HOLD_VALUES = [3, 5, 7, 10, 12]

# Stage 3: Trail sweep
SHARES_TRAIL_TRIGGER_VALUES = [0.005, 0.010, 0.015, 0.020, 0.030]
SHARES_TRAIL_EXIT_VALUES = [0.002, 0.003, 0.005, 0.007, 0.010]

# Stage 4: Position sizing sweep
SHARES_RISK_PCT_VALUES = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
SHARES_MAX_POSITIONS_VALUES = [1, 2, 3, 4, 5]
SHARES_MIN_HOLD_VALUES = [0, 1, 2, 3, 5]
SHARES_COOLDOWN_VALUES = [1, 2, 3, 5]


def optimize_shares_ticker(
    ticker: str,
    mode: str = "precision_shares",
    starting_capital: float = 10_000.0,
    min_trades: int = 10,
) -> TickerOptResult | None:
    """Run staged grid search for a single ticker on the shares engine.

    Stage 1: IBS × RSI sweep (best entry)
    Stage 2: TP × max_hold sweep (best exit, using Stage 1 entry)
    Stage 3: trail_trigger × trail_exit sweep (best trail, using Stage 1+2)
    """
    logger.info("optimizer_start", ticker=ticker, engine="shares")
    trials: list[TrialResult] = []

    # Baseline: run with default params
    baseline = _run_single_trial(ticker, {}, mode=mode, starting_capital=starting_capital)
    if not baseline:
        logger.info("optimizer_skip", ticker=ticker, reason="insufficient_trades_baseline")
        return None

    # ── Stage 1: IBS × RSI ──
    best_ibs = 0.20
    best_rsi = 45.0
    best_score = -1.0

    for ibs in SHARES_IBS_VALUES:
        for rsi in SHARES_RSI_VALUES:
            params = {"ibs_threshold": ibs, "rsi_threshold": rsi}
            trial = _run_single_trial(
                ticker, params, mode=mode, starting_capital=starting_capital,
            )
            if trial and trial.total_trades >= min_trades:
                trials.append(trial)
                if trial.composite_score > best_score:
                    best_score = trial.composite_score
                    best_ibs = ibs
                    best_rsi = rsi

    logger.info(
        "optimizer_stage1",
        ticker=ticker,
        best_ibs=best_ibs,
        best_rsi=best_rsi,
        best_score=round(best_score, 4),
        trials=len(trials),
    )

    # ── Stage 2: TP × max_hold ──
    best_tp = 0.025
    best_hold = 12

    for tp in SHARES_TP_VALUES:
        for hold in SHARES_HOLD_VALUES:
            params = {
                "ibs_threshold": best_ibs,
                "rsi_threshold": best_rsi,
                "tp_pct": tp,
                "max_hold": hold,
            }
            trial = _run_single_trial(
                ticker, params, mode=mode, starting_capital=starting_capital,
            )
            if trial and trial.total_trades >= min_trades:
                trials.append(trial)
                if trial.composite_score > best_score:
                    best_score = trial.composite_score
                    best_tp = tp
                    best_hold = hold

    logger.info(
        "optimizer_stage2",
        ticker=ticker,
        best_tp=best_tp,
        best_hold=best_hold,
        best_score=round(best_score, 4),
    )

    # ── Stage 3: trail_trigger × trail_exit ──
    best_trail_trigger = 0.015
    best_trail_exit = 0.005

    for trigger in SHARES_TRAIL_TRIGGER_VALUES:
        for exit_val in SHARES_TRAIL_EXIT_VALUES:
            if exit_val >= trigger:
                continue  # Exit must be below trigger
            params = {
                "ibs_threshold": best_ibs,
                "rsi_threshold": best_rsi,
                "tp_pct": best_tp,
                "max_hold": best_hold,
                "trail_trigger": trigger,
                "trail_exit": exit_val,
            }
            trial = _run_single_trial(
                ticker, params, mode=mode, starting_capital=starting_capital,
            )
            if trial and trial.total_trades >= min_trades:
                trials.append(trial)
                if trial.composite_score > best_score:
                    best_score = trial.composite_score
                    best_trail_trigger = trigger
                    best_trail_exit = exit_val

    logger.info(
        "optimizer_stage3",
        ticker=ticker,
        best_trail_trigger=best_trail_trigger,
        best_trail_exit=best_trail_exit,
        best_score=round(best_score, 4),
    )

    # ── Stage 4: Position sizing (risk_pct, min_hold, cooldown) ──
    best_risk_pct = 0.10
    best_min_hold = 3
    best_cooldown = 3

    for risk in SHARES_RISK_PCT_VALUES:
        for min_h in SHARES_MIN_HOLD_VALUES:
            if min_h >= best_hold:
                continue
            params = {
                "ibs_threshold": best_ibs,
                "rsi_threshold": best_rsi,
                "tp_pct": best_tp,
                "max_hold": best_hold,
                "trail_trigger": best_trail_trigger,
                "trail_exit": best_trail_exit,
                "risk_pct": risk,
                "min_hold": min_h,
            }
            trial = _run_single_trial(
                ticker, params, mode=mode, starting_capital=starting_capital,
            )
            if trial and trial.total_trades >= min_trades:
                trials.append(trial)
                if trial.composite_score > best_score:
                    best_score = trial.composite_score
                    best_risk_pct = risk
                    best_min_hold = min_h

    for cd in SHARES_COOLDOWN_VALUES:
        params = {
            "ibs_threshold": best_ibs,
            "rsi_threshold": best_rsi,
            "tp_pct": best_tp,
            "max_hold": best_hold,
            "trail_trigger": best_trail_trigger,
            "trail_exit": best_trail_exit,
            "risk_pct": best_risk_pct,
            "min_hold": best_min_hold,
            "cooldown": cd,
        }
        trial = _run_single_trial(
            ticker, params, mode=mode, starting_capital=starting_capital,
        )
        if trial and trial.total_trades >= min_trades:
            trials.append(trial)
            if trial.composite_score > best_score:
                best_score = trial.composite_score
                best_cooldown = cd

    logger.info(
        "optimizer_stage4",
        ticker=ticker,
        best_risk_pct=best_risk_pct,
        best_min_hold=best_min_hold,
        best_cooldown=best_cooldown,
        best_score=round(best_score, 4),
    )

    # ── Final optimized run ──
    final_params: dict[str, Any] = {
        "ibs_threshold": best_ibs,
        "rsi_threshold": best_rsi,
        "tp_pct": best_tp,
        "max_hold": best_hold,
        "trail_trigger": best_trail_trigger,
        "trail_exit": best_trail_exit,
        "risk_pct": best_risk_pct,
        "min_hold": best_min_hold,
        "cooldown": best_cooldown,
    }
    # Run final with full result to compute annual metrics
    final_result = run_shares_backtest(
        mode=mode, starting_capital=starting_capital,
        tickers=[ticker], params=final_params, _record=False,
    )
    final = None
    if final_result.total_trades >= min_trades:
        score = _composite_score(final_result)
        final = TrialResult(
            params=dict(final_params),
            total_trades=final_result.total_trades,
            wins=final_result.wins,
            win_rate=final_result.win_rate,
            profit_factor=final_result.profit_factor,
            total_pnl_pct=final_result.total_pnl_pct,
            portfolio_return_pct=final_result.portfolio_return_pct,
            max_drawdown_pct=final_result.max_drawdown_pct,
            sharpe_ratio=final_result.sharpe_ratio,
            avg_hold_days=final_result.avg_hold_days,
            composite_score=score,
        )
        trials.append(final)

    opt_wr = final.win_rate if final else baseline.win_rate
    opt_ret = final.portfolio_return_pct if final else baseline.portfolio_return_pct
    opt_pf = final.profit_factor if final else baseline.profit_factor
    opt_trades = final.total_trades if final else baseline.total_trades

    # Compute annual P&L: avg per-year return assuming $10K start each year
    years_of_data = 0.0
    avg_annual_ret = 0.0
    avg_annual_pnl = 0.0
    if final_result.trades:
        first_date = min(t.entry_date for t in final_result.trades)
        last_date = max(
            t.exit_date or t.entry_date for t in final_result.trades
        )
        days_span = (last_date - first_date).days
        years_of_data = max(0.1, days_span / 365.25)
        total_ret = final_result.portfolio_return_pct
        avg_annual_ret = round(total_ret / years_of_data, 2)
        avg_annual_pnl = round(starting_capital * avg_annual_ret / 100, 2)

    return TickerOptResult(
        ticker=ticker,
        best_params={k: float(v) for k, v in final_params.items()},
        best_score=best_score,
        baseline_win_rate=baseline.win_rate,
        optimized_win_rate=opt_wr,
        baseline_return_pct=baseline.portfolio_return_pct,
        optimized_return_pct=opt_ret,
        baseline_profit_factor=baseline.profit_factor,
        optimized_profit_factor=opt_pf,
        total_trades=opt_trades,
        trials_run=len(trials),
        years_of_data=round(years_of_data, 1),
        avg_annual_return_pct=avg_annual_ret,
        avg_annual_pnl_dollars=avg_annual_pnl,
        all_trials=trials,
    )


# ── Scalp Grid Search ───────────────────────────────────────────────────────


SCALP_IBS_VALUES = [0.05, 0.08, 0.10, 0.15, 0.20]
SCALP_RSI_VALUES = [10.0, 15.0, 20.0, 25.0, 30.0]
SCALP_TP_VALUES = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
SCALP_HOLD_VALUES = [3, 4, 6, 8, 12]
SCALP_TRAIL_VALUES = [0.02, 0.03, 0.04, 0.06, 0.08]


def _run_single_scalp_trial(
    ticker: str,
    params: dict[str, Any],
    starting_capital: float = 25_000.0,
) -> TrialResult | None:
    """Run a single scalp_real backtest trial for one ticker."""
    from flowedge.scanner.backtest.scalp_real import run_scalp_real_backtest

    result = run_scalp_real_backtest(
        tickers=[ticker],
        starting_capital=starting_capital,
        params=params,
    )
    if result.total_trades < 5:
        return None

    score = _composite_score(result)
    return TrialResult(
        params=dict(params),
        total_trades=result.total_trades,
        wins=result.wins,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        total_pnl_pct=result.total_pnl_pct,
        portfolio_return_pct=result.portfolio_return_pct,
        max_drawdown_pct=result.max_drawdown_pct,
        sharpe_ratio=result.sharpe_ratio,
        avg_hold_days=result.avg_hold_days,
        composite_score=score,
    )


def optimize_scalp_ticker(
    ticker: str,
    starting_capital: float = 25_000.0,
    min_trades: int = 5,
) -> TickerOptResult | None:
    """Run staged grid search for a single ticker on the scalp options engine.

    Stage 1: IBS × RSI sweep
    Stage 2: TP × max_hold_bars sweep
    Stage 3: trail_pct sweep
    """
    logger.info("optimizer_start", ticker=ticker, engine="scalp_real")
    trials: list[TrialResult] = []

    # Baseline
    baseline = _run_single_scalp_trial(ticker, {}, starting_capital=starting_capital)
    if not baseline:
        logger.info("optimizer_skip", ticker=ticker, reason="insufficient_trades_baseline")
        return None

    # ── Stage 1: IBS × RSI ──
    best_ibs = 0.10
    best_rsi = 20.0
    best_score = -1.0

    for ibs in SCALP_IBS_VALUES:
        for rsi in SCALP_RSI_VALUES:
            params: dict[str, Any] = {"ibs_thresh": ibs, "rsi3_thresh": rsi}
            trial = _run_single_scalp_trial(ticker, params, starting_capital)
            if trial and trial.total_trades >= min_trades:
                trials.append(trial)
                if trial.composite_score > best_score:
                    best_score = trial.composite_score
                    best_ibs = ibs
                    best_rsi = rsi

    logger.info(
        "optimizer_scalp_stage1",
        ticker=ticker,
        best_ibs=best_ibs,
        best_rsi=best_rsi,
        trials=len(trials),
    )

    # ── Stage 2: TP × max_hold_bars ──
    best_tp = 0.10
    best_hold = 6

    for tp in SCALP_TP_VALUES:
        for hold in SCALP_HOLD_VALUES:
            params = {
                "ibs_thresh": best_ibs,
                "rsi3_thresh": best_rsi,
                "tp_pct": tp,
                "max_hold_bars": hold,
            }
            trial = _run_single_scalp_trial(ticker, params, starting_capital)
            if trial and trial.total_trades >= min_trades:
                trials.append(trial)
                if trial.composite_score > best_score:
                    best_score = trial.composite_score
                    best_tp = tp
                    best_hold = hold

    logger.info(
        "optimizer_scalp_stage2",
        ticker=ticker,
        best_tp=best_tp,
        best_hold=best_hold,
    )

    # ── Stage 3: trail_pct ──
    best_trail = 0.04

    for trail in SCALP_TRAIL_VALUES:
        params = {
            "ibs_thresh": best_ibs,
            "rsi3_thresh": best_rsi,
            "tp_pct": best_tp,
            "max_hold_bars": best_hold,
            "trail_pct": trail,
        }
        trial = _run_single_scalp_trial(ticker, params, starting_capital)
        if trial and trial.total_trades >= min_trades:
            trials.append(trial)
            if trial.composite_score > best_score:
                best_score = trial.composite_score
                best_trail = trail

    # ── Final ──
    final_params: dict[str, Any] = {
        "ibs_thresh": best_ibs,
        "rsi3_thresh": best_rsi,
        "tp_pct": best_tp,
        "max_hold_bars": best_hold,
        "trail_pct": best_trail,
    }
    final = _run_single_scalp_trial(ticker, final_params, starting_capital)
    if final:
        trials.append(final)

    opt_wr = final.win_rate if final else baseline.win_rate
    opt_ret = final.portfolio_return_pct if final else baseline.portfolio_return_pct
    opt_pf = final.profit_factor if final else baseline.profit_factor
    opt_trades = final.total_trades if final else baseline.total_trades

    return TickerOptResult(
        ticker=ticker,
        best_params={k: float(v) for k, v in final_params.items()},
        best_score=best_score,
        baseline_win_rate=baseline.win_rate,
        optimized_win_rate=opt_wr,
        baseline_return_pct=baseline.portfolio_return_pct,
        optimized_return_pct=opt_ret,
        baseline_profit_factor=baseline.profit_factor,
        optimized_profit_factor=opt_pf,
        total_trades=opt_trades,
        trials_run=len(trials),
        all_trials=trials,
    )


# ── Full Grid Search Runners ────────────────────────────────────────────────


def run_shares_grid_search(
    tickers: list[str] | None = None,
    mode: str = "precision_shares",
    starting_capital: float = 10_000.0,
) -> GridSearchResult:
    """Run shares grid search across all tickers."""
    if tickers is None:
        # Discover all tickers with cached data
        tickers = sorted(
            d.name for d in CACHE_DIR.iterdir()
            if d.is_dir() and (d / "1min").exists()
        )

    logger.info("grid_search_start", engine="shares", tickers=len(tickers))
    t0 = time.time()
    results: list[TickerOptResult] = []
    total_trials = 0

    for ticker in tickers:
        opt = optimize_shares_ticker(
            ticker, mode=mode, starting_capital=starting_capital,
        )
        if opt:
            results.append(opt)
            total_trials += opt.trials_run
            logger.info(
                "ticker_optimized",
                ticker=ticker,
                baseline_wr=opt.baseline_win_rate,
                optimized_wr=opt.optimized_win_rate,
                delta_wr=round(opt.optimized_win_rate - opt.baseline_win_rate, 3),
                optimized_ret=opt.optimized_return_pct,
                trials=opt.trials_run,
            )

    elapsed = time.time() - t0

    # Sort by best composite score
    results.sort(key=lambda r: r.best_score, reverse=True)

    grid_result = GridSearchResult(
        engine="shares",
        tickers=results,
        total_trials=total_trials,
        elapsed_seconds=round(elapsed, 1),
    )

    # Save results
    _save_results(grid_result, OUTPUT_DIR / "shares_grid_results.json")
    _print_shares_report(grid_result)

    return grid_result


def run_scalp_grid_search(
    tickers: list[str] | None = None,
    starting_capital: float = 25_000.0,
) -> GridSearchResult:
    """Run scalp options grid search across tickers with options data."""
    if tickers is None:
        tickers = sorted(
            d.name for d in CACHE_DIR.iterdir()
            if d.is_dir() and (d / "options_1min").exists()
        )

    logger.info("grid_search_start", engine="scalp_real", tickers=len(tickers))
    t0 = time.time()
    results: list[TickerOptResult] = []
    total_trials = 0

    for ticker in tickers:
        opt = optimize_scalp_ticker(
            ticker, starting_capital=starting_capital,
        )
        if opt:
            results.append(opt)
            total_trials += opt.trials_run
            logger.info(
                "ticker_optimized",
                ticker=ticker,
                baseline_wr=opt.baseline_win_rate,
                optimized_wr=opt.optimized_win_rate,
                optimized_ret=opt.optimized_return_pct,
                trials=opt.trials_run,
            )

    elapsed = time.time() - t0
    results.sort(key=lambda r: r.best_score, reverse=True)

    grid_result = GridSearchResult(
        engine="scalp_real",
        tickers=results,
        total_trials=total_trials,
        elapsed_seconds=round(elapsed, 1),
    )

    _save_results(grid_result, OUTPUT_DIR / "scalp_grid_results.json")
    _print_scalp_report(grid_result)

    return grid_result


# ── Persistence & Reporting ──────────────────────────────────────────────────


def _save_results(result: GridSearchResult, path: Path) -> None:
    """Save grid search results to JSON (without full trial details)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "engine": result.engine,
        "total_trials": result.total_trials,
        "elapsed_seconds": result.elapsed_seconds,
        "tickers": [],
    }
    for t in result.tickers:
        data["tickers"].append({
            "ticker": t.ticker,
            "best_params": t.best_params,
            "best_score": round(t.best_score, 4),
            "baseline_win_rate": t.baseline_win_rate,
            "optimized_win_rate": t.optimized_win_rate,
            "baseline_return_pct": t.baseline_return_pct,
            "optimized_return_pct": t.optimized_return_pct,
            "baseline_profit_factor": t.baseline_profit_factor,
            "optimized_profit_factor": t.optimized_profit_factor,
            "total_trades": t.total_trades,
            "trials_run": t.trials_run,
            "years_of_data": t.years_of_data,
            "avg_annual_return_pct": t.avg_annual_return_pct,
            "avg_annual_pnl_dollars": t.avg_annual_pnl_dollars,
        })

    path.write_text(json.dumps(data, indent=2))
    logger.info("results_saved", path=str(path))


def _print_shares_report(result: GridSearchResult) -> None:
    """Print a ranked summary table of shares grid search results."""
    print("\n" + "=" * 130)
    print(f"SHARES GRID SEARCH — {len(result.tickers)} tickers, "
          f"{result.total_trials} trials, {result.elapsed_seconds:.0f}s")
    print("=" * 130)
    print(
        f"{'Ticker':<8} {'Base WR':>8} {'Opt WR':>8} {'Delta':>7} "
        f"{'Tot Ret':>9} {'Yrs':>5} {'Avg/Yr%':>8} {'Avg/Yr$':>9} "
        f"{'PF':>6} {'Trades':>7} "
        f"{'IBS':>6} {'RSI':>6} {'TP%':>6} {'Hold':>5} {'Risk%':>6} {'Score':>7}"
    )
    print("-" * 130)

    for t in result.tickers:
        delta_wr = t.optimized_win_rate - t.baseline_win_rate
        p = t.best_params
        print(
            f"{t.ticker:<8} "
            f"{t.baseline_win_rate:>7.1%} "
            f"{t.optimized_win_rate:>7.1%} "
            f"{delta_wr:>+6.1%} "
            f"{t.optimized_return_pct:>+8.1f}% "
            f"{t.years_of_data:>4.1f}y "
            f"{t.avg_annual_return_pct:>+7.1f}% "
            f"${t.avg_annual_pnl_dollars:>+8.0f} "
            f"{t.optimized_profit_factor:>5.2f} "
            f"{t.total_trades:>7} "
            f"{p.get('ibs_threshold', 0.2):>5.2f} "
            f"{p.get('rsi_threshold', 45):>5.0f} "
            f"{p.get('tp_pct', 0.025) * 100:>5.1f} "
            f"{p.get('max_hold', 12):>5.0f} "
            f"{p.get('risk_pct', 0.1) * 100:>5.0f}% "
            f"{t.best_score:>6.4f}"
        )

    # Summary
    if result.tickers:
        avg_base_wr = sum(t.baseline_win_rate for t in result.tickers) / len(result.tickers)
        avg_opt_wr = sum(t.optimized_win_rate for t in result.tickers) / len(result.tickers)
        avg_annual = sum(t.avg_annual_return_pct for t in result.tickers) / len(result.tickers)
        avg_pnl = sum(t.avg_annual_pnl_dollars for t in result.tickers) / len(result.tickers)
        improved = sum(
            1 for t in result.tickers
            if t.optimized_win_rate > t.baseline_win_rate
        )
        print("-" * 130)
        print(
            f"{'AVG':<8} {avg_base_wr:>7.1%} {avg_opt_wr:>7.1%} "
            f"{avg_opt_wr - avg_base_wr:>+6.1%} "
            f"{'':>9} {'':>5} "
            f"{avg_annual:>+7.1f}% "
            f"${avg_pnl:>+8.0f} "
            f"{'':>6} {'':>7}    "
            f"Improved: {improved}/{len(result.tickers)} tickers"
        )
    print("=" * 130 + "\n")


def _print_scalp_report(result: GridSearchResult) -> None:
    """Print a ranked summary table of scalp grid search results."""
    print("\n" + "=" * 100)
    print(f"SCALP OPTIONS GRID SEARCH — {len(result.tickers)} tickers, "
          f"{result.total_trials} trials, {result.elapsed_seconds:.0f}s")
    print("=" * 100)
    print(
        f"{'Ticker':<8} {'Base WR':>8} {'Opt WR':>8} {'Delta':>7} "
        f"{'Base Ret':>9} {'Opt Ret':>9} {'PF':>6} {'Trades':>7} "
        f"{'IBS':>6} {'RSI':>6} {'TP%':>6} {'Hold':>5} {'Score':>7}"
    )
    print("-" * 100)

    for t in result.tickers:
        delta_wr = t.optimized_win_rate - t.baseline_win_rate
        p = t.best_params
        print(
            f"{t.ticker:<8} "
            f"{t.baseline_win_rate:>7.1%} "
            f"{t.optimized_win_rate:>7.1%} "
            f"{delta_wr:>+6.1%} "
            f"{t.baseline_return_pct:>+8.1f}% "
            f"{t.optimized_return_pct:>+8.1f}% "
            f"{t.optimized_profit_factor:>5.2f} "
            f"{t.total_trades:>7} "
            f"{p.get('ibs_thresh', 0.1):>5.2f} "
            f"{p.get('rsi3_thresh', 20):>5.0f} "
            f"{p.get('tp_pct', 0.1) * 100:>5.0f} "
            f"{p.get('max_hold_bars', 6):>5.0f} "
            f"{t.best_score:>6.4f}"
        )

    print("=" * 100 + "\n")


# ── CLI Entry Point ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    from flowedge.config.logging import setup_logging

    setup_logging("INFO")

    engine = sys.argv[1] if len(sys.argv) > 1 else "shares"

    if engine == "shares":
        run_shares_grid_search()
    elif engine == "scalp":
        run_scalp_grid_search()
    elif engine == "all":
        run_shares_grid_search()
        run_scalp_grid_search()
    else:
        print("Usage: python -m flowedge.scanner.backtest.optimizer [shares|scalp|all]")
        sys.exit(1)
