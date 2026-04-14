"""Tournament engine — runs all 5 models + ensemble on the same bar data.

Each model gets its own Portfolio instance and independently decides
entries and exits. The ensemble uses consensus-weighted sizing.
"""

from __future__ import annotations

import uuid
from math import sqrt
from typing import Any

import structlog

from flowedge.scanner.backtest.engine import (
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    WARMUP_BARS,
    OpenPosition,
    Portfolio,
)
from flowedge.scanner.backtest.pricing import estimate_iv_from_atr
from flowedge.scanner.backtest.schemas import TradeOutcome
from flowedge.scanner.backtest.strategies import (
    EntrySignal,
    MarketRegime,
    compute_indicators,
    detect_regime,
    scan_for_entries,
)
from flowedge.scanner.tournament.models import (
    StopConfig,
    TournamentModel,
    build_all_models,
    build_regime_chameleon,
    compute_category_scores,
)
from flowedge.scanner.tournament.schemas import (
    ConsensusEntry,
    ModelName,
    ModelPerformance,
    TournamentResult,
)

logger = structlog.get_logger()


# ── Exit Logic (per-model stops) ───────────────────────────────────


def _check_model_exits(
    portfolio: Portfolio,
    today_bars: dict[str, dict[str, Any]],
    ticker_history: dict[str, list[dict[str, Any]]],
    current_date: str,
    stop_config: StopConfig,
    *,
    no_followthrough_days: int | None = None,
) -> None:
    """Check exits using model-specific stop parameters."""
    to_close: list[tuple[OpenPosition, str]] = []

    for pos in portfolio.positions:
        bar = today_bars.get(pos.ticker)
        if not bar:
            continue

        if pos.entry_premium <= 0:
            to_close.append((pos, "invalid"))
            continue

        pnl_pct = (pos.current_premium - pos.entry_premium) / pos.entry_premium

        # 1. Hard stop
        if pnl_pct <= stop_config.hard_stop_pct:
            to_close.append((pos, "hard_stop"))
            continue

        # 2. Take profit
        if pnl_pct >= stop_config.take_profit_pct:
            to_close.append((pos, "take_profit"))
            continue

        # 3. Trailing stop
        if pos.max_premium > pos.entry_premium * 1.20:
            trail_level = pos.max_premium * (1.0 - stop_config.trailing_stop_pct)
            if pos.current_premium <= trail_level:
                to_close.append((pos, "trailing_stop"))
                continue

        # 4. Time exit
        if pos.days_held >= stop_config.max_hold_days:
            to_close.append((pos, "time_exit"))
            continue

        # 5. No-followthrough exit (FLOW_HUNTER)
        if (
            no_followthrough_days is not None
            and pos.days_held >= no_followthrough_days
            and pnl_pct <= 0.0
        ):
            to_close.append((pos, "no_followthrough"))
            continue

        # 6. Regime reversal
        history = ticker_history.get(pos.ticker, [])
        if len(history) >= WARMUP_BARS and pos.days_held >= 3:
            indicators = compute_indicators(history)
            regime = detect_regime(indicators)
            if pos.is_call and regime in (
                MarketRegime.DOWNTREND,
                MarketRegime.STRONG_DOWNTREND,
            ):
                to_close.append((pos, "regime_reversal"))
                continue
            if not pos.is_call and regime in (
                MarketRegime.UPTREND,
                MarketRegime.STRONG_UPTREND,
            ):
                to_close.append((pos, "regime_reversal"))
                continue

    for pos, reason in to_close:
        if pos in portfolio.positions:
            portfolio.close_position(pos, current_date, reason)


# ── Performance Computation ────────────────────────────────────────


def _compute_max_drawdown(daily_values: list[tuple[str, float]]) -> float:
    if not daily_values:
        return 0.0
    peak = daily_values[0][1]
    max_dd = 0.0
    for _, val in daily_values:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


def _compute_sharpe(daily_values: list[tuple[str, float]]) -> float:
    if len(daily_values) < 10:
        return 0.0
    daily_returns: list[float] = []
    for i in range(1, len(daily_values)):
        prev = daily_values[i - 1][1]
        curr = daily_values[i][1]
        if prev > 0:
            daily_returns.append((curr - prev) / prev)
    if not daily_returns:
        return 0.0
    mean_r = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean_r) ** 2 for r in daily_returns) / len(daily_returns)
    std_r = sqrt(variance) if variance > 0 else 0.001
    annual_return = mean_r * TRADING_DAYS_PER_YEAR
    annual_std = std_r * sqrt(TRADING_DAYS_PER_YEAR)
    if annual_std <= 0:
        return 0.0
    return round((annual_return - RISK_FREE_RATE) / annual_std, 3)


def _build_model_performance(
    model_name: str,
    portfolio: Portfolio,
    starting_capital: float,
) -> ModelPerformance:
    """Build ModelPerformance from a portfolio's closed trades and daily values."""
    trades = portfolio.closed_trades
    total = len(trades)
    wins = sum(1 for t in trades if t.outcome == TradeOutcome.WIN)
    losses = total - wins

    win_pnls = [t.pnl_pct for t in trades if t.outcome == TradeOutcome.WIN]
    loss_pnls = [t.pnl_pct for t in trades if t.outcome != TradeOutcome.WIN]

    gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))

    ending = portfolio.total_value
    total_return = (ending - starting_capital) / starting_capital * 100

    return ModelPerformance(
        model_name=model_name,
        total_return_pct=round(total_return, 2),
        sharpe_ratio=_compute_sharpe(portfolio.daily_values),
        win_rate=round(wins / total, 3) if total > 0 else 0.0,
        profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
        total_trades=total,
        wins=wins,
        losses=losses,
        avg_win_pct=round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0.0,
        avg_loss_pct=round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0,
        max_drawdown_pct=_compute_max_drawdown(portfolio.daily_values),
        avg_hold_days=round(sum(t.hold_days for t in trades) / total, 1) if total > 0 else 0.0,
        best_trade_pct=round(max((t.pnl_pct for t in trades), default=0.0), 2),
        worst_trade_pct=round(min((t.pnl_pct for t in trades), default=0.0), 2),
        starting_capital=starting_capital,
        ending_value=round(ending, 2),
    )


# ── Tournament Runner ──────────────────────────────────────────────


def run_tournament_on_bars(
    all_bars: dict[str, list[dict[str, Any]]],
    starting_capital: float = 10_000.0,
    max_positions: int = 5,
    dte: int = 15,
    min_conviction: float = 5.0,
) -> TournamentResult:
    """Run all 5 models + ensemble on pre-fetched bar data.

    This is the core tournament logic, separated from data fetching
    so it can be tested with synthetic data.

    Args:
        all_bars: Dict of ticker → list of OHLCV bars (sorted ascending).
        starting_capital: Initial capital per model.
        max_positions: Max concurrent positions per model.
        dte: Days to expiration for options.
        min_conviction: Minimum conviction to consider a signal.
    """
    if not all_bars:
        return TournamentResult(
            run_id=str(uuid.uuid4())[:12],
            starting_capital=starting_capital,
        )

    # Build date-indexed structure
    bars_by_date: dict[str, dict[str, dict[str, Any]]] = {}
    all_dates: set[str] = set()
    for ticker, bars in all_bars.items():
        for bar in bars:
            d: str = bar["date"]
            all_dates.add(d)
            bars_by_date.setdefault(d, {})[ticker] = bar

    sorted_dates = sorted(all_dates)

    # Initialize one portfolio per model + ensemble
    model_names = [
        ModelName.EDGE_CORE,
        ModelName.MOMENTUM_ALPHA,
        ModelName.FLOW_HUNTER,
        ModelName.CONTRARIAN_EDGE,
        ModelName.REGIME_CHAMELEON,
        ModelName.ENSEMBLE,
    ]
    portfolios: dict[str, Portfolio] = {}
    for mn in model_names:
        portfolios[mn.value] = Portfolio(
            cash=starting_capital,
            initial_capital=starting_capital,
            max_positions=max_positions,
        )

    ticker_history: dict[str, list[dict[str, Any]]] = {t: [] for t in all_bars}
    consensus_entries: list[ConsensusEntry] = []

    # Walk each day
    for current_date in sorted_dates:
        today_bars = bars_by_date.get(current_date, {})

        for ticker in all_bars:
            if ticker in today_bars:
                ticker_history[ticker].append(today_bars[ticker])

        max_history = max(len(ticker_history[t]) for t in all_bars)
        if max_history < WARMUP_BARS:
            continue

        # Update all portfolios
        for pf in portfolios.values():
            pf.update_positions(today_bars)

        # Build models with current regime info (for chameleon)
        # Use first ticker with enough history to detect regime
        current_regime = MarketRegime.SIDEWAYS
        for ticker in all_bars:
            hist = ticker_history.get(ticker, [])
            if len(hist) >= WARMUP_BARS:
                ind = compute_indicators(hist)
                current_regime = detect_regime(ind)
                break

        base_models = build_all_models(current_regime)
        model_map: dict[str, TournamentModel] = {m.name.value: m for m in base_models}

        # Check exits for each model
        for mn in model_names:
            if mn == ModelName.ENSEMBLE:
                # Ensemble uses EDGE_CORE stops
                sc = model_map[ModelName.EDGE_CORE.value].stop_config
                _check_model_exits(
                    portfolios[mn.value], today_bars, ticker_history,
                    current_date, sc,
                )
            else:
                model = model_map[mn.value]
                _check_model_exits(
                    portfolios[mn.value], today_bars, ticker_history,
                    current_date, model.stop_config,
                    no_followthrough_days=model.no_followthrough_days,
                )

        # Scan for entries
        for ticker in all_bars:
            history = ticker_history.get(ticker, [])
            if len(history) < WARMUP_BARS:
                continue

            indicators = compute_indicators(history)
            regime = detect_regime(indicators)
            category_scores = compute_category_scores(indicators, regime)

            # Get base signals from the existing strategy scanner
            signals = scan_for_entries(ticker, history, indicators, regime)
            if not signals:
                continue

            best_signal = signals[0]
            if best_signal.conviction < min_conviction:
                continue

            # Update chameleon for this ticker's regime
            model_map[ModelName.REGIME_CHAMELEON.value] = build_regime_chameleon(
                regime, base_models[:4],
            )

            # Score with each model
            model_scores: dict[str, float] = {}
            model_enter: dict[str, bool] = {}
            for mn_val, model in model_map.items():
                score = model.score_setup(indicators, regime, category_scores)
                model_scores[mn_val] = score
                model_enter[mn_val] = model.should_enter(score, indicators)

            # Track consensus
            agreeing = [mn for mn, enters in model_enter.items() if enters]
            disagreeing = [mn for mn, enters in model_enter.items() if not enters]
            consensus_level = len(agreeing)

            if agreeing:
                avg_score = sum(model_scores[mn] for mn in agreeing) / len(agreeing)
                consensus_entries.append(
                    ConsensusEntry(
                        ticker=ticker,
                        date=current_date,
                        models_agreeing=agreeing,
                        models_disagreeing=disagreeing,
                        consensus_score=round(avg_score, 2),
                        consensus_level=min(consensus_level, 5),
                        direction=best_signal.direction,
                    )
                )

            # Execute entries per model
            sig_bar = today_bars.get(ticker)
            if not sig_bar:
                continue

            iv = estimate_iv_from_atr(indicators.atr14, float(sig_bar["close"]))

            for mn_val, enters in model_enter.items():
                if not enters:
                    continue
                pf = portfolios[mn_val]
                if not pf.can_open():
                    continue
                if any(p.ticker == ticker for p in pf.positions):
                    continue
                pf.open_position(best_signal, sig_bar, iv, dte=dte)

            # Ensemble entry: consensus-weighted sizing
            if consensus_level >= 2:
                ens_pf = portfolios[ModelName.ENSEMBLE.value]
                if ens_pf.can_open() and not any(
                    p.ticker == ticker for p in ens_pf.positions
                ):
                    # Boost conviction by consensus level
                    ens_signal = EntrySignal(
                        ticker=best_signal.ticker,
                        direction=best_signal.direction,
                        strategy=best_signal.strategy,
                        conviction=min(10.0, best_signal.conviction * consensus_level / 3.0),
                        regime=best_signal.regime,
                        otm_pct=best_signal.otm_pct,
                        reason=f"ensemble|consensus={consensus_level}",
                    )
                    ens_pf.open_position(ens_signal, sig_bar, iv, dte=dte)

        # Record snapshots
        for pf in portfolios.values():
            pf.record_snapshot(current_date)

    # Close remaining positions
    if sorted_dates:
        last_date = sorted_dates[-1]
        for pf in portfolios.values():
            for pos in pf.positions[:]:
                pf.close_position(pos, last_date, "end_of_tournament")

    # Build results
    model_results: dict[str, ModelPerformance] = {}
    for mn in model_names:
        if mn == ModelName.ENSEMBLE:
            continue
        perf = _build_model_performance(mn.value, portfolios[mn.value], starting_capital)
        model_results[mn.value] = perf

    ensemble_perf = _build_model_performance(
        ModelName.ENSEMBLE.value, portfolios[ModelName.ENSEMBLE.value], starting_capital,
    )

    # Consensus analysis
    avg_consensus = 0.0
    high_consensus_wins = 0
    high_consensus_total = 0
    if consensus_entries:
        avg_consensus = sum(c.consensus_level for c in consensus_entries) / len(consensus_entries)
        for ce in consensus_entries:
            if ce.consensus_level >= 4:
                high_consensus_total += 1
                # Check if ensemble had a winning trade for this ticker/date
                for t in portfolios[ModelName.ENSEMBLE.value].closed_trades:
                    if t.ticker == ce.ticker and t.entry_date.isoformat() == ce.date:
                        if t.outcome == TradeOutcome.WIN:
                            high_consensus_wins += 1
                        break

    # Rankings
    sorted_by_return = sorted(
        model_results.values(), key=lambda m: m.total_return_pct, reverse=True,
    )
    sorted_by_sharpe = sorted(
        model_results.values(), key=lambda m: m.sharpe_ratio, reverse=True,
    )
    sorted_by_wr = sorted(
        model_results.values(), key=lambda m: m.win_rate, reverse=True,
    )

    result = TournamentResult(
        run_id=str(uuid.uuid4())[:12],
        tickers=list(all_bars.keys()),
        lookback_days=len(sorted_dates),
        starting_capital=starting_capital,
        model_results=model_results,
        ensemble_result=ensemble_perf,
        consensus_entries=consensus_entries,
        avg_consensus_level=round(avg_consensus, 2),
        high_consensus_win_rate=(
            round(high_consensus_wins / high_consensus_total, 3)
            if high_consensus_total > 0
            else 0.0
        ),
        ranking_by_return=[m.model_name for m in sorted_by_return],
        ranking_by_sharpe=[m.model_name for m in sorted_by_sharpe],
        ranking_by_win_rate=[m.model_name for m in sorted_by_wr],
    )

    logger.info(
        "tournament_complete",
        models=len(model_results),
        tickers=list(all_bars.keys()),
        ensemble_return=f"{ensemble_perf.total_return_pct:.1f}%",
        best_model=result.ranking_by_return[0] if result.ranking_by_return else "none",
    )
    return result
