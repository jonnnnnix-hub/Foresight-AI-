"""PHANTOM v2 Performance Simulator — real data historical simulation.

Uses ONLY real Polygon price data via the Grouped Daily endpoint
(one API call per trading day for ALL tickers at once).

v2 Improvements over v1:
  - Black-Scholes option pricing with ATR-derived IV
  - Multi-strategy signals (trend pullback, breakout, mean reversion, vol squeeze)
  - Regime-aware direction (bullish AND bearish trades)
  - Portfolio management (max positions, trailing stops, regime exits)
  - Dynamic position sizing by conviction score
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.backtest.gex_proxy import classify_gex_proxy, compute_gex_adjustment
from flowedge.scanner.backtest.kronos_signal import compute_kronos_adjustment
from flowedge.scanner.backtest.momentum_score import (
    classify_momentum_bias,
    compute_momentum_adjustment,
)
from flowedge.scanner.backtest.pricing import bs_price, estimate_iv_from_atr
from flowedge.scanner.backtest.strategies import (
    EntrySignal,
    MarketRegime,
    compute_indicators,
    detect_regime,
    scan_for_entries,
)
from flowedge.scanner.performance.schemas import (
    DailySnapshot,
    ModelAccuracy,
    MonthlyReturn,
    PerformanceReport,
    SimulatedTrade,
    TickerPerformance,
    TradeResult,
)
from flowedge.scanner.providers.polygon import PolygonProvider

logger = structlog.get_logger()

DATA_DIR = Path("./data/performance")
PERF_FILE = DATA_DIR / "performance.json"
TRADES_FILE = DATA_DIR / "trades.json"
WARMUP_BARS = 55
RISK_FREE_RATE = 0.05
TRADING_DAYS_PER_YEAR = 252


# ── Open Position Tracking ───────────────────────────────────────────


class _OpenPos:
    """Track an open simulated position."""

    __slots__ = (
        "trade", "is_call", "strike", "iv", "dte_at_entry",
        "entry_underlying", "entry_premium", "max_premium",
        "strategy", "regime", "conviction",
    )

    def __init__(
        self,
        trade: SimulatedTrade,
        is_call: bool,
        strike: float,
        iv: float,
        dte_at_entry: int,
        strategy: str,
        regime: str,
        conviction: float,
    ) -> None:
        self.trade = trade
        self.is_call = is_call
        self.strike = strike
        self.iv = iv
        self.dte_at_entry = dte_at_entry
        self.entry_underlying = trade.entry_underlying
        self.entry_premium = trade.entry_premium
        self.max_premium = trade.entry_premium
        self.strategy = strategy
        self.regime = regime
        self.conviction = conviction


# ── Core Simulation ──────────────────────────────────────────────────


async def run_historical_simulation(
    tickers: list[str] | None = None,
    start_date: date = date(2026, 1, 1),
    starting_capital: float = 1000.0,
    max_positions: int = 5,
    max_risk_pct: float = 0.10,
    min_conviction: float = 3.5,
    max_hold_days: int = 12,
    take_profit_pct: float = 250.0,
    stop_loss_pct: float = -50.0,
    trailing_stop_pct: float = 0.40,
    dte: int = 15,
    settings: Settings | None = None,
) -> PerformanceReport:
    """Run simulation using REAL Polygon historical prices.

    Every price in this simulation comes from actual market data.
    """
    settings = settings or get_settings()
    polygon = PolygonProvider(settings)

    if tickers is None:
        tickers = [
            "TSLA", "NVDA", "AAPL", "META", "AMZN",
            "SPY", "QQQ", "AMD", "GOOGL", "MSFT",
        ]

    end_date = date.today()
    cash = starting_capital
    trades: list[SimulatedTrade] = []
    open_positions: list[_OpenPos] = []
    snapshots: list[DailySnapshot] = []
    prev_value = starting_capital
    trade_counter = 0

    try:
        import asyncio

        logger.info("fetching_grouped_daily", start=str(start_date), end=str(end_date))
        prices_by_date: dict[str, dict[str, dict[str, Any]]] = {}
        price_history: dict[str, list[dict[str, Any]]] = {t: [] for t in tickers}

        # Walk through each trading day using grouped daily endpoint
        current = start_date
        while current <= end_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            for attempt in range(3):
                try:
                    grouped = await polygon.get_grouped_daily(current.isoformat())
                    day_str = current.isoformat()
                    day_bars: dict[str, dict[str, Any]] = {}

                    for ticker in tickers:
                        bar = grouped.get(ticker)
                        if bar:
                            full_bar = {
                                "date": day_str,
                                "open": float(bar.get("open", bar.get("close", 0))),
                                "high": float(bar["high"]),
                                "low": float(bar["low"]),
                                "close": float(bar["close"]),
                                "volume": int(bar["volume"]),
                            }
                            day_bars[ticker] = full_bar
                            price_history[ticker].append(full_bar)

                    if day_bars:
                        prices_by_date[day_str] = day_bars

                    logger.info(
                        "grouped_daily_loaded",
                        date=day_str,
                        tickers_found=len(day_bars),
                        source="polygon.io/grouped",
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        wait = 15 * (attempt + 1)
                        logger.info("rate_limit_retry", date=current.isoformat(), wait=f"{wait}s")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(
                            "grouped_fetch_failed",
                            date=current.isoformat(),
                            error=str(e),
                        )
                        break

            await asyncio.sleep(1)  # Paid tier — no rate limit  # Polygon free-tier rate limit
            current += timedelta(days=1)

        sorted_dates = sorted(prices_by_date.keys())
        logger.info(
            "price_data_complete",
            trading_days=len(sorted_dates),
            tickers=len(tickers),
            method="grouped_daily (1 call per date for ALL tickers)",
        )

        # ── Walk each day ────────────────────────────────────────────
        for day_str in sorted_dates:
            day = date.fromisoformat(day_str)
            today_bars = prices_by_date[day_str]
            trades_opened = 0
            trades_closed = 0

            # Check if we have enough warmup
            max_history = max(len(price_history[t]) for t in tickers)
            if max_history < WARMUP_BARS:
                # Still record snapshots during warmup
                snapshots.append(DailySnapshot(
                    date=day,
                    portfolio_value=round(cash, 2),
                    cash=round(cash, 2),
                ))
                prev_value = cash
                continue

            # ── 1. Update open positions + check exits ───────────────
            still_open: list[_OpenPos] = []
            for pos in open_positions:
                bar = today_bars.get(pos.trade.ticker)
                if not bar:
                    still_open.append(pos)
                    continue

                days_held = (day - pos.trade.entry_date).days
                current_underlying = float(bar["close"])

                # Reprice with Black-Scholes
                remaining_dte = max(1, pos.dte_at_entry - days_held)
                t_years = remaining_dte / TRADING_DAYS_PER_YEAR
                current_premium = bs_price(
                    current_underlying, pos.strike, t_years,
                    RISK_FREE_RATE, pos.iv, pos.is_call,
                )

                if current_premium > pos.max_premium:
                    pos.max_premium = current_premium

                pnl_pct = (
                    (current_premium - pos.entry_premium) / pos.entry_premium * 100
                    if pos.entry_premium > 0 else 0.0
                )

                # Exit conditions
                should_exit = False
                exit_reason = ""

                # Hard stop
                if pnl_pct <= stop_loss_pct:
                    should_exit = True
                    exit_reason = f"hard_stop ({pnl_pct:.0f}%)"

                # Take profit
                elif pnl_pct >= take_profit_pct:
                    should_exit = True
                    exit_reason = f"take_profit ({pnl_pct:.0f}%)"

                # Trailing stop
                elif pos.max_premium > pos.entry_premium * 1.20:
                    trail_level = pos.max_premium * (1.0 - trailing_stop_pct)
                    if current_premium <= trail_level:
                        should_exit = True
                        exit_reason = "trailing_stop"

                # Time exit
                elif days_held >= max_hold_days:
                    should_exit = True
                    exit_reason = f"time_exit ({days_held}d)"

                # Regime reversal (check every 3 days to save compute)
                elif days_held >= 3 and days_held % 3 == 0:
                    history = price_history.get(pos.trade.ticker, [])
                    if len(history) >= WARMUP_BARS:
                        ind = compute_indicators(history)
                        regime = detect_regime(ind)
                        if (
                            pos.is_call
                            and regime in (MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND)
                        ) or (
                            not pos.is_call
                            and regime in (MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND)
                        ):
                            should_exit = True
                            exit_reason = "regime_reversal"

                if should_exit:
                    exit_premium = max(0.0, current_premium)
                    exit_value = exit_premium * pos.trade.contracts * 100
                    pnl_dollars = exit_value - pos.trade.cost_basis

                    pos.trade.exit_date = day
                    pos.trade.exit_underlying = current_underlying
                    pos.trade.exit_premium = round(exit_premium, 4)
                    pos.trade.exit_value = round(exit_value, 2)
                    pos.trade.pnl_dollars = round(pnl_dollars, 2)
                    pos.trade.pnl_pct = round(pnl_pct, 2)
                    pos.trade.hold_days = days_held
                    pos.trade.exit_reason = exit_reason
                    pos.trade.result = TradeResult.WIN if pnl_dollars > 0 else TradeResult.LOSS

                    cash += exit_value
                    trades_closed += 1
                else:
                    still_open.append(pos)

            open_positions = still_open

            # ── 2. Scan for new entries ──────────────────────────────
            if len(open_positions) < max_positions and cash > 50:
                all_signals: list[EntrySignal] = []
                regime_counts: dict[str, int] = {}
                raw_signal_count = 0
                for ticker in tickers:
                    history = price_history.get(ticker, [])
                    if len(history) < WARMUP_BARS:
                        continue
                    if any(p.trade.ticker == ticker for p in open_positions):
                        continue

                    ind = compute_indicators(history)
                    regime = detect_regime(ind)
                    regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
                    signals = scan_for_entries(ticker, history, ind, regime)
                    raw_signal_count += len(signals)
                    all_signals.extend(signals)

                all_signals.sort(key=lambda s: s.conviction, reverse=True)

                logger.debug(
                    "signal_scan",
                    date=day_str,
                    regimes=regime_counts,
                    raw_signals=raw_signal_count,
                    post_sort=len(all_signals),
                )

                for signal in all_signals:
                    if len(open_positions) >= max_positions:
                        break
                    if signal.conviction < min_conviction:
                        break

                    bar = today_bars.get(signal.ticker)
                    if not bar:
                        continue

                    underlying = float(bar["close"])
                    is_call = signal.direction == "bullish"
                    otm_mult = signal.otm_pct if is_call else -signal.otm_pct
                    strike = underlying * (1.0 + otm_mult)

                    # Estimate IV from ATR
                    history = price_history.get(signal.ticker, [])
                    ind = compute_indicators(history)
                    iv = estimate_iv_from_atr(ind.atr14, underlying)

                    # Multi-factor conviction adjustment
                    closes = [float(b.get("close", 0)) for b in history]
                    m_bias, m_score = classify_momentum_bias(ind, closes)

                    # Pullback/reversion strategies EXPECT opposing RSI — the low RSI
                    # is what triggered them. Don't penalize them for it.
                    REVERSION_STRATEGIES = {
                        "trend_pullback", "ibs_reversion", "mean_reversion",
                    }
                    if signal.strategy in REVERSION_STRATEGIES:
                        # Only apply positive momentum adjustment; ignore negatives
                        raw_m_adj = compute_momentum_adjustment(
                            m_bias, m_score, signal.direction,
                        )
                        m_adj = max(0.0, raw_m_adj)
                    else:
                        m_adj = compute_momentum_adjustment(
                            m_bias, m_score, signal.direction,
                        )

                    g_regime, g_score = classify_gex_proxy(ind, history)
                    g_adj = compute_gex_adjustment(
                        g_regime, g_score, signal.direction,
                    )
                    k_adj = compute_kronos_adjustment(history, signal.direction)

                    # Cap total negative adjustment at -1.5 so three simultaneous
                    # opposing signals can't veto an otherwise valid setup
                    total_adj = m_adj + g_adj + k_adj
                    total_adj = max(-1.5, total_adj)

                    adjusted = signal.conviction + total_adj
                    signal.conviction = max(0.0, min(10.0, adjusted))
                    if signal.conviction < min_conviction:
                        logger.debug(
                            "signal_killed_by_adjustments",
                            date=day_str,
                            ticker=signal.ticker,
                            strategy=signal.strategy,
                            regime=signal.regime,
                            final_conviction=round(signal.conviction, 2),
                            m_adj=round(m_adj, 2),
                            g_adj=round(g_adj, 2),
                            k_adj=round(k_adj, 2),
                        )
                        continue

                    # Black-Scholes premium
                    t_years = max(dte, 1) / TRADING_DAYS_PER_YEAR
                    premium = bs_price(underlying, strike, t_years, RISK_FREE_RATE, iv, is_call)

                    if premium < 0.05:
                        continue

                    # Dynamic position sizing by conviction
                    pos_val = sum(
                        max(0, p.trade.entry_premium * p.trade.contracts * 100)
                        for p in open_positions
                    )
                    total_value = cash + pos_val
                    budget = total_value * max_risk_pct * (0.3 + 0.7 * signal.conviction / 10.0)
                    contracts = max(1, int(budget / (premium * 100)))
                    cost = contracts * premium * 100

                    if cost > cash * 0.90:
                        contracts = max(1, int(cash * 0.85 / (premium * 100)))
                        cost = contracts * premium * 100

                    if cost > cash or cost < 10:
                        continue

                    trade_counter += 1
                    # Convert 0-10 conviction to 0-100 nexus score
                    nexus_100 = min(100, round(signal.conviction * 10))

                    trade = SimulatedTrade(
                        trade_id=f"SIM-{trade_counter:04d}",
                        ticker=signal.ticker,
                        direction=signal.direction,
                        entry_date=day,
                        option_type="call" if is_call else "put",
                        strike=round(strike, 2),
                        expiration=day + timedelta(days=dte),
                        entry_underlying=underlying,
                        entry_premium=round(premium, 4),
                        contracts=contracts,
                        cost_basis=round(cost, 2),
                        nexus_score=nexus_100,
                    )
                    open_positions.append(_OpenPos(
                        trade=trade,
                        is_call=is_call,
                        strike=round(strike, 2),
                        iv=iv,
                        dte_at_entry=dte,
                        strategy=signal.strategy,
                        regime=signal.regime,
                        conviction=signal.conviction,
                    ))
                    trades.append(trade)
                    cash -= cost
                    trades_opened += 1

            # ── 3. Daily snapshot ────────────────────────────────────
            open_value = 0.0
            for pos in open_positions:
                bar = today_bars.get(pos.trade.ticker)
                if not bar:
                    open_value += max(0.0, pos.entry_premium * pos.trade.contracts * 100)
                    continue

                days_held = (day - pos.trade.entry_date).days
                remaining_dte = max(1, pos.dte_at_entry - days_held)
                t_years = remaining_dte / TRADING_DAYS_PER_YEAR
                current_premium = bs_price(
                    float(bar["close"]), pos.strike, t_years,
                    RISK_FREE_RATE, pos.iv, pos.is_call,
                )
                open_value += max(0.0, current_premium * pos.trade.contracts * 100)

            portfolio_value = round(cash + open_value, 2)
            daily_pnl = round(portfolio_value - prev_value, 2)
            daily_return = round(daily_pnl / prev_value * 100 if prev_value > 0 else 0, 2)
            cum_return = round(
                (portfolio_value - starting_capital) / starting_capital * 100, 2
            )

            snapshots.append(DailySnapshot(
                date=day,
                portfolio_value=portfolio_value,
                cash=round(cash, 2),
                open_positions_value=round(open_value, 2),
                daily_pnl=daily_pnl,
                daily_return_pct=daily_return,
                cumulative_return_pct=cum_return,
                trades_opened=trades_opened,
                trades_closed=trades_closed,
            ))
            prev_value = portfolio_value

    finally:
        await polygon.close()

    # ── Aggregate Stats ──────────────────────────────────────────────
    closed = [t for t in trades if t.result != TradeResult.OPEN]
    wins = [t for t in closed if t.result == TradeResult.WIN]
    losses = [t for t in closed if t.result in (TradeResult.LOSS, TradeResult.EXPIRED)]
    still_open_trades = [t for t in trades if t.result == TradeResult.OPEN]

    gross_profit = sum(t.pnl_dollars for t in wins)
    gross_loss = abs(sum(t.pnl_dollars for t in losses))
    ending = snapshots[-1].portfolio_value if snapshots else starting_capital

    peak = starting_capital
    max_dd = 0.0
    for snap in snapshots:
        if snap.portfolio_value > peak:
            peak = snap.portfolio_value
        dd_val = (peak - snap.portfolio_value) / peak * 100
        if dd_val > max_dd:
            max_dd = dd_val

    buckets: dict[str, dict[str, float]] = {}
    for bucket_name, lo, hi in [
        ("0-30", 0, 30), ("30-50", 30, 50),
        ("50-70", 50, 70), ("70-100", 70, 100),
    ]:
        bt = [t for t in closed if lo <= t.nexus_score < hi]
        if bt:
            bw = sum(1 for t in bt if t.result == TradeResult.WIN)
            buckets[bucket_name] = {
                "trades": float(len(bt)),
                "win_rate": round(bw / len(bt), 3),
                "avg_pnl": round(sum(t.pnl_pct for t in bt) / len(bt), 2),
                "total_pnl": round(sum(t.pnl_dollars for t in bt), 2),
            }

    monthly_returns = _compute_monthly_returns(snapshots, trades, starting_capital)
    by_ticker = _compute_ticker_performance(closed)
    model_accuracy = _compute_model_accuracy(closed, snapshots, max_dd, starting_capital)

    report = PerformanceReport(
        start_date=start_date,
        end_date=end_date,
        starting_capital=starting_capital,
        ending_value=round(ending, 2),
        total_return_dollars=round(ending - starting_capital, 2),
        total_return_pct=round((ending - starting_capital) / starting_capital * 100, 2),
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        open_trades=len(still_open_trades),
        win_rate=round(len(wins) / len(closed), 3) if closed else 0.0,
        avg_win_pct=round(sum(t.pnl_pct for t in wins) / len(wins), 2) if wins else 0.0,
        avg_loss_pct=round(sum(t.pnl_pct for t in losses) / len(losses), 2) if losses else 0.0,
        best_trade_pct=round(max((t.pnl_pct for t in closed), default=0), 2),
        worst_trade_pct=round(min((t.pnl_pct for t in closed), default=0), 2),
        profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
        max_drawdown_pct=round(max_dd, 2),
        avg_hold_days=round(sum(t.hold_days for t in closed) / len(closed), 1) if closed else 0.0,
        daily_snapshots=snapshots,
        trades=trades,
        by_score_bucket=buckets,
        monthly_returns=monthly_returns,
        by_ticker=by_ticker,
        model_accuracy=model_accuracy,
    )

    _save_report(report)
    logger.info(
        "simulation_complete",
        data_source="polygon.io (real)",
        start=str(start_date),
        end=str(end_date),
        trades=len(trades),
        return_pct=report.total_return_pct,
        ending_value=report.ending_value,
    )
    return report


# ── Helpers ──────────────────────────────────────────────────────────


def _compute_monthly_returns(
    snapshots: list[DailySnapshot],
    trades: list[SimulatedTrade],
    starting_capital: float,
) -> list[MonthlyReturn]:
    """Break down performance by calendar month."""
    if not snapshots:
        return []

    months: dict[str, list[DailySnapshot]] = {}
    for snap in snapshots:
        key = snap.date.strftime("%Y-%m")
        months.setdefault(key, []).append(snap)

    results: list[MonthlyReturn] = []
    for month_key in sorted(months):
        month_snaps = months[month_key]
        first = month_snaps[0]
        last = month_snaps[-1]

        month_trades_opened = sum(
            1 for t in trades if t.entry_date.strftime("%Y-%m") == month_key
        )
        month_trades_closed = [
            t for t in trades
            if t.exit_date and t.exit_date.strftime("%Y-%m") == month_key
            and t.result != TradeResult.OPEN
        ]
        month_wins = sum(1 for t in month_trades_closed if t.result == TradeResult.WIN)
        month_losses = len(month_trades_closed) - month_wins

        start_val = first.portfolio_value
        end_val = last.portfolio_value
        ret_pct = (end_val - start_val) / start_val * 100 if start_val > 0 else 0

        results.append(MonthlyReturn(
            month=month_key,
            starting_value=round(start_val, 2),
            ending_value=round(end_val, 2),
            return_pct=round(ret_pct, 2),
            return_dollars=round(end_val - start_val, 2),
            trades_opened=month_trades_opened,
            trades_closed=len(month_trades_closed),
            wins=month_wins,
            losses=month_losses,
            win_rate=(
                round(month_wins / len(month_trades_closed), 3)
                if month_trades_closed else 0.0
            ),
        ))

    return results


def _compute_ticker_performance(closed: list[SimulatedTrade]) -> list[TickerPerformance]:
    """Break down performance by ticker."""
    by_ticker: dict[str, list[SimulatedTrade]] = {}
    for t in closed:
        by_ticker.setdefault(t.ticker, []).append(t)

    results: list[TickerPerformance] = []
    for ticker in sorted(by_ticker):
        ticker_trades = by_ticker[ticker]
        tw = sum(1 for t in ticker_trades if t.result == TradeResult.WIN)
        tl = len(ticker_trades) - tw
        pnls = [t.pnl_pct for t in ticker_trades]

        results.append(TickerPerformance(
            ticker=ticker,
            total_trades=len(ticker_trades),
            wins=tw,
            losses=tl,
            win_rate=round(tw / len(ticker_trades), 3) if ticker_trades else 0.0,
            total_pnl_dollars=round(sum(t.pnl_dollars for t in ticker_trades), 2),
            avg_pnl_pct=round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
            best_trade_pct=round(max(pnls, default=0), 2),
            worst_trade_pct=round(min(pnls, default=0), 2),
        ))

    results.sort(key=lambda x: x.total_pnl_dollars, reverse=True)
    return results


def _compute_model_accuracy(
    closed: list[SimulatedTrade],
    snapshots: list[DailySnapshot],
    max_dd: float,
    starting_capital: float,
) -> ModelAccuracy:
    """Compute model prediction accuracy and risk-adjusted metrics."""
    import math

    if not closed:
        return ModelAccuracy()

    correct = sum(
        1 for t in closed
        if (t.direction == "bullish" and t.pnl_dollars > 0)
        or (t.direction == "bearish" and t.pnl_dollars > 0)
    )
    total = len(closed)

    winners = [t for t in closed if t.result == TradeResult.WIN]
    losers = [t for t in closed if t.result != TradeResult.WIN]
    avg_score_w = sum(t.nexus_score for t in winners) / len(winners) if winners else 0
    avg_score_l = sum(t.nexus_score for t in losers) / len(losers) if losers else 0

    high_score = [t for t in closed if t.nexus_score >= 60]
    low_score = [t for t in closed if t.nexus_score < 40]
    hs_wr = (
        sum(1 for t in high_score if t.result == TradeResult.WIN)
        / len(high_score) if high_score else 0
    )
    ls_wr = (
        sum(1 for t in low_score if t.result == TradeResult.WIN)
        / len(low_score) if low_score else 0
    )

    daily_returns = [s.daily_return_pct / 100 for s in snapshots if s.daily_return_pct != 0]
    if daily_returns:
        avg_ret = sum(daily_returns) / len(daily_returns)
        variance = sum((r - avg_ret) ** 2 for r in daily_returns) / max(len(daily_returns) - 1, 1)
        std_ret = math.sqrt(variance)
        sharpe = (avg_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0
    else:
        sharpe = 0.0

    neg_returns = [r for r in daily_returns if r < 0]
    if neg_returns:
        avg_ret = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        downside_dev = math.sqrt(sum(r ** 2 for r in neg_returns) / len(neg_returns))
        sortino = (avg_ret / downside_dev) * math.sqrt(252) if downside_dev > 0 else 0
    else:
        sortino = 0.0

    if snapshots and max_dd > 0:
        total_days = (snapshots[-1].date - snapshots[0].date).days
        total_return = (snapshots[-1].portfolio_value - starting_capital) / starting_capital
        annualized = total_return * (365 / max(total_days, 1))
        calmar = annualized / (max_dd / 100) if max_dd > 0 else 0
    else:
        calmar = 0.0

    expectancy = sum(t.pnl_dollars for t in closed) / len(closed) if closed else 0

    max_w_streak = 0
    max_l_streak = 0
    current_streak = 0
    last_result: TradeResult | None = None
    for t in sorted(closed, key=lambda x: x.entry_date):
        if t.result == last_result:
            current_streak += 1
        else:
            current_streak = 1
            last_result = t.result
        if last_result == TradeResult.WIN:
            max_w_streak = max(max_w_streak, current_streak)
        elif last_result in (TradeResult.LOSS, TradeResult.EXPIRED):
            max_l_streak = max(max_l_streak, current_streak)

    return ModelAccuracy(
        total_predictions=total,
        correct_direction=correct,
        direction_accuracy=round(correct / total, 3) if total > 0 else 0.0,
        avg_score_winners=round(avg_score_w, 1),
        avg_score_losers=round(avg_score_l, 1),
        score_separation=round(avg_score_w - avg_score_l, 1),
        high_score_win_rate=round(hs_wr, 3),
        low_score_win_rate=round(ls_wr, 3),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        calmar_ratio=round(calmar, 3),
        expectancy=round(expectancy, 2),
        consecutive_wins_max=max_w_streak,
        consecutive_losses_max=max_l_streak,
    )


def _save_report(report: PerformanceReport) -> None:
    """Save performance report to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PERF_FILE.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, default=str)
    )
    logger.info("performance_saved", path=str(PERF_FILE))


def load_report() -> PerformanceReport | None:
    """Load the last saved performance report."""
    if not PERF_FILE.exists():
        return None
    try:
        data = json.loads(PERF_FILE.read_text())
        return PerformanceReport.model_validate(data)
    except Exception as e:
        logger.warning("performance_load_failed", error=str(e))
        return None
