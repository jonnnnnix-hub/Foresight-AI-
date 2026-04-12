"""Index ETF specialist backtester.

Dedicated engine for SPY/QQQ/IWM that runs three sub-models:
1. Scalp mode — 1-3 day holds, 70%+ WR target, 50%+ gain target
2. Swing mode — 4-10 day holds, 80%+ WR target
3. Medium mode — 10-20 day holds, 90% WR target

Key differences from main engine:
- Much higher conviction thresholds (8.0-8.5 minimum)
- Monte Carlo probability filter (must pass 55%+ P(profit) for scalps)
- Tighter OTM selection (1-2% vs 2.5% for single names)
- Horizon-specific stop/profit profiles
- Self-learning feedback loop with per-horizon weight tuning
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from math import sqrt
from pathlib import Path
from typing import Any

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.backtest.gex_proxy import classify_gex_proxy, compute_gex_adjustment
from flowedge.scanner.backtest.index_specialist.schemas import (
    IndexBacktestConfig,
    IndexBacktestResult,
    IndexSignal,
    IndexTradeResult,
    TradeHorizon,
)
from flowedge.scanner.backtest.index_specialist.strategies import (
    classify_index_regime,
    scan_index_entries,
)
from flowedge.scanner.backtest.kronos_signal import compute_kronos_adjustment
from flowedge.scanner.backtest.momentum_score import (
    classify_momentum_bias,
    compute_momentum_adjustment,
)
from flowedge.scanner.backtest.monte_carlo import compute_mc_conviction
from flowedge.scanner.backtest.pricing import bs_price, estimate_iv_from_atr
from flowedge.scanner.backtest.strategies import compute_indicators
from flowedge.scanner.providers.polygon import PolygonProvider

logger = structlog.get_logger()

WARMUP_BARS = 55
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05


# ── Portfolio for Index Trading ──────────────────────────────────────


@dataclass
class IndexPosition:
    """An open index option position."""

    ticker: str
    direction: str
    horizon: TradeHorizon
    strategy: str
    is_call: bool
    entry_date: str
    entry_underlying: float
    strike: float
    entry_premium: float
    contracts: int
    cost_basis: float
    iv: float
    conviction: float
    regime: str
    dte_at_entry: int
    mc_prob_profit: float = 0.0
    days_held: int = 0
    max_premium: float = 0.0
    current_premium: float = 0.0
    current_underlying: float = 0.0

    def __post_init__(self) -> None:
        self.max_premium = self.entry_premium
        self.current_premium = self.entry_premium
        self.current_underlying = self.entry_underlying


@dataclass
class IndexPortfolio:
    """Portfolio manager for index-only trading."""

    cash: float
    initial_capital: float
    max_positions: int = 3
    max_risk_pct: float = 0.10
    positions: list[IndexPosition] = field(default_factory=list)
    closed_trades: list[IndexTradeResult] = field(default_factory=list)
    daily_values: list[tuple[str, float]] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        pos_value = sum(p.current_premium * p.contracts * 100 for p in self.positions)
        return self.cash + max(0.0, pos_value)

    def can_open(self) -> bool:
        return len(self.positions) < self.max_positions

    def position_size(self, conviction: float) -> float:
        base = self.total_value * self.max_risk_pct
        scale = 0.4 + 0.6 * (conviction / 10.0)
        return base * scale

    def open_position(
        self,
        signal: IndexSignal,
        bar: dict[str, Any],
        iv: float,
        dte: int = 15,
    ) -> IndexPosition | None:
        if not self.can_open():
            return None

        is_call = signal.direction == "bullish"
        close = float(bar["close"])
        otm = signal.otm_pct if is_call else -signal.otm_pct
        strike = close * (1.0 + otm)

        t_years = max(dte, 1) / TRADING_DAYS_PER_YEAR
        premium = bs_price(close, strike, t_years, RISK_FREE_RATE, iv, is_call)
        if premium < 0.05:
            return None

        budget = self.position_size(signal.conviction)
        contracts = max(1, int(budget / (premium * 100)))
        cost = contracts * premium * 100

        if cost > self.cash * 0.90:
            contracts = max(1, int(self.cash * 0.85 / (premium * 100)))
            cost = contracts * premium * 100

        if cost > self.cash or cost < 10:
            return None

        self.cash -= cost
        pos = IndexPosition(
            ticker=signal.ticker,
            direction=signal.direction,
            horizon=signal.horizon,
            strategy=signal.strategy,
            is_call=is_call,
            entry_date=bar["date"],
            entry_underlying=close,
            strike=round(strike, 2),
            entry_premium=premium,
            contracts=contracts,
            cost_basis=cost,
            iv=iv,
            conviction=signal.conviction,
            regime=signal.regime.value,
            dte_at_entry=dte,
            mc_prob_profit=signal.mc_prob_profit,
        )
        self.positions.append(pos)
        return pos

    def close_position(
        self, pos: IndexPosition, current_date: str, reason: str,
    ) -> IndexTradeResult:
        exit_premium = max(0.0, pos.current_premium)
        exit_value = exit_premium * pos.contracts * 100
        pnl = exit_value - pos.cost_basis
        pnl_pct = (pnl / pos.cost_basis * 100) if pos.cost_basis > 0 else 0.0

        self.cash += exit_value
        if pos in self.positions:
            self.positions.remove(pos)

        result = IndexTradeResult(
            ticker=pos.ticker,
            direction=pos.direction,
            horizon=pos.horizon,
            entry_date=date.fromisoformat(pos.entry_date),
            exit_date=date.fromisoformat(current_date),
            entry_underlying=pos.entry_underlying,
            exit_underlying=pos.current_underlying,
            strike=pos.strike,
            entry_premium=round(pos.entry_premium, 4),
            exit_premium=round(exit_premium, 4),
            pnl_pct=round(pnl_pct, 2),
            pnl_dollars=round(pnl, 2),
            hold_days=pos.days_held,
            exit_reason=reason,
            conviction=round(pos.conviction, 2),
            regime=pos.regime,
            strategy=pos.strategy,
            mc_prob_profit=pos.mc_prob_profit,
            is_win=pnl_pct >= 10.0,
        )
        self.closed_trades.append(result)
        return result

    def update_positions(self, bars_by_ticker: dict[str, dict[str, Any]]) -> None:
        for pos in self.positions:
            bar = bars_by_ticker.get(pos.ticker)
            if not bar:
                continue
            pos.days_held += 1
            pos.current_underlying = float(bar["close"])
            remaining_dte = max(1, pos.dte_at_entry - pos.days_held)
            t_years = remaining_dte / TRADING_DAYS_PER_YEAR
            pos.current_premium = bs_price(
                pos.current_underlying, pos.strike, t_years,
                RISK_FREE_RATE, pos.iv, pos.is_call,
            )
            if pos.current_premium > pos.max_premium:
                pos.max_premium = pos.current_premium

    def record_snapshot(self, date_str: str) -> None:
        self.daily_values.append((date_str, self.total_value))


# ── Exit Logic ──────────────────────────────────────────────────────


def _get_horizon_stops(
    horizon: TradeHorizon, config: IndexBacktestConfig,
) -> tuple[float, float, float, int, int]:
    """Get stop/profit/hold params for a given horizon.

    Returns: (hard_stop, trailing_stop, take_profit, max_hold, dte)
    """
    if horizon == TradeHorizon.SCALP:
        return (
            config.scalp_hard_stop,
            config.scalp_trailing_stop,
            config.scalp_take_profit,
            config.scalp_max_hold,
            config.scalp_dte,
        )
    if horizon == TradeHorizon.SWING:
        return (
            config.swing_hard_stop,
            config.swing_trailing_stop,
            config.swing_take_profit,
            config.swing_max_hold,
            config.swing_dte,
        )
    return (
        config.medium_hard_stop,
        config.medium_trailing_stop,
        config.medium_take_profit,
        config.medium_max_hold,
        config.medium_dte,
    )


def _check_index_exits(
    portfolio: IndexPortfolio,
    today_bars: dict[str, dict[str, Any]],
    ticker_history: dict[str, list[dict[str, Any]]],
    current_date: str,
    config: IndexBacktestConfig,
) -> None:
    """Check exit conditions for all open index positions."""
    to_close: list[tuple[IndexPosition, str]] = []

    for pos in portfolio.positions:
        bar = today_bars.get(pos.ticker)
        if not bar:
            continue
        if pos.entry_premium <= 0:
            to_close.append((pos, "invalid"))
            continue

        pnl_pct = (pos.current_premium - pos.entry_premium) / pos.entry_premium

        h_stop, t_stop, tp, max_hold, _ = _get_horizon_stops(pos.horizon, config)

        # High conviction gets wider stops
        if pos.conviction >= 9.0:
            h_stop *= 1.15
            max_hold = int(max_hold * 1.2)

        # 1. Hard stop
        if pnl_pct <= h_stop:
            to_close.append((pos, "hard_stop"))
            continue

        # 2. Take profit
        if pnl_pct >= tp:
            to_close.append((pos, "take_profit"))
            continue

        # 3. Scalp quick-profit exit — take 50%+ gains immediately
        if pos.horizon == TradeHorizon.SCALP and pnl_pct >= 0.50:
            to_close.append((pos, "scalp_target_hit"))
            continue

        # 4. Trailing stop
        if pos.max_premium > pos.entry_premium * 1.15:
            trail_level = pos.max_premium * (1.0 - t_stop)
            if pos.current_premium <= trail_level:
                to_close.append((pos, "trailing_stop"))
                continue

        # 5. Time exit
        if pos.days_held >= max_hold:
            to_close.append((pos, "time_exit"))
            continue

        # 6. Regime reversal for swings/medium
        if pos.horizon != TradeHorizon.SCALP:
            history = ticker_history.get(pos.ticker, [])
            if len(history) >= WARMUP_BARS and pos.days_held >= 3:
                indicators = compute_indicators(history)
                regime = classify_index_regime(indicators)
                from flowedge.scanner.backtest.index_specialist.schemas import IndexRegime
                if pos.is_call and regime in (IndexRegime.BEAR, IndexRegime.STRONG_BEAR):
                    to_close.append((pos, "regime_reversal"))
                    continue
                if not pos.is_call and regime in (IndexRegime.BULL, IndexRegime.STRONG_BULL):
                    to_close.append((pos, "regime_reversal"))
                    continue

    for pos, reason in to_close:
        if pos in portfolio.positions:
            portfolio.close_position(pos, current_date, reason)


# ── Data Fetching ──────────────────────────────────────────────────


async def _fetch_bars(
    polygon: PolygonProvider,
    ticker: str,
    from_date: str,
    to_date: str,
) -> list[dict[str, Any]]:
    """Fetch daily bars for an index ETF."""
    data = await polygon._get(
        f"{polygon._base_url}/v2/aggs/ticker/{ticker}"
        f"/range/1/day/{from_date}/{to_date}",
        params={"apiKey": polygon._api_key, "limit": "5000", "sort": "asc"},
    )
    return [
        {
            "date": date.fromtimestamp(r["t"] / 1000).isoformat()
            if "t" in r else "",
            "open": float(r.get("o", 0)),
            "high": float(r.get("h", 0)),
            "low": float(r.get("l", 0)),
            "close": float(r.get("c", 0)),
            "volume": int(r.get("v", 0)),
        }
        for r in data.get("results", [])
    ]


# ── Metrics ─────────────────────────────────────────────────────────


def _max_drawdown(daily_values: list[tuple[str, float]]) -> float:
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


def _sharpe(daily_values: list[tuple[str, float]]) -> float:
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
    annual_return = mean_r * 252
    annual_std = std_r * sqrt(252)
    return round((annual_return - RISK_FREE_RATE) / annual_std, 3) if annual_std > 0 else 0.0


# ── Main Entry Point ──────────────────────────────────────────────


async def run_index_backtest(
    config: IndexBacktestConfig | None = None,
    settings: Settings | None = None,
) -> IndexBacktestResult:
    """Run the index specialist backtest.

    Processes SPY/QQQ/IWM with horizon-specific strategies and
    aggressive filtering for high win rates.
    """
    import asyncio

    config = config or IndexBacktestConfig()
    settings = settings or get_settings()
    polygon = PolygonProvider(settings)

    end_date = date.today()
    start_date = end_date - timedelta(days=config.lookback_days)

    # ── 1. Fetch data ──
    all_bars: dict[str, list[dict[str, Any]]] = {}
    for i, ticker in enumerate(config.tickers):
        if i > 0:
            await asyncio.sleep(13)
        try:
            bars = await _fetch_bars(
                polygon, ticker, start_date.isoformat(), end_date.isoformat(),
            )
            if len(bars) >= WARMUP_BARS + 20:
                all_bars[ticker] = bars
                logger.info("index_bars_loaded", ticker=ticker, count=len(bars))
            else:
                logger.warning("index_insufficient_bars", ticker=ticker, count=len(bars))
        except Exception as e:
            logger.warning("index_fetch_failed", ticker=ticker, error=str(e))

    await polygon.close()

    if not all_bars:
        return IndexBacktestResult(
            run_id=str(uuid.uuid4())[:12],
            tickers=config.tickers,
            lookback_days=config.lookback_days,
            starting_capital=config.starting_capital,
        )

    # ── 2. Build date index ──
    bars_by_date: dict[str, dict[str, dict[str, Any]]] = {}
    all_dates: set[str] = set()
    for ticker, bars in all_bars.items():
        for bar in bars:
            d = bar["date"]
            all_dates.add(d)
            bars_by_date.setdefault(d, {})[ticker] = bar

    sorted_dates = sorted(all_dates)

    # ── 3. Initialize portfolio ──
    portfolio = IndexPortfolio(
        cash=config.starting_capital,
        initial_capital=config.starting_capital,
        max_positions=config.max_positions,
        max_risk_pct=config.max_risk_per_trade,
    )
    ticker_history: dict[str, list[dict[str, Any]]] = {t: [] for t in all_bars}

    # ── 4. Walk each day ──
    for current_date in sorted_dates:
        today_bars = bars_by_date.get(current_date, {})

        for ticker in all_bars:
            if ticker in today_bars:
                ticker_history[ticker].append(today_bars[ticker])

        max_history = max(len(ticker_history[t]) for t in all_bars)
        if max_history < WARMUP_BARS:
            continue

        # a. Update positions
        portfolio.update_positions(today_bars)

        # b. Check exits
        _check_index_exits(
            portfolio, today_bars, ticker_history, current_date, config,
        )

        # c. Scan for entries
        if portfolio.can_open():
            all_signals: list[IndexSignal] = []
            for ticker in all_bars:
                history = ticker_history.get(ticker, [])
                if len(history) < WARMUP_BARS:
                    continue
                if any(p.ticker == ticker for p in portfolio.positions):
                    continue

                indicators = compute_indicators(history)
                regime = classify_index_regime(indicators)
                signals = scan_index_entries(ticker, history, indicators, regime)
                all_signals.extend(signals)

            all_signals.sort(key=lambda s: s.conviction, reverse=True)

            for signal in all_signals:
                if not portfolio.can_open():
                    break

                # Horizon-specific minimum conviction
                if signal.horizon == TradeHorizon.SCALP:
                    min_conv = config.scalp_min_conviction
                elif signal.horizon == TradeHorizon.SWING:
                    min_conv = config.swing_min_conviction
                else:
                    min_conv = config.medium_min_conviction

                if signal.conviction < min_conv:
                    continue

                sig_bar = today_bars.get(signal.ticker)
                if not sig_bar:
                    continue

                history = ticker_history.get(signal.ticker, [])
                indicators = compute_indicators(history)
                iv = estimate_iv_from_atr(indicators.atr14, float(sig_bar["close"]))
                closes = [float(b.get("close", 0)) for b in history]

                # ── Multi-factor conviction adjustment ──

                # PULSE momentum
                m_bias, m_score = classify_momentum_bias(indicators, closes)
                m_adj = compute_momentum_adjustment(m_bias, m_score, signal.direction)

                # GEX proxy
                g_regime, g_score = classify_gex_proxy(indicators, history)
                g_adj = compute_gex_adjustment(g_regime, g_score, signal.direction)

                # Kronos pattern
                k_adj = compute_kronos_adjustment(history, signal.direction)

                # Monte Carlo — KEY FILTER for index specialist
                _, _, _, h_max_hold, _ = _get_horizon_stops(signal.horizon, config)
                mc_adj, mc_analysis = compute_mc_conviction(
                    history, signal.direction,
                    otm_pct=signal.otm_pct,
                    hold_days=h_max_hold,
                    n_simulations=config.mc_simulations,
                )
                mc_prob = mc_analysis.get("prob_profit", 0.5)
                signal.mc_prob_profit = mc_prob

                # MC probability filter — block low-probability trades
                if signal.horizon == TradeHorizon.SCALP:
                    if mc_prob < config.mc_min_prob_profit_scalp:
                        continue
                else:
                    if mc_prob < config.mc_min_prob_profit_swing:
                        continue

                # Combine adjustments — less conservative cap for indices
                total_adj = m_adj + g_adj + k_adj + mc_adj
                if total_adj < -1.5:
                    total_adj = -1.5  # Tighter cap for indices

                adjusted = signal.conviction + total_adj
                signal.conviction = max(0.0, min(10.0, adjusted))

                # Re-check after adjustment
                if signal.conviction < min_conv:
                    continue

                # Get DTE for this horizon
                _, _, _, _, dte = _get_horizon_stops(signal.horizon, config)
                portfolio.open_position(signal, sig_bar, iv, dte=dte)

        # d. Snapshot
        portfolio.record_snapshot(current_date)

    # ── 5. Close remaining ──
    if sorted_dates:
        for pos in portfolio.positions[:]:
            portfolio.close_position(pos, sorted_dates[-1], "end_of_backtest")

    # ── 6. Compile results ──
    trades = portfolio.closed_trades
    total = len(trades)
    wins = sum(1 for t in trades if t.is_win)
    losses = total - wins

    win_pnls = [t.pnl_pct for t in trades if t.is_win]
    loss_pnls = [t.pnl_pct for t in trades if not t.is_win]

    gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))

    # By horizon
    scalp_t = [t for t in trades if t.horizon == TradeHorizon.SCALP]
    swing_t = [t for t in trades if t.horizon == TradeHorizon.SWING]
    medium_t = [t for t in trades if t.horizon == TradeHorizon.MEDIUM]

    scalp_wins = sum(1 for t in scalp_t if t.is_win)
    swing_wins = sum(1 for t in swing_t if t.is_win)
    medium_wins = sum(1 for t in medium_t if t.is_win)

    # By ticker
    by_ticker: dict[str, dict[str, float]] = {}
    for ticker in config.tickers:
        tt = [t for t in trades if t.ticker == ticker]
        if tt:
            tw = sum(1 for t in tt if t.is_win)
            by_ticker[ticker] = {
                "trades": float(len(tt)),
                "win_rate": round(tw / len(tt), 3),
                "avg_pnl_pct": round(sum(t.pnl_pct for t in tt) / len(tt), 2),
                "total_pnl_pct": round(sum(t.pnl_pct for t in tt), 2),
            }

    ending = portfolio.total_value
    portfolio_return = (ending - config.starting_capital) / config.starting_capital * 100

    result = IndexBacktestResult(
        run_id=str(uuid.uuid4())[:12],
        tickers=config.tickers,
        lookback_days=config.lookback_days,
        starting_capital=config.starting_capital,
        ending_value=round(ending, 2),
        portfolio_return_pct=round(portfolio_return, 2),
        max_drawdown_pct=_max_drawdown(portfolio.daily_values),
        sharpe_ratio=_sharpe(portfolio.daily_values),
        total_trades=total,
        wins=wins,
        losses=losses,
        win_rate=round(wins / total, 3) if total > 0 else 0.0,
        avg_win_pct=round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0.0,
        avg_loss_pct=round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0,
        profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
        expectancy_pct=round(sum(t.pnl_pct for t in trades) / total, 2) if total > 0 else 0.0,
        scalp_trades=len(scalp_t),
        scalp_win_rate=round(scalp_wins / len(scalp_t), 3) if scalp_t else 0.0,
        scalp_avg_pnl=round(sum(t.pnl_pct for t in scalp_t) / len(scalp_t), 2) if scalp_t else 0.0,
        swing_trades=len(swing_t),
        swing_win_rate=round(swing_wins / len(swing_t), 3) if swing_t else 0.0,
        swing_avg_pnl=round(sum(t.pnl_pct for t in swing_t) / len(swing_t), 2) if swing_t else 0.0,
        medium_trades=len(medium_t),
        medium_win_rate=round(medium_wins / len(medium_t), 3) if medium_t else 0.0,
        medium_avg_pnl=(
            round(sum(t.pnl_pct for t in medium_t) / len(medium_t), 2)
            if medium_t else 0.0
        ),
        by_ticker=by_ticker,
        trades=trades,
    )

    logger.info(
        "index_backtest_complete",
        tickers=list(all_bars.keys()),
        total_trades=total,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        portfolio_return=f"{portfolio_return:.1f}%",
        sharpe=result.sharpe_ratio,
        scalp_wr=result.scalp_win_rate,
        swing_wr=result.swing_win_rate,
        medium_wr=result.medium_win_rate,
    )

    return result


# ── Self-Learning Feedback Loop ───────────────────────────────────


def run_index_learning_cycle(
    result: IndexBacktestResult,
    config: IndexBacktestConfig,
) -> IndexBacktestConfig:
    """Analyze backtest results and adjust config for next iteration.

    Adaptive feedback:
    - If scalp WR < 70%: raise scalp_min_conviction, tighten mc_min_prob
    - If swing WR < 80%: raise swing_min_conviction
    - If medium WR < 90%: raise medium_min_conviction
    - Adjust stops based on exit reason distribution
    """
    updated = config.model_copy(deep=True)
    notes: list[str] = []

    # Scalp tuning
    if result.scalp_trades > 5:
        if result.scalp_win_rate < 0.70:
            gap = 0.70 - result.scalp_win_rate
            bump = min(0.5, gap * 2)
            updated.scalp_min_conviction = min(
                9.5, config.scalp_min_conviction + bump,
            )
            updated.mc_min_prob_profit_scalp = min(
                0.70, config.mc_min_prob_profit_scalp + 0.03,
            )
            notes.append(
                f"Scalp WR {result.scalp_win_rate:.1%} < 70% → "
                f"raised min_conviction to {updated.scalp_min_conviction:.1f}, "
                f"MC prob to {updated.mc_min_prob_profit_scalp:.2f}"
            )
        # Check if scalp gains are hitting 50%+ target
        scalp_wins = [t for t in result.trades if t.horizon == TradeHorizon.SCALP and t.is_win]
        if scalp_wins:
            avg_gain = sum(t.pnl_pct for t in scalp_wins) / len(scalp_wins)
            if avg_gain < 50.0:
                # Widen take profit to let winners run
                updated.scalp_take_profit = min(2.0, config.scalp_take_profit + 0.25)
                notes.append(
                    f"Scalp avg win {avg_gain:.0f}% < 50% "
                    f"→ widened TP to {updated.scalp_take_profit:.2f}"
                )

    # Swing tuning
    if result.swing_trades > 5 and result.swing_win_rate < 0.80:
            gap = 0.80 - result.swing_win_rate
            bump = min(0.5, gap * 2)
            updated.swing_min_conviction = min(
                9.5, config.swing_min_conviction + bump,
            )
            notes.append(
                f"Swing WR {result.swing_win_rate:.1%} < 80% → "
                f"raised min_conviction to {updated.swing_min_conviction:.1f}"
            )

    # Medium tuning
    if result.medium_trades > 5 and result.medium_win_rate < 0.90:
            gap = 0.90 - result.medium_win_rate
            bump = min(0.5, gap * 2)
            updated.medium_min_conviction = min(
                9.5, config.medium_min_conviction + bump,
            )
            notes.append(
                f"Medium WR {result.medium_win_rate:.1%} < 90% → "
                f"raised min_conviction to {updated.medium_min_conviction:.1f}"
            )

    # Stop-loss tuning from exit reasons
    hard_stops = [t for t in result.trades if t.exit_reason == "hard_stop"]
    if len(hard_stops) > 3:
        # Check if hard-stopped trades were correct direction
        premature = [
            t for t in hard_stops
            if (t.direction == "bullish" and t.exit_underlying > t.entry_underlying)
            or (t.direction == "bearish" and t.exit_underlying < t.entry_underlying)
        ]
        if len(premature) / len(hard_stops) > 0.30:
            updated.scalp_hard_stop *= 1.10  # 10% wider
            updated.swing_hard_stop *= 1.10
            notes.append(
                f"{len(premature)}/{len(hard_stops)} hard stops were premature → widened stops 10%"
            )

    if notes:
        for note in notes:
            logger.info("index_learning", adjustment=note)

    return updated


def save_index_results(
    result: IndexBacktestResult,
    config: IndexBacktestConfig,
    output_dir: str = "data/backtest",
) -> str:
    """Save index backtest results and config to JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    filename = f"index_specialist_{result.run_id}.json"
    filepath = path / filename

    output = {
        "result": result.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
    }
    filepath.write_text(json.dumps(output, indent=2, default=str))
    logger.info("index_results_saved", path=str(filepath))
    return str(filepath)
