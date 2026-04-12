"""PHANTOM v2 — Multi-strategy portfolio options backtester.

Complete redesign from v1. Key improvements:
1. Black-Scholes option pricing with ATR-derived IV
2. Four strategies: trend pullback, breakout, mean reversion, vol squeeze
3. Regime-aware direction (SMA crossover + ADX)
4. Portfolio-level risk management (max positions, max heat, cash reserve)
5. Dynamic position sizing by conviction score
6. Trailing stops for winners + regime reversal exits
7. Cross-ticker daily portfolio processing (not per-ticker sequential)
8. Proper Greeks-based option repricing through hold period
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from math import sqrt
from typing import Any

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.backtest.adaptive_scorer import (
    compute_adaptive_conviction,
    extract_features,
    load_scorer_weights,
)
from flowedge.scanner.backtest.pricing import bs_price, estimate_iv_from_atr
from flowedge.scanner.backtest.schemas import (
    BacktestMonthly,
    BacktestResult,
    BacktestTrade,
    TradeOutcome,
)
from flowedge.scanner.backtest.strategies import (
    EntrySignal,
    MarketRegime,
    compute_indicators,
    detect_regime,
    scan_for_entries,
)
from flowedge.scanner.providers.polygon import PolygonProvider
from flowedge.scanner.providers.registry import ProviderRegistry

logger = structlog.get_logger()

WARMUP_BARS = 55  # Need 50 for SMA50 + buffer
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05


# ── Portfolio State ─────────────────────────────────────────────────


@dataclass
class OpenPosition:
    """An open option position in the portfolio."""

    ticker: str
    direction: str
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
    days_held: int = 0
    max_premium: float = 0.0
    current_premium: float = 0.0
    current_underlying: float = 0.0

    def __post_init__(self) -> None:
        self.max_premium = self.entry_premium
        self.current_premium = self.entry_premium
        self.current_underlying = self.entry_underlying


@dataclass
class Portfolio:
    """Manages portfolio state, position sizing, and risk limits."""

    cash: float
    initial_capital: float
    max_positions: int = 5
    max_risk_pct: float = 0.08
    max_heat: float = 0.45
    positions: list[OpenPosition] = field(default_factory=list)
    closed_trades: list[BacktestTrade] = field(default_factory=list)
    daily_values: list[tuple[str, float]] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        pos_value = sum(p.current_premium * p.contracts * 100 for p in self.positions)
        return self.cash + max(0.0, pos_value)

    @property
    def deployed_pct(self) -> float:
        tv = self.total_value
        if tv <= 0:
            return 1.0
        pos_value = sum(p.current_premium * p.contracts * 100 for p in self.positions)
        return pos_value / tv

    def can_open(self) -> bool:
        return len(self.positions) < self.max_positions and self.deployed_pct < self.max_heat

    def position_size(self, conviction: float) -> float:
        """Calculate position size based on conviction (0-10)."""
        base = self.total_value * self.max_risk_pct
        scale = 0.3 + 0.7 * (conviction / 10.0)
        return base * scale

    def open_position(
        self,
        signal: EntrySignal,
        bar: dict[str, Any],
        iv: float,
        dte: int = 15,
    ) -> OpenPosition | None:
        """Open a new option position."""
        if not self.can_open():
            return None

        is_call = signal.direction == "bullish"
        if is_call:
            strike = bar["close"] * (1.0 + signal.otm_pct)
        else:
            strike = bar["close"] * (1.0 - signal.otm_pct)

        t_years = max(dte, 1) / TRADING_DAYS_PER_YEAR
        premium = bs_price(bar["close"], strike, t_years, RISK_FREE_RATE, iv, is_call)

        if premium < 0.05:
            return None

        budget = self.position_size(signal.conviction)
        contracts = max(1, int(budget / (premium * 100)))
        cost = contracts * premium * 100

        # Keep minimum cash reserve
        if cost > self.cash * 0.90:
            contracts = max(1, int(self.cash * 0.85 / (premium * 100)))
            cost = contracts * premium * 100

        if cost > self.cash or cost < 10:
            return None

        self.cash -= cost
        pos = OpenPosition(
            ticker=signal.ticker,
            direction=signal.direction,
            strategy=signal.strategy,
            is_call=is_call,
            entry_date=bar["date"],
            entry_underlying=bar["close"],
            strike=round(strike, 2),
            entry_premium=premium,
            contracts=contracts,
            cost_basis=cost,
            iv=iv,
            conviction=signal.conviction,
            regime=signal.regime,
            dte_at_entry=dte,
        )
        self.positions.append(pos)
        return pos

    def close_position(
        self,
        pos: OpenPosition,
        current_date: str,
        reason: str,
    ) -> BacktestTrade:
        """Close a position and record the trade."""
        exit_premium = max(0.0, pos.current_premium)
        exit_value = exit_premium * pos.contracts * 100
        pnl = exit_value - pos.cost_basis
        pnl_pct = (pnl / pos.cost_basis * 100) if pos.cost_basis > 0 else 0.0

        self.cash += exit_value
        if pos in self.positions:
            self.positions.remove(pos)

        if pnl_pct >= 10:
            outcome = TradeOutcome.WIN
        elif pnl_pct <= -90:
            outcome = TradeOutcome.EXPIRED
        elif pnl_pct < -10:
            outcome = TradeOutcome.LOSS
        else:
            outcome = TradeOutcome.BREAKEVEN

        underlying_move = 0.0
        if pos.entry_underlying > 0:
            underlying_move = (
                (pos.current_underlying - pos.entry_underlying)
                / pos.entry_underlying
                * 100
            )

        trade = BacktestTrade(
            ticker=pos.ticker,
            entry_date=date.fromisoformat(pos.entry_date),
            exit_date=date.fromisoformat(current_date),
            option_type="call" if pos.is_call else "put",
            strike=pos.strike,
            entry_price=round(pos.entry_premium, 4),
            exit_price=round(exit_premium, 4),
            underlying_entry=pos.entry_underlying,
            underlying_exit=pos.current_underlying,
            underlying_move_pct=round(underlying_move, 2),
            pnl_per_contract=round(pnl / max(pos.contracts, 1), 2),
            pnl_pct=round(pnl_pct, 2),
            outcome=outcome,
            signal_score=round(pos.conviction, 1),
            signal_type=f"{pos.strategy}|{pos.regime}",
            hold_days=pos.days_held,
            strategy=pos.strategy,
            regime=pos.regime,
            conviction=round(pos.conviction, 2),
            exit_reason=reason,
            contracts=pos.contracts,
            cost_basis=round(pos.cost_basis, 2),
            exit_value=round(exit_value, 2),
        )
        self.closed_trades.append(trade)
        return trade

    def update_positions(self, bars_by_ticker: dict[str, dict[str, Any]]) -> None:
        """Reprice all open positions using today's underlying prices."""
        for pos in self.positions:
            bar = bars_by_ticker.get(pos.ticker)
            if not bar:
                continue

            pos.days_held += 1
            pos.current_underlying = float(bar["close"])

            # Reprice option with BS
            remaining_dte = max(1, pos.dte_at_entry - pos.days_held)
            t_years = remaining_dte / TRADING_DAYS_PER_YEAR
            pos.current_premium = bs_price(
                pos.current_underlying,
                pos.strike,
                t_years,
                RISK_FREE_RATE,
                pos.iv,
                pos.is_call,
            )

            if pos.current_premium > pos.max_premium:
                pos.max_premium = pos.current_premium

    def record_snapshot(self, date_str: str) -> None:
        self.daily_values.append((date_str, self.total_value))


# ── Data Fetching ────────────────────────────────────────────────────


async def _fetch_price_history(
    polygon: PolygonProvider,
    ticker: str,
    from_date: str,
    to_date: str,
) -> list[dict[str, Any]]:
    """Fetch daily bars for a ticker over a date range."""
    data = await polygon._get(
        f"{polygon._base_url}/v2/aggs/ticker/{ticker}"
        f"/range/1/day/{from_date}/{to_date}",
        params={"apiKey": polygon._api_key, "limit": "5000", "sort": "asc"},
    )
    return [
        {
            "date": date.fromtimestamp(r["t"] / 1000).isoformat()
            if "t" in r
            else "",
            "open": float(r.get("o", 0)),
            "high": float(r.get("h", 0)),
            "low": float(r.get("l", 0)),
            "close": float(r.get("c", 0)),
            "volume": int(r.get("v", 0)),
        }
        for r in data.get("results", [])
    ]


# ── Exit Logic ───────────────────────────────────────────────────────


# v7: Strategy-specific stop profiles — data-driven from v5 analysis
# Key insight: winners hold longer (7.1d avg), hard stops kill 40% of trades
# at 0% WR. Winners exit via take_profit (100% WR) or time_exit (67% WR).
STRATEGY_STOPS: dict[str, dict[str, float]] = {
    "trend_pullback": {
        "hard_stop": -0.50,      # v7: wider — hard stops are 0% WR
        "trailing_stop": 0.30,   # Tighter trail to lock trend profits
        "take_profit": 3.50,     # v7: wider — let big winners run
        "max_hold": 12,          # v7: extended — winners avg 8.4d hold
    },
    "ibs_reversion": {
        "hard_stop": -0.35,      # v7: wider — was cutting winners
        "trailing_stop": 0.25,   # Tight trail — fast moves
        "take_profit": 2.50,     # v7: wider — avg IBS win is +70%
        "max_hold": 7,           # v7: extended from 5 — give more room
    },
    "mean_reversion": {
        "hard_stop": -0.35,
        "trailing_stop": 0.25,
        "take_profit": 2.00,
        "max_hold": 5,
    },
    "vol_squeeze": {
        "hard_stop": -0.50,      # v7: wider for squeeze patience
        "trailing_stop": 0.35,
        "take_profit": 4.00,     # v7: let explosions run further
        "max_hold": 14,          # v7: extended for squeeze plays
    },
}

# v8: Ticker/strategy performance now handled by adaptive_scorer.py
# Historical WR lookups live there (TICKER_HISTORICAL_WR, STRATEGY_HISTORICAL_WR)

# v7: Drawdown circuit breaker
MAX_CONSECUTIVE_LOSSES_BEFORE_PAUSE = 4
PAUSE_TRADES_AFTER_STREAK = 3  # Skip this many potential trades


def _get_strategy_stops(
    strategy: str,
    hard_stop_pct: float,
    trailing_stop_pct: float,
    take_profit_pct: float,
    max_hold_days: int,
) -> tuple[float, float, float, int]:
    """Get strategy-specific or default stop parameters."""
    profile = STRATEGY_STOPS.get(strategy)
    if profile:
        return (
            profile["hard_stop"],
            profile["trailing_stop"],
            profile["take_profit"],
            int(profile["max_hold"]),
        )
    return hard_stop_pct, trailing_stop_pct, take_profit_pct, max_hold_days


def _check_exits(
    portfolio: Portfolio,
    today_bars: dict[str, dict[str, Any]],
    ticker_history: dict[str, list[dict[str, Any]]],
    current_date: str,
    *,
    hard_stop_pct: float,
    trailing_stop_pct: float,
    take_profit_pct: float,
    max_hold_days: int,
) -> None:
    """Check all open positions for exit conditions.

    v4: Uses strategy-specific stop profiles. Trend-following strategies
    get wider stops; reversion strategies get tighter stops.
    """
    to_close: list[tuple[OpenPosition, str]] = []

    for pos in portfolio.positions:
        bar = today_bars.get(pos.ticker)
        if not bar:
            continue

        if pos.entry_premium <= 0:
            to_close.append((pos, "invalid"))
            continue

        pnl_pct = (pos.current_premium - pos.entry_premium) / pos.entry_premium

        # Get strategy-specific stops
        s_hard, s_trail, s_tp, s_hold = _get_strategy_stops(
            pos.strategy, hard_stop_pct, trailing_stop_pct,
            take_profit_pct, max_hold_days,
        )

        # High-conviction trades get 15% wider stops (learned from premature stop analysis)
        if pos.conviction >= 8.0:
            s_hard *= 1.15  # e.g., -0.40 → -0.46
            s_hold = int(s_hold * 1.2)  # e.g., 10 → 12

        # 1. Hard stop — limit max loss
        if pnl_pct <= s_hard:
            to_close.append((pos, "hard_stop"))
            continue

        # 2. Take profit — lock in big wins
        if pnl_pct >= s_tp:
            to_close.append((pos, "take_profit"))
            continue

        # 3. Trailing stop — protect accumulated gains
        if pos.max_premium > pos.entry_premium * 1.20:
            trail_level = pos.max_premium * (1.0 - s_trail)
            if pos.current_premium <= trail_level:
                to_close.append((pos, "trailing_stop"))
                continue

        # 4. Time exit — avoid theta acceleration
        if pos.days_held >= s_hold:
            to_close.append((pos, "time_exit"))
            continue

        # 5. Regime reversal — exit if regime flips against position
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


# ── Aggregation Helpers ──────────────────────────────────────────────


def _compute_score_buckets(
    trades: list[BacktestTrade],
) -> dict[str, dict[str, float]]:
    """Group trade performance by signal score ranges."""
    buckets: dict[str, list[BacktestTrade]] = {
        "0-3": [],
        "3-5": [],
        "5-7": [],
        "7-10": [],
    }
    for t in trades:
        if t.signal_score < 3:
            buckets["0-3"].append(t)
        elif t.signal_score < 5:
            buckets["3-5"].append(t)
        elif t.signal_score < 7:
            buckets["5-7"].append(t)
        else:
            buckets["7-10"].append(t)

    result: dict[str, dict[str, float]] = {}
    for name, bt in buckets.items():
        if not bt:
            continue
        wins = sum(1 for t in bt if t.outcome == TradeOutcome.WIN)
        avg_pnl = sum(t.pnl_pct for t in bt) / len(bt)
        result[name] = {
            "count": float(len(bt)),
            "win_rate": round(wins / len(bt), 3),
            "avg_pnl_pct": round(avg_pnl, 2),
        }
    return result


def _compute_group_stats(
    trades: list[BacktestTrade],
    key_fn: Any,
) -> dict[str, dict[str, float]]:
    """Generic grouping stats by a key function."""
    groups: dict[str, list[BacktestTrade]] = {}
    for t in trades:
        k = key_fn(t)
        if k:
            groups.setdefault(k, []).append(t)

    result: dict[str, dict[str, float]] = {}
    for name, gt in groups.items():
        wins = sum(1 for t in gt if t.outcome == TradeOutcome.WIN)
        result[name] = {
            "trades": float(len(gt)),
            "win_rate": round(wins / len(gt), 3) if gt else 0.0,
            "avg_pnl_pct": round(sum(t.pnl_pct for t in gt) / len(gt), 2),
            "total_pnl_pct": round(sum(t.pnl_pct for t in gt), 2),
        }
    return result


def _compute_max_drawdown(daily_values: list[tuple[str, float]]) -> float:
    """Compute max drawdown percentage from daily portfolio values."""
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
    """Annualized Sharpe ratio from daily portfolio values."""
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

    # Annualize: Sharpe = (mean * 252 - rf) / (std * sqrt(252))
    annual_return = mean_r * 252
    annual_std = std_r * sqrt(252)
    return round((annual_return - RISK_FREE_RATE) / annual_std, 3) if annual_std > 0 else 0.0


# ── Main Entry Point ────────────────────────────────────────────────


async def run_backtest(
    tickers: list[str],
    lookback_days: int = 730,
    starting_capital: float = 10_000.0,
    max_positions: int = 5,
    max_risk_per_trade: float = 0.08,
    max_hold_days: int = 12,
    trailing_stop_pct: float = 0.35,
    hard_stop_pct: float = -0.50,
    take_profit_pct: float = 3.50,
    dte: int = 15,
    min_conviction: float = 7.0,
    settings: Settings | None = None,
    **_kwargs: Any,
) -> BacktestResult:
    """Run multi-strategy portfolio backtest over historical data.

    Args:
        tickers: Tickers to trade.
        lookback_days: How far back to fetch data.
        starting_capital: Initial portfolio value.
        max_positions: Max concurrent open positions.
        max_risk_per_trade: Max fraction of portfolio per trade.
        max_hold_days: Max days to hold a position.
        trailing_stop_pct: Trail stop level (fraction from max premium).
        hard_stop_pct: Hard stop level (fraction loss from entry).
        take_profit_pct: Take profit level (multiple of entry premium).
        dte: Days to expiration for options.
        min_conviction: Minimum conviction score to enter.
        settings: App settings.
    """
    settings = settings or get_settings()
    polygon = PolygonProvider(settings)
    registry = ProviderRegistry(settings)

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    # ── 1. Fetch all price data ──────────────────────────────────────
    import asyncio

    all_bars: dict[str, list[dict[str, Any]]] = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            await asyncio.sleep(1)  # Paid tier — no rate limit  # Polygon free-tier rate limit
        try:
            bars = await _fetch_price_history(
                polygon, ticker, start_date.isoformat(), end_date.isoformat()
            )
            if len(bars) >= WARMUP_BARS + 20:
                all_bars[ticker] = bars
                logger.info("bars_loaded", ticker=ticker, count=len(bars))
            else:
                logger.warning("insufficient_bars", ticker=ticker, count=len(bars))
        except Exception as e:
            logger.warning("fetch_failed", ticker=ticker, error=str(e))

    await polygon.close()
    await registry.close_all()

    if not all_bars:
        return BacktestResult(
            run_id=str(uuid.uuid4())[:12],
            tickers=tickers,
            lookback_days=lookback_days,
            starting_capital=starting_capital,
        )

    # v8: Load adaptive scorer weights
    scorer_weights = load_scorer_weights()

    # ── 2. Build date-indexed structure ──────────────────────────────
    bars_by_date: dict[str, dict[str, dict[str, Any]]] = {}
    all_dates: set[str] = set()
    for ticker, bars in all_bars.items():
        for bar in bars:
            d = bar["date"]
            all_dates.add(d)
            bars_by_date.setdefault(d, {})[ticker] = bar

    sorted_dates = sorted(all_dates)

    # ── 3. Initialize portfolio ──────────────────────────────────────
    portfolio = Portfolio(
        cash=starting_capital,
        initial_capital=starting_capital,
        max_positions=max_positions,
        max_risk_pct=max_risk_per_trade,
    )

    ticker_history: dict[str, list[dict[str, Any]]] = {t: [] for t in all_bars}

    # v7: Drawdown circuit breaker state
    consecutive_losses = 0
    pause_remaining = 0

    # ── 4. Walk each day ─────────────────────────────────────────────
    for current_date in sorted_dates:
        today_bars = bars_by_date.get(current_date, {})

        # Append today's data to ticker histories
        for ticker in all_bars:
            if ticker in today_bars:
                ticker_history[ticker].append(today_bars[ticker])

        # Skip warmup period
        max_history = max(len(ticker_history[t]) for t in all_bars)
        if max_history < WARMUP_BARS:
            continue

        # a. Update open position prices
        portfolio.update_positions(today_bars)

        # b. Check exit conditions
        pre_exit_count = len(portfolio.closed_trades)
        _check_exits(
            portfolio,
            today_bars,
            ticker_history,
            current_date,
            hard_stop_pct=hard_stop_pct,
            trailing_stop_pct=trailing_stop_pct,
            take_profit_pct=take_profit_pct,
            max_hold_days=max_hold_days,
        )

        # v7: Track consecutive losses for circuit breaker
        new_closes = portfolio.closed_trades[pre_exit_count:]
        for closed in new_closes:
            if closed.outcome == TradeOutcome.WIN:
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES_BEFORE_PAUSE:
                    pause_remaining = PAUSE_TRADES_AFTER_STREAK

        # c. Scan for new entries
        if portfolio.can_open():
            all_signals: list[EntrySignal] = []
            for ticker in all_bars:
                history = ticker_history.get(ticker, [])
                if len(history) < WARMUP_BARS:
                    continue

                # Skip if already have position in this ticker
                if any(p.ticker == ticker for p in portfolio.positions):
                    continue

                indicators = compute_indicators(history)
                regime = detect_regime(indicators)
                signals = scan_for_entries(ticker, history, indicators, regime)
                all_signals.extend(signals)

            # Sort by conviction, take best ones
            all_signals.sort(key=lambda s: s.conviction, reverse=True)

            for signal in all_signals:
                if not portfolio.can_open():
                    break
                if signal.conviction < min_conviction:
                    break

                # v7: Circuit breaker — skip entries during loss streaks
                if pause_remaining > 0:
                    pause_remaining -= 1
                    continue

                sig_bar = today_bars.get(signal.ticker)
                if not sig_bar:
                    continue

                # Estimate IV from ATR
                history = ticker_history.get(signal.ticker, [])
                indicators = compute_indicators(history)
                iv = estimate_iv_from_atr(indicators.atr14, float(sig_bar["close"]))

                # ── v8: Adaptive scorer — replaces v7 multi-factor stacking ──
                # v3-v7 conviction had r=0.004 PnL correlation (noise).
                # Adaptive scorer uses data-driven feature weights trained
                # on historical winner/loser patterns.
                # Recover regime enum from the signal's stored value
                sig_regime = MarketRegime(signal.regime)
                features = extract_features(
                    signal.ticker,
                    signal.direction,
                    signal.strategy,
                    indicators,
                    sig_regime,
                    history,
                )
                adaptive_conv, _breakdown = compute_adaptive_conviction(
                    features, scorer_weights,
                )

                # v8.1: Use adaptive scorer as PRIMARY with veto power
                # The adaptive scorer has r=0.22 PnL correlation (strong).
                # Original signal conviction is pattern quality only.
                # Adaptive must independently pass min_conviction to enter.
                if adaptive_conv < min_conviction:
                    continue  # Adaptive scorer veto — ticker/regime/features say no

                # Blend: 20% original + 80% adaptive for final score
                blended = 0.20 * signal.conviction + 0.80 * adaptive_conv
                signal.conviction = round(max(0.0, min(10.0, blended)), 2)

                portfolio.open_position(signal, sig_bar, iv, dte=dte)

        # d. Record daily snapshot
        portfolio.record_snapshot(current_date)

    # ── 5. Close remaining positions at end ──────────────────────────
    for pos in portfolio.positions[:]:
        portfolio.close_position(pos, sorted_dates[-1], "end_of_backtest")

    # ── 6. Compile results ───────────────────────────────────────────
    all_trades = portfolio.closed_trades
    total = len(all_trades)

    wins = sum(1 for t in all_trades if t.outcome == TradeOutcome.WIN)
    losses = sum(
        1 for t in all_trades if t.outcome in (TradeOutcome.LOSS, TradeOutcome.EXPIRED)
    )
    expired = sum(1 for t in all_trades if t.outcome == TradeOutcome.EXPIRED)

    win_pnls = [t.pnl_pct for t in all_trades if t.outcome == TradeOutcome.WIN]
    loss_pnls = [t.pnl_pct for t in all_trades if t.outcome != TradeOutcome.WIN]

    gross_profit = sum(t.pnl_pct for t in all_trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct for t in all_trades if t.pnl_pct < 0))

    # Per-ticker stats
    ticker_stats = _compute_group_stats(all_trades, lambda t: t.ticker)

    # Per-strategy stats
    strategy_stats = _compute_group_stats(all_trades, lambda t: t.strategy)

    # Per-regime stats
    regime_stats = _compute_group_stats(all_trades, lambda t: t.regime)

    # Monthly breakdown
    monthly_data: dict[str, list[BacktestTrade]] = {}
    for t in all_trades:
        key = t.entry_date.strftime("%Y-%m")
        monthly_data.setdefault(key, []).append(t)

    monthly_list: list[BacktestMonthly] = []
    for month_key in sorted(monthly_data):
        mt = monthly_data[month_key]
        mw = sum(1 for t in mt if t.outcome == TradeOutcome.WIN)
        ml = len(mt) - mw
        monthly_list.append(
            BacktestMonthly(
                month=month_key,
                trades=len(mt),
                wins=mw,
                losses=ml,
                win_rate=round(mw / len(mt), 3) if mt else 0.0,
                avg_pnl_pct=round(sum(t.pnl_pct for t in mt) / len(mt), 2) if mt else 0.0,
                total_pnl_pct=round(sum(t.pnl_pct for t in mt), 2),
            )
        )

    # Consecutive streaks
    max_w = 0
    max_l = 0
    cur = 0
    last_out: TradeOutcome | None = None
    for t in sorted(all_trades, key=lambda x: x.entry_date):
        if t.outcome == last_out:
            cur += 1
        else:
            cur = 1
            last_out = t.outcome
        if last_out == TradeOutcome.WIN:
            max_w = max(max_w, cur)
        elif last_out in (TradeOutcome.LOSS, TradeOutcome.EXPIRED):
            max_l = max(max_l, cur)

    expectancy = round(sum(t.pnl_pct for t in all_trades) / total, 2) if total > 0 else 0.0

    # Portfolio-level metrics
    ending = portfolio.total_value
    portfolio_return = (ending - starting_capital) / starting_capital * 100

    result = BacktestResult(
        run_id=str(uuid.uuid4())[:12],
        tickers=tickers,
        lookback_days=lookback_days,
        total_trades=total,
        wins=wins,
        losses=losses,
        expired_worthless=expired,
        win_rate=round(wins / total, 3) if total > 0 else 0.0,
        avg_win_pct=round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0.0,
        avg_loss_pct=round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0,
        best_trade_pct=round(max((t.pnl_pct for t in all_trades), default=0), 2),
        worst_trade_pct=round(min((t.pnl_pct for t in all_trades), default=0), 2),
        total_pnl_pct=round(sum(t.pnl_pct for t in all_trades), 2),
        profit_factor=round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
        avg_hold_days=(
            round(sum(t.hold_days for t in all_trades) / total, 1) if total > 0 else 0.0
        ),
        max_consecutive_wins=max_w,
        max_consecutive_losses=max_l,
        expectancy_pct=expectancy,
        trades=all_trades,
        by_score_bucket=_compute_score_buckets(all_trades),
        by_ticker=ticker_stats,
        monthly=monthly_list,
        # v2 fields
        starting_capital=starting_capital,
        ending_value=round(ending, 2),
        portfolio_return_pct=round(portfolio_return, 2),
        max_drawdown_pct=_compute_max_drawdown(portfolio.daily_values),
        sharpe_ratio=_compute_sharpe(portfolio.daily_values),
        by_strategy=strategy_stats,
        by_regime=regime_stats,
    )

    logger.info(
        "backtest_v2_complete",
        tickers=tickers,
        total_trades=total,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        portfolio_return=f"{portfolio_return:.1f}%",
        sharpe=result.sharpe_ratio,
    )
    return result
