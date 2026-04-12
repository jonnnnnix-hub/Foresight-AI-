"""PHANTOM v9 — SPY/QQQ near-ATM options engine.

Fundamental redesign driven by slippage analysis:
- v7 generated +129% gross but -1% net after spreads
- Single stocks have 3-5x wider spreads than index ETFs
- OTM options (2.5-3.5%) have 3-4x wider spreads than near-ATM (0.5-1%)

v9 changes:
1. SPY and QQQ ONLY (tightest option spreads in the market)
2. Near-ATM strikes: 0.3-0.8% OTM (was 1.5-3.5%)
3. Slippage deducted from every fill (entry at ask, exit at bid)
4. Higher minimum premium ($0.50) to avoid penny-option spread death
5. Longer DTE (21-30 days) — higher premium = lower spread as % of cost
6. Adaptive scorer for conviction (r=0.196 correlation proven)
7. Aggressive min_conviction (8.0+) — only best setups
8. Max 2-3 positions — concentrated portfolio on highest quality
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
    ScorerWeights,
    compute_adaptive_conviction,
    extract_features,
)
from flowedge.scanner.backtest.pricing import bs_price, estimate_iv_from_atr
from flowedge.scanner.backtest.schemas import (
    BacktestResult,
    BacktestTrade,
    TradeOutcome,
)
from flowedge.scanner.backtest.slippage import (
    SlippageModel,
    apply_entry_slippage,
    apply_exit_slippage,
)
from flowedge.scanner.backtest.strategies import (
    EntrySignal,
    MarketRegime,
    compute_indicators,
    detect_regime,
    scan_for_entries,
)
from flowedge.scanner.providers.polygon import PolygonProvider

logger = structlog.get_logger()

WARMUP_BARS = 55
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05

# v9.2: SPY only — data-driven from v9.0 (QQQ=15.4% WR) and v9.1 (QQQ=-82%)
V9_TICKERS = ["SPY"]

# v9.2: Both strong regimes allowed — SPY profits in both when alone
# strong_downtrend was bad with QQQ included but SPY-alone is viable
V9_ALLOWED_REGIMES = {"strong_uptrend", "strong_downtrend"}

# v9 OTM override: near-ATM for tight spreads
V9_OTM_PCT = 0.005  # 0.5% OTM (was 1.5-3.5%)

# v9 slippage model calibrated for index ETF options
V9_SLIPPAGE = SlippageModel(
    base_spread_pct=0.012,  # 1.2% of premium (SPY/QQQ are tight)
    otm_spread_multiplier=0.5,  # Near-ATM, so minimal OTM penalty
    cheap_option_floor=0.02,
    market_impact_pct=0.002,  # Minimal for index ETFs
    enabled=True,
)

# v9.1: Near-ATM stop profiles — calibrated for higher delta (~0.45-0.50)
# Near-ATM moves ~2x more per underlying % than OTM → smaller % stops needed
# but also smaller % gains → lower TP targets.
# Key insight: trend_pullback was 37.5% WR, IBS only 16.7% near-ATM.
# Near-ATM favors trend continuation (responsive delta) over mean reversion.
V9_STOPS: dict[str, dict[str, float]] = {
    "trend_pullback": {
        "hard_stop": -0.25,      # Tighter — higher delta amplifies losses
        "trailing_stop": 0.18,   # Tighter trail — lock gains quickly
        "take_profit": 0.60,     # 60% TP — realistic for near-ATM on index
        "max_hold": 12,
    },
    "ibs_reversion": {
        "hard_stop": -0.20,      # Quick cut — IBS underperforms near-ATM
        "trailing_stop": 0.15,
        "take_profit": 0.40,     # 40% TP — take what you can
        "max_hold": 5,
    },
    "mean_reversion": {
        "hard_stop": -0.20,
        "trailing_stop": 0.15,
        "take_profit": 0.40,
        "max_hold": 5,
    },
    "vol_squeeze": {
        "hard_stop": -0.30,
        "trailing_stop": 0.20,
        "take_profit": 0.80,
        "max_hold": 14,
    },
}

# Circuit breaker
MAX_CONSEC_LOSSES = 3  # Tighter than v7 — fewer trades, protect capital
PAUSE_AFTER_STREAK = 5  # Skip more signals after a losing streak


@dataclass
class V9Position:
    """An open near-ATM option position with slippage tracking."""

    ticker: str
    direction: str
    strategy: str
    is_call: bool
    entry_date: str
    entry_underlying: float
    strike: float
    entry_premium_theo: float  # Theoretical BS price
    entry_premium_fill: float  # Actual fill (after slippage)
    contracts: int
    cost_basis: float  # Based on fill price
    iv: float
    conviction: float
    regime: str
    dte_at_entry: int
    otm_pct: float
    days_held: int = 0
    max_premium: float = 0.0
    current_premium: float = 0.0
    current_underlying: float = 0.0

    def __post_init__(self) -> None:
        self.max_premium = self.entry_premium_fill
        self.current_premium = self.entry_premium_fill
        self.current_underlying = self.entry_underlying


@dataclass
class V9Portfolio:
    """Portfolio for v9 index-only near-ATM trading."""

    cash: float
    initial_capital: float
    max_positions: int = 3
    max_risk_pct: float = 0.12  # Larger per-trade since fewer trades
    positions: list[V9Position] = field(default_factory=list)
    closed_trades: list[BacktestTrade] = field(default_factory=list)
    daily_values: list[tuple[str, float]] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        pos_value = sum(
            p.current_premium * p.contracts * 100 for p in self.positions
        )
        return self.cash + max(0.0, pos_value)

    def can_open(self) -> bool:
        return len(self.positions) < self.max_positions

    def position_size(self, conviction: float) -> float:
        base = self.total_value * self.max_risk_pct
        scale = 0.4 + 0.6 * (conviction / 10.0)
        return base * scale

    def open_position(
        self,
        signal: EntrySignal,
        bar: dict[str, Any],
        iv: float,
        dte: int = 21,
    ) -> V9Position | None:
        if not self.can_open():
            return None

        is_call = signal.direction == "bullish"
        close = float(bar["close"])

        # v9: Near-ATM strike (0.5% OTM)
        otm = V9_OTM_PCT
        if is_call:
            strike = close * (1.0 + otm)
        else:
            strike = close * (1.0 - otm)

        t_years = max(dte, 1) / TRADING_DAYS_PER_YEAR
        theo_premium = bs_price(close, strike, t_years, RISK_FREE_RATE, iv, is_call)

        # v9: Minimum premium filter — avoid penny options with huge spreads
        if theo_premium < 0.50:
            return None

        # v9: Apply entry slippage (buy at ask, not mid)
        fill_premium = apply_entry_slippage(theo_premium, otm, signal.ticker, V9_SLIPPAGE)

        budget = self.position_size(signal.conviction)
        contracts = max(1, int(budget / (fill_premium * 100)))
        cost = contracts * fill_premium * 100

        if cost > self.cash * 0.90:
            contracts = max(1, int(self.cash * 0.85 / (fill_premium * 100)))
            cost = contracts * fill_premium * 100

        if cost > self.cash or cost < 10:
            return None

        self.cash -= cost
        pos = V9Position(
            ticker=signal.ticker,
            direction=signal.direction,
            strategy=signal.strategy,
            is_call=is_call,
            entry_date=bar["date"],
            entry_underlying=close,
            strike=round(strike, 2),
            entry_premium_theo=theo_premium,
            entry_premium_fill=fill_premium,
            contracts=contracts,
            cost_basis=cost,
            iv=iv,
            conviction=signal.conviction,
            regime=signal.regime,
            dte_at_entry=dte,
            otm_pct=otm,
        )
        self.positions.append(pos)
        return pos

    def close_position(
        self,
        pos: V9Position,
        current_date: str,
        reason: str,
    ) -> BacktestTrade:
        # v9: Apply exit slippage (sell at bid, not mid)
        theo_exit = max(0.0, pos.current_premium)
        fill_exit = apply_exit_slippage(theo_exit, pos.otm_pct, pos.ticker, V9_SLIPPAGE)
        exit_value = fill_exit * pos.contracts * 100
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
                / pos.entry_underlying * 100
            )

        trade = BacktestTrade(
            ticker=pos.ticker,
            entry_date=date.fromisoformat(pos.entry_date),
            exit_date=date.fromisoformat(current_date),
            option_type="call" if pos.is_call else "put",
            strike=pos.strike,
            entry_price=round(pos.entry_premium_fill, 4),
            exit_price=round(fill_exit, 4),
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


def _v9_check_exits(
    portfolio: V9Portfolio,
    today_bars: dict[str, dict[str, Any]],
    ticker_history: dict[str, list[dict[str, Any]]],
    current_date: str,
) -> None:
    to_close: list[tuple[V9Position, str]] = []

    for pos in portfolio.positions:
        bar = today_bars.get(pos.ticker)
        if not bar:
            continue
        if pos.entry_premium_fill <= 0:
            to_close.append((pos, "invalid"))
            continue

        pnl_pct = (pos.current_premium - pos.entry_premium_fill) / pos.entry_premium_fill
        profile = V9_STOPS.get(pos.strategy, V9_STOPS["trend_pullback"])

        h_stop = profile["hard_stop"]
        t_stop = profile["trailing_stop"]
        tp = profile["take_profit"]
        max_hold = int(profile["max_hold"])

        # High conviction → 10% wider stops
        if pos.conviction >= 9.0:
            h_stop *= 1.10
            max_hold = int(max_hold * 1.15)

        if pnl_pct <= h_stop:
            to_close.append((pos, "hard_stop"))
            continue

        if pnl_pct >= tp:
            to_close.append((pos, "take_profit"))
            continue

        if pos.max_premium > pos.entry_premium_fill * 1.15:
            trail_level = pos.max_premium * (1.0 - t_stop)
            if pos.current_premium <= trail_level:
                to_close.append((pos, "trailing_stop"))
                continue

        if pos.days_held >= max_hold:
            to_close.append((pos, "time_exit"))
            continue

        # Regime reversal
        history = ticker_history.get(pos.ticker, [])
        if len(history) >= WARMUP_BARS and pos.days_held >= 3:
            indicators = compute_indicators(history)
            regime = detect_regime(indicators)
            if pos.is_call and regime in (
                MarketRegime.DOWNTREND, MarketRegime.STRONG_DOWNTREND,
            ):
                to_close.append((pos, "regime_reversal"))
                continue
            if not pos.is_call and regime in (
                MarketRegime.UPTREND, MarketRegime.STRONG_UPTREND,
            ):
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


# ── Metrics ──────────────────────────────────────────────────────


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
    return round(
        (annual_return - RISK_FREE_RATE) / annual_std, 3
    ) if annual_std > 0 else 0.0


# ── Main Entry Point ──────────────────────────────────────────────


async def run_v9_backtest(
    tickers: list[str] | None = None,
    lookback_days: int = 730,
    starting_capital: float = 10_000.0,
    max_positions: int = 3,
    min_conviction: float = 8.0,
    dte: int = 21,
    settings: Settings | None = None,
) -> BacktestResult:
    """Run v9 backtest: SPY/QQQ only, near-ATM, slippage-aware.

    Key differences from main engine:
    - Only trades SPY and QQQ
    - Near-ATM strikes (0.5% OTM)
    - Slippage deducted from every fill
    - Higher DTE (21 days) for more premium
    - Adaptive scorer with veto power
    - Aggressive min_conviction (8.0)
    """
    import asyncio

    tickers = tickers or V9_TICKERS
    settings = settings or get_settings()
    polygon = PolygonProvider(settings)

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    # Fetch data
    all_bars: dict[str, list[dict[str, Any]]] = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            await asyncio.sleep(13)
        try:
            bars = await _fetch_bars(
                polygon, ticker, start_date.isoformat(), end_date.isoformat(),
            )
            if len(bars) >= WARMUP_BARS + 20:
                all_bars[ticker] = bars
                logger.info("v9_bars_loaded", ticker=ticker, count=len(bars))
        except Exception as e:
            logger.warning("v9_fetch_failed", ticker=ticker, error=str(e))

    await polygon.close()

    if not all_bars:
        return BacktestResult(
            run_id=f"v9-{uuid.uuid4().hex[:8]}",
            tickers=tickers,
            lookback_days=lookback_days,
            starting_capital=starting_capital,
        )

    # Build date index
    bars_by_date: dict[str, dict[str, dict[str, Any]]] = {}
    all_dates: set[str] = set()
    for ticker, bars in all_bars.items():
        for bar in bars:
            d = bar["date"]
            all_dates.add(d)
            bars_by_date.setdefault(d, {})[ticker] = bar
    sorted_dates = sorted(all_dates)

    # Scorer weights
    scorer_weights = ScorerWeights(
        ticker_wr_weight=3.0,
        strategy_wr_weight=2.5,
        regime_wr_weight=1.5,
        bias=5.5,  # Slightly higher for index ETFs
    )

    # Initialize portfolio
    portfolio = V9Portfolio(
        cash=starting_capital,
        initial_capital=starting_capital,
        max_positions=max_positions,
    )
    ticker_history: dict[str, list[dict[str, Any]]] = {t: [] for t in all_bars}

    consecutive_losses = 0
    pause_remaining = 0

    # Walk each day
    for current_date in sorted_dates:
        today_bars = bars_by_date.get(current_date, {})

        for ticker in all_bars:
            if ticker in today_bars:
                ticker_history[ticker].append(today_bars[ticker])

        max_history = max(len(ticker_history[t]) for t in all_bars)
        if max_history < WARMUP_BARS:
            continue

        # Update positions
        portfolio.update_positions(today_bars)

        # Check exits
        pre_exit_count = len(portfolio.closed_trades)
        _v9_check_exits(portfolio, today_bars, ticker_history, current_date)

        # Track losses for circuit breaker
        new_closes = portfolio.closed_trades[pre_exit_count:]
        for closed in new_closes:
            if closed.outcome == TradeOutcome.WIN:
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                if consecutive_losses >= MAX_CONSEC_LOSSES:
                    pause_remaining = PAUSE_AFTER_STREAK

        # Scan for entries
        if portfolio.can_open():
            all_signals: list[EntrySignal] = []
            for ticker in all_bars:
                history = ticker_history.get(ticker, [])
                if len(history) < WARMUP_BARS:
                    continue
                if any(p.ticker == ticker for p in portfolio.positions):
                    continue

                indicators = compute_indicators(history)
                regime = detect_regime(indicators)

                # v9.2: Regime filter — only trade in allowed regimes
                if regime.value not in V9_ALLOWED_REGIMES:
                    continue

                signals = scan_for_entries(ticker, history, indicators, regime)

                # v9: Override OTM to near-ATM
                for sig in signals:
                    sig.otm_pct = V9_OTM_PCT
                all_signals.extend(signals)

            all_signals.sort(key=lambda s: s.conviction, reverse=True)

            for signal in all_signals:
                if not portfolio.can_open():
                    break

                # Circuit breaker
                if pause_remaining > 0:
                    pause_remaining -= 1
                    continue

                sig_bar = today_bars.get(signal.ticker)
                if not sig_bar:
                    continue

                history = ticker_history.get(signal.ticker, [])
                indicators = compute_indicators(history)
                iv = estimate_iv_from_atr(indicators.atr14, float(sig_bar["close"]))

                # Adaptive scorer with veto
                sig_regime = MarketRegime(signal.regime)
                features = extract_features(
                    signal.ticker, signal.direction, signal.strategy,
                    indicators, sig_regime, history,
                )
                adaptive_conv, _ = compute_adaptive_conviction(
                    features, scorer_weights,
                )

                if adaptive_conv < min_conviction:
                    continue

                blended = 0.20 * signal.conviction + 0.80 * adaptive_conv
                signal.conviction = round(max(0.0, min(10.0, blended)), 2)

                if signal.conviction < min_conviction:
                    continue

                portfolio.open_position(signal, sig_bar, iv, dte=dte)

        portfolio.record_snapshot(current_date)

    # Close remaining
    if sorted_dates:
        for pos in portfolio.positions[:]:
            portfolio.close_position(pos, sorted_dates[-1], "end_of_backtest")

    # Compile results
    all_trades = portfolio.closed_trades
    total = len(all_trades)
    wins = sum(1 for t in all_trades if t.outcome == TradeOutcome.WIN)
    losses = sum(
        1 for t in all_trades
        if t.outcome in (TradeOutcome.LOSS, TradeOutcome.EXPIRED)
    )
    expired = sum(1 for t in all_trades if t.outcome == TradeOutcome.EXPIRED)

    win_pnls = [t.pnl_pct for t in all_trades if t.outcome == TradeOutcome.WIN]
    loss_pnls = [t.pnl_pct for t in all_trades if t.outcome != TradeOutcome.WIN]

    gross_profit = sum(t.pnl_pct for t in all_trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct for t in all_trades if t.pnl_pct < 0))

    ticker_stats: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        tt = [t for t in all_trades if t.ticker == ticker]
        if tt:
            tw = sum(1 for t in tt if t.outcome == TradeOutcome.WIN)
            ticker_stats[ticker] = {
                "trades": float(len(tt)),
                "win_rate": round(tw / len(tt), 3),
                "avg_pnl_pct": round(sum(t.pnl_pct for t in tt) / len(tt), 2),
                "total_pnl_pct": round(sum(t.pnl_pct for t in tt), 2),
            }

    strategy_stats: dict[str, dict[str, float]] = {}
    for strat in {t.strategy for t in all_trades}:
        st = [t for t in all_trades if t.strategy == strat]
        sw = sum(1 for t in st if t.outcome == TradeOutcome.WIN)
        strategy_stats[strat] = {
            "trades": float(len(st)),
            "win_rate": round(sw / len(st), 3) if st else 0.0,
            "avg_pnl_pct": round(sum(t.pnl_pct for t in st) / len(st), 2) if st else 0.0,
            "total_pnl_pct": round(sum(t.pnl_pct for t in st), 2),
        }

    ending = portfolio.total_value
    port_return = (ending - starting_capital) / starting_capital * 100

    result = BacktestResult(
        run_id=f"v9-{uuid.uuid4().hex[:8]}",
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
            round(sum(t.hold_days for t in all_trades) / total, 1)
            if total > 0 else 0.0
        ),
        max_consecutive_wins=0,
        max_consecutive_losses=0,
        expectancy_pct=(
            round(sum(t.pnl_pct for t in all_trades) / total, 2)
            if total > 0 else 0.0
        ),
        trades=all_trades,
        by_ticker=ticker_stats,
        by_strategy=strategy_stats,
        starting_capital=starting_capital,
        ending_value=round(ending, 2),
        portfolio_return_pct=round(port_return, 2),
        max_drawdown_pct=_max_drawdown(portfolio.daily_values),
        sharpe_ratio=_sharpe(portfolio.daily_values),
    )

    logger.info(
        "v9_backtest_complete",
        tickers=list(all_bars.keys()),
        total_trades=total,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        portfolio_return=f"{port_return:.1f}%",
        sharpe=result.sharpe_ratio,
        note="slippage_included",
    )
    return result
