"""PHANTOM Hybrid — "FlowEdge Active" model.

Higher frequency model (15-30 trades/year) designed for $25K-$100K accounts.
Runs alongside v10.2 ultra-selective as a parallel strategy.

Key design (all data-driven from v3-v10 forensics):

TICKERS: SPY, QQQ, IWM + AAPL, META, GOOGL, AMD, NVDA (8 tickers)
- Indices: 0.5% OTM (tight spreads, higher delta)
- Stocks: 1.0% OTM (balance spread cost vs leverage)

STRATEGIES:
- IBS reversion (46% WR historically — core edge)
- Trend pullback (29% base but 37.5% on SPY near-ATM — viable with filters)
- Vol squeeze (rare but large payoffs — included for diversification)

DIRECTION: Both calls AND puts
- Calls in strong_uptrend (buy dips)
- Puts in strong_downtrend (buy bounces)

EXITS (data-driven, no hard stops):
- Take profit: 50% for indices, 80% for stocks (100% WR historically)
- Time exit: 12 days max (90% WR historically)
- Trailing stop: 25% trail, activates at 30% profit (wider than v7-v9)
- Regime reversal: exit if regime flips against position
- Emergency stop: -60% only (catastrophic protection, not regular stop)
- Min hold: 3 days (day 1-2 = 17% WR, too early to exit)

SCORING: Adaptive scorer (r=0.196 correlation)
- Veto power: adaptive must pass 8.0+ conviction
- Blend: 25% signal + 75% adaptive

RISK:
- Max 4 concurrent positions
- Max 1 position per ticker
- 10% of portfolio per trade
- 3-day cooldown between entries on same ticker
- Circuit breaker: pause after 3 consecutive losses
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

# ── Hybrid Configuration ──────────────────────────────────────────

# v7: Proven 5-ticker universe — only tickers with 60%+ WR in v6 forensics
# SPY=80%, META=71%, IWM=60%, QQQ=50% (kept for frequency), AAPL=50%
# NVDA/XLF dropped — added noise on the v7 expansion test
HYBRID_TICKERS = [
    "SPY", "QQQ", "IWM",  # Index ETFs (proven)
    "AAPL", "META",        # Best single names
]

INDEX_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}

# OTM by tier: near-ATM for indices, slightly OTM for stocks
# v5: OTM by tier — indices tight, stocks slightly wider
OTM_BY_TICKER: dict[str, float] = {
    # Index ETFs: near-ATM
    "SPY": 0.005, "QQQ": 0.003, "IWM": 0.005, "DIA": 0.005,
    "XLF": 0.005, "XLK": 0.004,
    # Single names: slightly OTM
    "AAPL": 0.008, "META": 0.008, "NVDA": 0.010,
    "MSFT": 0.008, "JPM": 0.008, "V": 0.008,
}

# Take-profit by tier
TP_BY_TIER: dict[str, float] = {
    "index": 0.50,   # 50% TP for indices (near-ATM, smaller % moves)
    "stock": 0.80,   # 80% TP for stocks (slightly OTM, larger % moves)
}

# Slippage model
HYBRID_SLIPPAGE = SlippageModel(
    base_spread_pct=0.015,
    otm_spread_multiplier=0.6,
    cheap_option_floor=0.02,
    market_impact_pct=0.003,
    enabled=True,
)

# Exit parameters
EMERGENCY_STOP = -0.60  # Only catastrophic protection
MAX_HOLD = 7  # v7: keep at 7 — conviction 9.5+ filter handles quality
MIN_HOLD = 3
TRAIL_ACTIVATION = 0.30
TRAIL_PCT = 0.25

# Entry parameters
MIN_CONVICTION = 9.5  # v7: raised from 8.5 — conv 9.5+ = 76% WR vs 29% below
MIN_PREMIUM = 0.30  # Lower than v10 since stocks have cheaper OTM options
DTE = 21

# Risk management
MAX_POSITIONS = 4
MAX_PER_TICKER = 1
RISK_PER_TRADE = 0.10
COOLDOWN_DAYS = 3  # Between entries on same ticker
MAX_CONSEC_LOSSES = 3
PAUSE_AFTER_STREAK = 3

# v4: Ticker-specific IBS thresholds
# IWM is more volatile — needs more extreme IBS to signal
# SPY/QQQ are smoother — standard threshold works
IBS_THRESHOLD_BY_TICKER: dict[str, float] = {
    "SPY": 0.20, "QQQ": 0.18, "IWM": 0.15, "DIA": 0.20,
    "XLF": 0.18, "XLK": 0.18,
    "AAPL": 0.18, "META": 0.18, "NVDA": 0.15,
    "MSFT": 0.18, "JPM": 0.18, "V": 0.18,
}
IBS_OVERSOLD_DEFAULT = 0.18
IBS_OVERBOUGHT_DEFAULT = 0.82
RSI_OVERSOLD = 45.0
RSI_OVERBOUGHT = 55.0

# v2: strong_uptrend only — downtrend was 30.4% WR, barely positive
ALLOWED_REGIMES = {MarketRegime.STRONG_UPTREND}


# ── Position & Portfolio ──────────────────────────────────────────


@dataclass
class HybridPosition:
    ticker: str
    direction: str
    strategy: str
    is_call: bool
    entry_date: str
    entry_underlying: float
    strike: float
    entry_premium_fill: float
    contracts: int
    cost_basis: float
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
class HybridPortfolio:
    cash: float
    initial_capital: float
    max_positions: int = MAX_POSITIONS
    positions: list[HybridPosition] = field(default_factory=list)
    closed_trades: list[BacktestTrade] = field(default_factory=list)
    daily_values: list[tuple[str, float]] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        pv = sum(p.current_premium * p.contracts * 100 for p in self.positions)
        return self.cash + max(0.0, pv)

    def can_open(self) -> bool:
        return len(self.positions) < self.max_positions

    def has_ticker(self, ticker: str) -> bool:
        return any(p.ticker == ticker for p in self.positions)

    def record_snapshot(self, date_str: str) -> None:
        self.daily_values.append((date_str, self.total_value))


# ── IBS Scanner ──────────────────────────────────────────────────


def _compute_ibs(bar: dict[str, Any]) -> float:
    high = float(bar.get("high", 0))
    low = float(bar.get("low", 0))
    close = float(bar.get("close", 0))
    rng = high - low
    return (close - low) / rng if rng > 0 else 0.5


def _compute_vol_ratio(bars: list[dict[str, Any]], period: int = 20) -> float:
    if len(bars) < period + 1:
        return 1.0
    vols = [float(b.get("volume", 0)) for b in bars[-(period + 1):-1]]
    avg = sum(vols) / len(vols) if vols else 1.0
    return float(bars[-1].get("volume", 0)) / avg if avg > 0 else 1.0


def _scan_hybrid_ibs(
    ticker: str,
    bars: list[dict[str, Any]],
    indicators: Any,
    regime: MarketRegime,
) -> dict[str, Any] | None:
    """Scan for IBS reversion entry on any ticker."""
    if regime not in ALLOWED_REGIMES:
        return None
    if len(bars) < 2:
        return None

    ibs = _compute_ibs(bars[-1])
    prev_close = float(bars[-2].get("close", 0))
    curr_close = float(bars[-1].get("close", 0))

    # v4: Ticker-specific IBS threshold
    ibs_thresh = IBS_THRESHOLD_BY_TICKER.get(ticker, IBS_OVERSOLD_DEFAULT)
    ibs_ob_thresh = 1.0 - ibs_thresh

    if regime == MarketRegime.STRONG_UPTREND:
        if ibs >= ibs_thresh or indicators.rsi14 >= RSI_OVERSOLD:
            return None
        if curr_close >= prev_close:
            return None  # Confirm dip
        direction = "bullish"
        ibs_extreme = (ibs_thresh - ibs) / ibs_thresh
    elif regime == MarketRegime.STRONG_DOWNTREND:
        if ibs <= ibs_ob_thresh or indicators.rsi14 <= RSI_OVERBOUGHT:
            return None
        if curr_close <= prev_close:
            return None  # Confirm bounce
        direction = "bearish"
        ibs_extreme = (ibs - ibs_ob_thresh) / ibs_thresh
    else:
        return None

    conviction = 7.0 + ibs_extreme * 2.0
    if indicators.adx14 > 25:
        conviction += 0.3
    if indicators.adx14 > 35:
        conviction += 0.3

    return {
        "ticker": ticker,
        "direction": direction,
        "strategy": "ibs_reversion",
        "conviction": min(10.0, conviction),
        "ibs": round(ibs, 4),
    }


# ── Open / Close ──────────────────────────────────────────────────


def _open_hybrid_position(
    portfolio: HybridPortfolio,
    ticker: str,
    bar: dict[str, Any],
    iv: float,
    conviction: float,
    regime: str,
    direction: str,
    strategy: str,
) -> HybridPosition | None:
    if not portfolio.can_open():
        return None
    if portfolio.has_ticker(ticker):
        return None

    close = float(bar["close"])
    otm = OTM_BY_TICKER.get(ticker, 0.010)
    is_call = direction == "bullish"
    strike = close * (1.0 + otm) if is_call else close * (1.0 - otm)

    t_years = max(DTE, 1) / TRADING_DAYS_PER_YEAR
    theo = bs_price(close, strike, t_years, RISK_FREE_RATE, iv, is_call)
    if theo < MIN_PREMIUM:
        return None

    fill = apply_entry_slippage(theo, otm, ticker, HYBRID_SLIPPAGE)

    budget = portfolio.total_value * RISK_PER_TRADE
    scale = 0.4 + 0.6 * (conviction / 10.0)
    budget *= scale

    contracts = max(1, int(budget / (fill * 100)))
    cost = contracts * fill * 100

    if cost > portfolio.cash * 0.90:
        contracts = max(1, int(portfolio.cash * 0.85 / (fill * 100)))
        cost = contracts * fill * 100

    if cost > portfolio.cash or cost < 10:
        return None

    portfolio.cash -= cost
    pos = HybridPosition(
        ticker=ticker,
        direction=direction,
        strategy=strategy,
        is_call=is_call,
        entry_date=bar["date"],
        entry_underlying=close,
        strike=round(strike, 2),
        entry_premium_fill=fill,
        contracts=contracts,
        cost_basis=cost,
        iv=iv,
        conviction=conviction,
        regime=regime,
        dte_at_entry=DTE,
        otm_pct=otm,
    )
    portfolio.positions.append(pos)
    return pos


def _close_hybrid_position(
    portfolio: HybridPortfolio,
    pos: HybridPosition,
    current_date: str,
    reason: str,
) -> BacktestTrade:
    theo_exit = max(0.0, pos.current_premium)
    fill_exit = apply_exit_slippage(
        theo_exit, pos.otm_pct, pos.ticker, HYBRID_SLIPPAGE,
    )
    exit_value = fill_exit * pos.contracts * 100
    pnl = exit_value - pos.cost_basis
    pnl_pct = (pnl / pos.cost_basis * 100) if pos.cost_basis > 0 else 0.0

    portfolio.cash += exit_value
    if pos in portfolio.positions:
        portfolio.positions.remove(pos)

    if pnl_pct >= 5:
        outcome = TradeOutcome.WIN
    elif pnl_pct <= -90:
        outcome = TradeOutcome.EXPIRED
    elif pnl_pct < -5:
        outcome = TradeOutcome.LOSS
    else:
        outcome = TradeOutcome.BREAKEVEN

    um = 0.0
    if pos.entry_underlying > 0:
        um = (pos.current_underlying - pos.entry_underlying) / pos.entry_underlying * 100

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
        underlying_move_pct=round(um, 2),
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
    portfolio.closed_trades.append(trade)
    return trade


# ── Exit Logic ──────────────────────────────────────────────────


def _check_hybrid_exits(
    portfolio: HybridPortfolio,
    ticker_history: dict[str, list[dict[str, Any]]],
    current_date: str,
) -> None:
    to_close: list[tuple[HybridPosition, str]] = []

    for pos in portfolio.positions:
        if pos.entry_premium_fill <= 0:
            to_close.append((pos, "invalid"))
            continue

        pnl_pct = (
            (pos.current_premium - pos.entry_premium_fill) / pos.entry_premium_fill
        )

        # TP tier
        tp = TP_BY_TIER["index"] if pos.ticker in INDEX_TICKERS else TP_BY_TIER["stock"]

        # 1. Take profit
        if pnl_pct >= tp:
            to_close.append((pos, "take_profit"))
            continue

        # Never exit before min hold
        if pos.days_held < MIN_HOLD:
            continue

        # 2. Emergency stop (catastrophic only)
        if pnl_pct <= EMERGENCY_STOP:
            to_close.append((pos, "emergency_stop"))
            continue

        # 3. Time exit
        if pos.days_held >= MAX_HOLD:
            to_close.append((pos, "time_exit"))
            continue

        # 4. Trailing stop (wide)
        if (
            pos.days_held >= MIN_HOLD
            and pos.max_premium > pos.entry_premium_fill * (1.0 + TRAIL_ACTIVATION)
        ):
            trail_level = pos.max_premium * (1.0 - TRAIL_PCT)
            if pos.current_premium <= trail_level:
                to_close.append((pos, "trailing_stop"))
                continue

        # 5. Regime reversal
        history = ticker_history.get(pos.ticker, [])
        if len(history) >= WARMUP_BARS and pos.days_held >= 4:
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
            _close_hybrid_position(portfolio, pos, current_date, reason)


# ── Data ──────────────────────────────────────────────────────────


async def _fetch_bars(
    polygon: PolygonProvider, ticker: str, from_d: str, to_d: str,
) -> list[dict[str, Any]]:
    data = await polygon._get(
        f"{polygon._base_url}/v2/aggs/ticker/{ticker}/range/1/day/{from_d}/{to_d}",
        params={"apiKey": polygon._api_key, "limit": "5000", "sort": "asc"},
    )
    return [
        {
            "date": date.fromtimestamp(r["t"] / 1000).isoformat() if "t" in r else "",
            "open": float(r.get("o", 0)),
            "high": float(r.get("h", 0)),
            "low": float(r.get("l", 0)),
            "close": float(r.get("c", 0)),
            "volume": int(r.get("v", 0)),
        }
        for r in data.get("results", [])
    ]


def _max_dd(dvs: list[tuple[str, float]]) -> float:
    if not dvs:
        return 0.0
    peak = dvs[0][1]
    md = 0.0
    for _, v in dvs:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0.0
        md = max(md, dd)
    return round(md, 2)


def _sharpe(dvs: list[tuple[str, float]]) -> float:
    if len(dvs) < 10:
        return 0.0
    rets = []
    for i in range(1, len(dvs)):
        p, c = dvs[i - 1][1], dvs[i][1]
        if p > 0:
            rets.append((c - p) / p)
    if not rets:
        return 0.0
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / len(rets)
    s = sqrt(var) if var > 0 else 0.001
    return round((m * 252 - RISK_FREE_RATE) / (s * sqrt(252)), 3) if s > 0 else 0.0


# ── Main Entry Point ──────────────────────────────────────────────


async def run_hybrid_backtest(
    tickers: list[str] | None = None,
    lookback_days: int = 730,
    starting_capital: float = 25_000.0,
    settings: Settings | None = None,
) -> BacktestResult:
    """Run the FlowEdge Active hybrid backtest.

    Expanded tickers, multiple strategies, both directions,
    slippage-aware, targeting 15-30 trades/year at 55-65% WR.
    """
    import asyncio

    tickers = tickers or HYBRID_TICKERS
    settings = settings or get_settings()
    polygon = PolygonProvider(settings)

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    # Fetch data
    all_bars: dict[str, list[dict[str, Any]]] = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            await asyncio.sleep(1)  # Paid tier — no rate limit
        try:
            bars = await _fetch_bars(
                polygon, ticker, start_date.isoformat(), end_date.isoformat(),
            )
            if len(bars) >= WARMUP_BARS + 20:
                all_bars[ticker] = bars
                logger.info("hybrid_bars_loaded", ticker=ticker, count=len(bars))
        except Exception as e:
            logger.warning("hybrid_fetch_failed", ticker=ticker, error=str(e))

    await polygon.close()

    if not all_bars:
        return BacktestResult(
            run_id=f"hybrid-{uuid.uuid4().hex[:8]}",
            tickers=tickers,
            lookback_days=lookback_days,
            starting_capital=starting_capital,
        )

    # Date index
    bars_by_date: dict[str, dict[str, dict[str, Any]]] = {}
    all_dates: set[str] = set()
    for ticker, bars in all_bars.items():
        for bar in bars:
            d = bar["date"]
            all_dates.add(d)
            bars_by_date.setdefault(d, {})[ticker] = bar
    sorted_dates = sorted(all_dates)

    # Scorer
    scorer_weights = ScorerWeights(
        ticker_wr_weight=3.0,
        strategy_wr_weight=2.5,
        regime_wr_weight=1.5,
        ibs_weight=1.2,
        bias=5.0,
    )

    # Portfolio
    portfolio = HybridPortfolio(
        cash=starting_capital,
        initial_capital=starting_capital,
    )
    ticker_history: dict[str, list[dict[str, Any]]] = {t: [] for t in all_bars}
    last_entry_by_ticker: dict[str, int] = {}

    consecutive_losses = 0
    pause_remaining = 0

    # Walk each day
    for day_idx, current_date in enumerate(sorted_dates):
        today_bars = bars_by_date.get(current_date, {})

        for ticker in all_bars:
            if ticker in today_bars:
                ticker_history[ticker].append(today_bars[ticker])

        max_hist = max(len(ticker_history[t]) for t in all_bars)
        if max_hist < WARMUP_BARS:
            continue

        # Update positions
        for pos in portfolio.positions:
            pos_bar = today_bars.get(pos.ticker)
            if not pos_bar:
                continue
            pos.days_held += 1
            pos.current_underlying = float(pos_bar["close"])
            remaining = max(1, pos.dte_at_entry - pos.days_held)
            t_years = remaining / TRADING_DAYS_PER_YEAR
            pos.current_premium = bs_price(
                pos.current_underlying, pos.strike, t_years,
                RISK_FREE_RATE, pos.iv, pos.is_call,
            )
            if pos.current_premium > pos.max_premium:
                pos.max_premium = pos.current_premium

        # Check exits
        pre_count = len(portfolio.closed_trades)
        _check_hybrid_exits(portfolio, ticker_history, current_date)

        for closed in portfolio.closed_trades[pre_count:]:
            if closed.outcome == TradeOutcome.WIN:
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                if consecutive_losses >= MAX_CONSEC_LOSSES:
                    pause_remaining = PAUSE_AFTER_STREAK

        # Scan for entries
        if portfolio.can_open() and pause_remaining <= 0:
            candidates: list[dict[str, Any]] = []

            for ticker in all_bars:
                history = ticker_history.get(ticker, [])
                if len(history) < WARMUP_BARS:
                    continue
                if portfolio.has_ticker(ticker):
                    continue

                # Cooldown check
                last_entry = last_entry_by_ticker.get(ticker, -999)
                if day_idx - last_entry < COOLDOWN_DAYS:
                    continue

                indicators = compute_indicators(history)
                regime = detect_regime(indicators)

                # IBS reversion scan (our primary edge)
                ibs_signal = _scan_hybrid_ibs(ticker, history, indicators, regime)
                if ibs_signal is not None:
                    candidates.append(ibs_signal)

                # v3: trend_pullback REMOVED — 40.9% WR, -137% total PnL
                # IBS reversion is the only strategy with proven edge (66.7% WR)
                # Vol squeeze retained for rare high-payoff diversification
                if regime in ALLOWED_REGIMES:
                    strat_signals = scan_for_entries(
                        ticker, history, indicators, regime,
                    )
                    for sig in strat_signals:
                        if sig.strategy == "vol_squeeze":
                            candidates.append({
                                "ticker": ticker,
                                "direction": sig.direction,
                                "strategy": sig.strategy,
                                "conviction": sig.conviction,
                                "ibs": 0.5,
                            })

            # Score all candidates with adaptive scorer
            scored: list[tuple[float, dict[str, Any]]] = []
            for cand in candidates:
                tk = cand["ticker"]
                history = ticker_history.get(tk, [])
                if len(history) < WARMUP_BARS:
                    continue

                indicators = compute_indicators(history)
                regime = detect_regime(indicators)

                features = extract_features(
                    tk, cand["direction"], cand["strategy"],
                    indicators, regime, history,
                )
                adaptive_conv, _ = compute_adaptive_conviction(
                    features, scorer_weights,
                )

                if adaptive_conv < MIN_CONVICTION:
                    continue

                final = 0.25 * cand["conviction"] + 0.75 * adaptive_conv
                final = round(min(10.0, final), 2)

                if final >= MIN_CONVICTION:
                    scored.append((final, cand))

            # Sort by conviction, take best
            scored.sort(key=lambda x: x[0], reverse=True)

            for final_conv, cand in scored:
                if not portfolio.can_open():
                    break
                if portfolio.has_ticker(cand["ticker"]):
                    continue

                tk = cand["ticker"]
                sig_bar = today_bars.get(tk)
                if not sig_bar:
                    continue

                history = ticker_history.get(tk, [])
                indicators = compute_indicators(history)
                iv = estimate_iv_from_atr(indicators.atr14, float(sig_bar["close"]))

                new_pos = _open_hybrid_position(
                    portfolio, tk, sig_bar, iv,
                    final_conv, detect_regime(indicators).value,
                    cand["direction"], cand["strategy"],
                )
                if new_pos is not None:
                    last_entry_by_ticker[tk] = day_idx
        elif pause_remaining > 0:
            pause_remaining -= 1

        portfolio.record_snapshot(current_date)

    # Close remaining
    if sorted_dates:
        for pos in portfolio.positions[:]:
            _close_hybrid_position(portfolio, pos, sorted_dates[-1], "end_of_backtest")

    # Compile results
    trades = portfolio.closed_trades
    total = len(trades)
    wins = sum(1 for t in trades if t.outcome == TradeOutcome.WIN)
    losses = sum(
        1 for t in trades if t.outcome in (TradeOutcome.LOSS, TradeOutcome.EXPIRED)
    )

    win_pnls = [t.pnl_pct for t in trades if t.outcome == TradeOutcome.WIN]
    loss_pnls = [t.pnl_pct for t in trades if t.outcome != TradeOutcome.WIN]
    gp = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gl = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))

    # By ticker
    ticker_stats: dict[str, dict[str, float]] = {}
    for tk in tickers:
        tt = [t for t in trades if t.ticker == tk]
        if tt:
            tw = sum(1 for t in tt if t.outcome == TradeOutcome.WIN)
            ticker_stats[tk] = {
                "trades": float(len(tt)),
                "win_rate": round(tw / len(tt), 3),
                "avg_pnl_pct": round(sum(t.pnl_pct for t in tt) / len(tt), 2),
                "total_pnl_pct": round(sum(t.pnl_pct for t in tt), 2),
            }

    # By strategy
    strat_stats: dict[str, dict[str, float]] = {}
    for strat in {t.strategy for t in trades}:
        st = [t for t in trades if t.strategy == strat]
        sw = sum(1 for t in st if t.outcome == TradeOutcome.WIN)
        strat_stats[strat] = {
            "trades": float(len(st)),
            "win_rate": round(sw / len(st), 3) if st else 0.0,
            "avg_pnl_pct": round(sum(t.pnl_pct for t in st) / len(st), 2) if st else 0.0,
            "total_pnl_pct": round(sum(t.pnl_pct for t in st), 2),
        }

    ending = portfolio.total_value
    port_return = (ending - starting_capital) / starting_capital * 100

    result = BacktestResult(
        run_id=f"hybrid-{uuid.uuid4().hex[:8]}",
        tickers=tickers,
        lookback_days=lookback_days,
        total_trades=total,
        wins=wins,
        losses=losses,
        win_rate=round(wins / total, 3) if total > 0 else 0.0,
        avg_win_pct=round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0.0,
        avg_loss_pct=round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0,
        best_trade_pct=round(max((t.pnl_pct for t in trades), default=0), 2),
        worst_trade_pct=round(min((t.pnl_pct for t in trades), default=0), 2),
        total_pnl_pct=round(sum(t.pnl_pct for t in trades), 2),
        profit_factor=round(gp / gl, 2) if gl > 0 else 0.0,
        avg_hold_days=round(sum(t.hold_days for t in trades) / total, 1) if total > 0 else 0.0,
        expectancy_pct=round(sum(t.pnl_pct for t in trades) / total, 2) if total > 0 else 0.0,
        trades=trades,
        by_ticker=ticker_stats,
        by_strategy=strat_stats,
        starting_capital=starting_capital,
        ending_value=round(ending, 2),
        portfolio_return_pct=round(port_return, 2),
        max_drawdown_pct=_max_dd(portfolio.daily_values),
        sharpe_ratio=_sharpe(portfolio.daily_values),
    )

    # Self-learning: update weights from trade outcomes
    from flowedge.scanner.backtest.learning_hook import post_backtest_learn
    if total >= 10:
        trade_dicts = [t.model_dump(mode="json") for t in trades]
        post_backtest_learn(trade_dicts, model_name="hybrid_active")

    return result
