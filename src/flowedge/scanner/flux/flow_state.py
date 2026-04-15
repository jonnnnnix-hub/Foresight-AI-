"""Extended per-ticker flow state that persists across scan cycles.

Builds on FLUX engine (Lee-Ready, cumulative delta, block detection)
with new sweep detection, rapid-fire tracking, and spread compression.

FlowStateManager maintains a dict[str, TickerFlowState] that the
unified orchestrator updates every 30 seconds from MassiveDataFeed
buffers. Scanners and the ScalpSignalScorer read from it.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.flux.engine import (
    _classify_trades_lee_ready,
    _compute_cumulative_delta,
    _compute_quote_imbalance,
    _detect_block_prints,
    _detect_divergence,
    _determine_bias,
)
from flowedge.scanner.flux.schemas import (
    BlockPrint,
    CumulativeDelta,
    DeltaDivergence,
    FlowBias,
    QuoteImbalance,
    TradeDirection,
)

logger = structlog.get_logger()


# ── Trigger conditions ────────────────────────────────────────


@dataclass
class TriggerConditions:
    """Which anomalies are currently active for a ticker."""

    volume_spike: bool = False       # 1m vol > 3x 5m avg
    sweep_detected: bool = False     # Large multi-trade cluster
    aggressive_bias: bool = False    # Aggressor ratio > 0.75 or < 0.25
    spread_compression: bool = False # Spread tightening + volume
    unusual_size: bool = False       # Trade > 2 std dev above mean
    rapid_fire: bool = False         # > 5 same-dir trades in 10s

    @property
    def any_active(self) -> bool:
        return (
            self.volume_spike
            or self.sweep_detected
            or self.aggressive_bias
            or self.spread_compression
            or self.unusual_size
            or self.rapid_fire
        )

    @property
    def count(self) -> int:
        return sum([
            self.volume_spike,
            self.sweep_detected,
            self.aggressive_bias,
            self.spread_compression,
            self.unusual_size,
            self.rapid_fire,
        ])


# ── Per-ticker flow state ─────────────────────────────────────


@dataclass
class TickerFlowState:
    """Persistent flow state for one ticker, updated every cycle."""

    ticker: str
    last_updated: float = 0.0  # time.monotonic()

    # Rolling cumulative delta
    delta_1m: CumulativeDelta = field(
        default_factory=lambda: CumulativeDelta(window_minutes=1),
    )
    delta_5m: CumulativeDelta = field(
        default_factory=lambda: CumulativeDelta(window_minutes=5),
    )
    delta_15m: CumulativeDelta = field(
        default_factory=lambda: CumulativeDelta(window_minutes=15),
    )

    # Quote imbalance
    quote_imbalance: QuoteImbalance = field(
        default_factory=lambda: QuoteImbalance(window_minutes=5),
    )

    # Block prints (last cycle)
    block_prints: list[BlockPrint] = field(default_factory=list)
    block_bias: TradeDirection = TradeDirection.UNKNOWN

    # Divergence
    divergence: DeltaDivergence = DeltaDivergence.NONE

    # Overall bias
    bias: FlowBias = FlowBias.NEUTRAL

    # Sweep detection
    sweep_count_5m: int = 0

    # Rapid-fire detection
    rapid_fire_count: int = 0

    # Spread compression
    spread_compression_ratio: float = 1.0  # < 0.8 = tightening

    # Volume ratio (1m / 5m avg)
    volume_ratio_1m: float = 1.0

    # Bias history (for persistence detection)
    bias_history: deque[FlowBias] = field(
        default_factory=lambda: deque(maxlen=12),
    )

    # Trigger conditions
    triggers: TriggerConditions = field(default_factory=TriggerConditions)

    @property
    def is_bias_persistent(self) -> bool:
        """Same directional bias for 3+ consecutive readings."""
        if len(self.bias_history) < 3:
            return False
        last_3 = list(self.bias_history)[-3:]
        if all(b in (FlowBias.BUY, FlowBias.STRONG_BUY) for b in last_3):
            return True
        if all(b in (FlowBias.SELL, FlowBias.STRONG_SELL) for b in last_3):
            return True
        return False

    @property
    def aggression_ratio(self) -> float:
        """5-min aggression ratio (0.0-1.0)."""
        return self.delta_5m.aggression_ratio


# ── Flow State Manager ────────────────────────────────────────


class FlowStateManager:
    """Maintains per-ticker flow state from MassiveDataFeed buffers.

    The unified orchestrator calls update_loop() as a background task.
    Scanners and scorers read state via get_state(ticker).
    """

    def __init__(
        self,
        data_feed: Any,
        settings: Settings | None = None,
    ) -> None:
        self._data_feed = data_feed
        self._settings = settings or get_settings()
        self._states: dict[str, TickerFlowState] = {}
        self._tickers: list[str] = []

    def set_tickers(self, tickers: list[str]) -> None:
        self._tickers = [t.upper() for t in tickers]
        for t in self._tickers:
            if t not in self._states:
                self._states[t] = TickerFlowState(ticker=t)

    def get_state(self, ticker: str) -> TickerFlowState | None:
        return self._states.get(ticker.upper())

    def get_all_states(self) -> dict[str, TickerFlowState]:
        return dict(self._states)

    async def update_loop(self, interval: float = 30.0) -> None:
        """Background coroutine: update all ticker states periodically."""
        logger.info(
            "flow_state_manager_started",
            tickers=len(self._tickers),
            interval=interval,
        )
        while True:
            try:
                for ticker in self._tickers:
                    self._update_ticker(ticker)
            except Exception as e:
                logger.warning("flow_state_update_error", error=str(e))
            await asyncio.sleep(interval)

    def _update_ticker(self, ticker: str) -> None:
        """Recompute flow state for one ticker from data feed buffers."""
        state = self._states.get(ticker)
        if state is None:
            state = TickerFlowState(ticker=ticker)
            self._states[ticker] = state

        # Read buffers
        trades = self._data_feed.get_trades(ticker, window_minutes=15)
        quotes = self._data_feed.get_quotes(ticker, window_minutes=15)

        if not trades:
            state.last_updated = time.monotonic()
            return

        # Classify trades (Lee-Ready)
        classified = _classify_trades_lee_ready(trades, quotes)

        # Cumulative delta at multiple windows
        state.delta_1m = _compute_cumulative_delta(classified, window_minutes=1)
        state.delta_5m = _compute_cumulative_delta(classified, window_minutes=5)
        state.delta_15m = _compute_cumulative_delta(classified, window_minutes=15)

        # Quote imbalance
        if quotes:
            state.quote_imbalance = _compute_quote_imbalance(
                quotes, window_minutes=5,
            )

        # Block prints
        state.block_prints = _detect_block_prints(
            classified,
            ticker,
            min_multiple=self._settings.flux_block_min_multiple,
        )
        if state.block_prints:
            buy_notional = sum(
                b.notional for b in state.block_prints
                if b.direction == TradeDirection.BUY
            )
            sell_notional = sum(
                b.notional for b in state.block_prints
                if b.direction == TradeDirection.SELL
            )
            if buy_notional > sell_notional * 1.5:
                state.block_bias = TradeDirection.BUY
            elif sell_notional > buy_notional * 1.5:
                state.block_bias = TradeDirection.SELL
            else:
                state.block_bias = TradeDirection.UNKNOWN

        # Divergence (need price change from bars)
        bars = self._data_feed.get_bars(ticker, count=10)
        price_change = 0.0
        if len(bars) >= 2:
            if hasattr(bars[-2], "close") and bars[-2].close > 0:
                price_change = (
                    (bars[-1].close - bars[-2].close) / bars[-2].close
                )
            elif isinstance(bars[-2], dict) and bars[-2].get("close", 0) > 0:
                price_change = (
                    (bars[-1]["close"] - bars[-2]["close"]) / bars[-2]["close"]
                )

        state.divergence = _detect_divergence(
            state.delta_5m, state.delta_15m, price_change,
        )

        # Overall bias
        state.bias = _determine_bias(
            state.delta_5m, state.quote_imbalance, state.block_bias,
        )
        state.bias_history.append(state.bias)

        # ── New detections ────────────────────────────────────

        # Volume ratio: 1m vs 5m average
        vol_1m = state.delta_1m.buy_volume + state.delta_1m.sell_volume
        vol_5m = state.delta_5m.buy_volume + state.delta_5m.sell_volume
        avg_1m_from_5m = vol_5m / 5 if vol_5m > 0 else 1
        state.volume_ratio_1m = vol_1m / avg_1m_from_5m if avg_1m_from_5m > 0 else 1.0

        # Sweep detection: cluster of 3+ aggressive same-direction
        # trades > 3x avg size within 30 seconds
        state.sweep_count_5m = self._detect_sweeps(classified)

        # Rapid-fire: 5+ same-direction trades within 10 seconds
        state.rapid_fire_count = self._detect_rapid_fire(classified)

        # Spread compression: current spread vs 15-min average
        state.spread_compression_ratio = self._compute_spread_compression(quotes)

        # ── Update trigger conditions ─────────────────────────

        ar = state.delta_5m.aggression_ratio
        state.triggers = TriggerConditions(
            volume_spike=state.volume_ratio_1m > 3.0,
            sweep_detected=state.sweep_count_5m > 0,
            aggressive_bias=ar > 0.75 or ar < 0.25,
            spread_compression=(
                state.spread_compression_ratio < 0.80
                and state.volume_ratio_1m > 1.5
            ),
            unusual_size=bool(state.block_prints),
            rapid_fire=state.rapid_fire_count > 0,
        )

        state.last_updated = time.monotonic()

    # ── Detection helpers ─────────────────────────────────────

    @staticmethod
    def _detect_sweeps(classified: list[Any]) -> int:
        """Detect sweep-like clusters: 3+ aggressive same-direction
        trades > 3x avg size within a 30-second window.
        """
        if len(classified) < 5:
            return 0

        avg_size = sum(abs(t.signed_volume) for t in classified) / len(classified)
        if avg_size == 0:
            return 0
        threshold = avg_size * 3

        sweep_count = 0
        window_ns = 30 * 1_000_000_000  # 30 seconds in nanoseconds

        for i, trade in enumerate(classified):
            if abs(trade.signed_volume) < threshold:
                continue

            # Look for 2 more trades in same direction within 30s
            direction = trade.direction
            cluster = 1
            for j in range(i + 1, len(classified)):
                if classified[j].timestamp - trade.timestamp > window_ns:
                    break
                if (
                    classified[j].direction == direction
                    and abs(classified[j].signed_volume) >= threshold
                ):
                    cluster += 1

            if cluster >= 3:
                sweep_count += 1

        return sweep_count

    @staticmethod
    def _detect_rapid_fire(classified: list[Any]) -> int:
        """Detect rapid-fire: 5+ same-direction trades within 10 seconds."""
        if len(classified) < 5:
            return 0

        window_ns = 10 * 1_000_000_000  # 10 seconds
        rapid_count = 0

        for i, trade in enumerate(classified):
            if trade.direction == TradeDirection.UNKNOWN:
                continue

            same_dir = 1
            for j in range(i + 1, len(classified)):
                if classified[j].timestamp - trade.timestamp > window_ns:
                    break
                if classified[j].direction == trade.direction:
                    same_dir += 1

            if same_dir >= 5:
                rapid_count += 1
                break  # Count once per window

        return rapid_count

    @staticmethod
    def _compute_spread_compression(quotes: list[Any]) -> float:
        """Current spread vs 15-min average. < 0.8 = tightening."""
        if len(quotes) < 10:
            return 1.0

        spreads = []
        for q in quotes:
            s = q.spread if hasattr(q, "spread") else (q.ask - q.bid)
            if s > 0:
                spreads.append(s)

        if len(spreads) < 5:
            return 1.0

        avg_spread = sum(spreads) / len(spreads)
        current_spread = spreads[-1]

        if avg_spread <= 0:
            return 1.0

        return current_spread / avg_spread
