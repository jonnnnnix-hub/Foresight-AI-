"""FlowEdge Volume Scalper v1 — high-frequency volume-driven signal generator.

Runs on the SAME Alpaca paper account as scalp_v2 for head-to-head
performance comparison. Every trade is tagged "vol_scalp_v1" in the
client_order_id so the dashboard can track per-model P&L.

Design rationale:
- Scalp v2 (90% WR) fires ~1 signal/week — too slow for live validation
- Volume Scalper v1 fires 2-5 signals/day to collect live data faster
- Uses the BEST features from each model at relaxed thresholds:
  * From Scalp v2: IBS oversold + below VWAP (strongest edge)
  * From Rapid: 3-of-4 confluence (vs all-4) + volume surge
  * From Hybrid: daily IBS + gap down (regime awareness)
- Wider ticker universe (16 tickers vs 8 for scalp v2)
- Wider TP: 0.3% (vs 0.2%) to give trades room
- Longer hold: 20 bars / 100 min (vs 12 / 60 min)
- Lower conviction threshold: 6.0 (vs 7.0+)

Expected outcomes:
- ~2-5 signals/day across 16 tickers
- ~50-60% WR (vs 90% for strict scalp v2)
- Win/loss data to refine ALL models faster

Exit rules:
- Take profit: underlying +0.3% from entry
- Trailing stop: option premium -5% from peak
- Time exit: 20 bars (100 minutes)
- Emergency stop: -15% on option premium

Usage:
    .venv/bin/python -m flowedge.scanner.live.volume_scalper_v1_scanner
    .venv/bin/python scripts/run_volume_scalper.py
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv

from flowedge.notifications.email_alerts import (
    send_trade_entry_alert,
    send_trade_exit_alert,
)
from flowedge.scanner.data_feeds.alpaca_execution import AlpacaExecutor, AlpacaOrder
from flowedge.scanner.data_feeds.polygon_intraday import PolygonIntradayProvider
from flowedge.scanner.data_feeds.schemas import BarData, Timeframe

logger = structlog.get_logger()

# ── Constants ────────────────────────────────────────────────────

MODEL_NAME = "vol_scalp_v1"

# Timezone
ET_OFFSET = timezone(timedelta(hours=-4))

# Market hours
MARKET_OPEN = time(9, 35)
MARKET_CLOSE = time(15, 55)

# Entry window — wider than scalp v2
ENTRY_START = time(9, 40)
ENTRY_END = time(15, 30)

# Scan intervals
SIGNAL_SCAN_INTERVAL = 60   # Check every 60s
EXIT_CHECK_INTERVAL = 20    # Check exits every 20s

# Wider ticker universe — 16 tickers across sectors
VOL_SCALP_TICKERS = [
    # ETFs (most liquid, daily 0DTE)
    "SPY", "QQQ", "IWM",
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN",
    # High-beta / momentum
    "PLTR", "SOFI", "AMD",
    # Sector ETFs
    "XLK", "XLF",
    # Consumer / stable
    "COST", "WMT",
]

# ── Relaxed Signal Thresholds ────────────────────────────────────
# These are intentionally looser than scalp v2 to fire more often

# Core conditions (need 3 of 5 to fire)
IBS_THRESHOLD = 0.25         # vs 0.12 for scalp v2
RSI3_THRESHOLD = 25.0        # vs 15.0 for scalp v2
VOL_SPIKE_THRESHOLD = 1.8    # vs 2.5 for scalp v2
INTRADAY_DROP_THRESHOLD = -0.001  # vs -0.002 for scalp v2
VWAP_BELOW_REQUIRED = True   # Keep this — strongest single predictor

# Conviction
MIN_CONVICTION = 6.0         # vs 7.0 for scalp v2

# Exit parameters — wider than scalp v2
TP_UNDERLYING = 0.003        # 0.3% vs 0.2%
TRAIL_PCT = 0.05             # 5% from peak vs 3%
MAX_HOLD_BARS = 20           # 100 min vs 60 min
EMERGENCY_STOP_PCT = 0.15    # -15% option premium

# Position sizing
MAX_POSITIONS = 3
EQUITY_FRACTION = 0.04       # 4% per trade (smaller since more trades)

# Option selection
OPTION_MIN_DTE = 0
OPTION_MAX_DTE = 7
OPTION_MIN_BID = 0.30
OPTION_MAX_SPREAD_PCT = 12.0


# ── Signal & Position Types ──────────────────────────────────────

class VolScalpSignal:
    """An accelerator entry signal — fires on 3-of-5 conditions."""

    __slots__ = (
        "ticker", "timestamp", "conditions_met", "ibs", "rsi3",
        "vwap_gap_pct", "vol_ratio", "intraday_drop_pct",
        "underlying_price", "conviction",
    )

    def __init__(
        self,
        ticker: str,
        timestamp: str,
        conditions_met: list[str],
        ibs: float,
        rsi3: float,
        vwap_gap_pct: float,
        vol_ratio: float,
        intraday_drop_pct: float,
        underlying_price: float,
    ) -> None:
        self.ticker = ticker
        self.timestamp = timestamp
        self.conditions_met = conditions_met
        self.ibs = ibs
        self.rsi3 = rsi3
        self.vwap_gap_pct = vwap_gap_pct
        self.vol_ratio = vol_ratio
        self.intraday_drop_pct = intraday_drop_pct
        self.underlying_price = underlying_price
        self.conviction = self._compute_conviction()

    def _compute_conviction(self) -> float:
        score = 4.0 + len(self.conditions_met)  # 4 base + conditions
        # Bonus for extreme values
        if self.ibs < 0.10:
            score += 1.0
        elif self.ibs < 0.15:
            score += 0.5
        if self.rsi3 < 12.0:
            score += 0.5
        if self.vol_ratio > 3.0:
            score += 0.5
        return min(score, 10.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": MODEL_NAME,
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "conditions_met": self.conditions_met,
            "conditions_count": len(self.conditions_met),
            "ibs": round(self.ibs, 4),
            "rsi3": round(self.rsi3, 2),
            "vwap_gap_pct": round(self.vwap_gap_pct, 4),
            "vol_ratio": round(self.vol_ratio, 2),
            "intraday_drop_pct": round(self.intraday_drop_pct, 4),
            "underlying_price": round(self.underlying_price, 2),
            "conviction": round(self.conviction, 2),
        }


class VolScalpPosition:
    """A live accelerator position being monitored."""

    __slots__ = (
        "ticker", "option_symbol", "qty", "entry_price_underlying",
        "entry_price_option", "entry_time", "peak_option_price",
        "bars_held", "order_id",
    )

    def __init__(
        self,
        ticker: str,
        option_symbol: str,
        qty: int,
        entry_price_underlying: float,
        entry_price_option: float,
        entry_time: str,
        order_id: str,
    ) -> None:
        self.ticker = ticker
        self.option_symbol = option_symbol
        self.qty = qty
        self.entry_price_underlying = entry_price_underlying
        self.entry_price_option = entry_price_option
        self.entry_time = entry_time
        self.peak_option_price = entry_price_option
        self.bars_held = 0
        self.order_id = order_id


# ── Scanner ──────────────────────────────────────────────────────

class VolumeScalperV1Scanner:
    """High-frequency signal generator combining all model insights.

    Pulls 1-min bars from Polygon, aggregates to 5-min for signals,
    uses a 3-of-5 confluence filter (vs 7-of-7 for scalp v2).
    """

    def __init__(
        self,
        polygon: PolygonIntradayProvider,
        alpaca: AlpacaExecutor,
        tickers: list[str] | None = None,
        log_dir: str = "data/live_logs/vol_scalp_v1",
    ) -> None:
        self.polygon = polygon
        self.alpaca = alpaca
        self.tickers = tickers or VOL_SCALP_TICKERS
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.bars_1m: dict[str, list[BarData]] = {}
        self.bars_5m: dict[str, list[BarData]] = {}
        self.daily_bars: dict[str, list[BarData]] = {}
        self.day_opens: dict[str, float] = {}
        self.active_positions: list[VolScalpPosition] = []
        self.signals_today: list[dict[str, Any]] = []
        self.trades_today: list[dict[str, Any]] = []
        self.exits_today: list[dict[str, Any]] = []
        self._scan_count = 0

    # ── Data Loading ─────────────────────────────────────────────

    async def load_daily_bars(self) -> None:
        """Load last 30 trading days for trend filtering."""
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=60)).isoformat()
        for ticker in self.tickers:
            try:
                bars = await self.polygon.get_intraday_bars(
                    ticker, Timeframe.DAILY, start, end, limit=60,
                )
                self.daily_bars[ticker] = bars
            except Exception as e:
                logger.warning(
                    "daily_bar_fetch_failed",
                    model=MODEL_NAME,
                    ticker=ticker,
                    error=str(e),
                )
            await asyncio.sleep(0.2)

    async def refresh_bars(self) -> None:
        """Pull 1-min bars and aggregate to 5-min."""
        today = date.today().isoformat()
        for ticker in self.tickers:
            try:
                bars_1m = await self.polygon.get_intraday_bars(
                    ticker, Timeframe.MIN_1, today, today, limit=500,
                )
                self.bars_1m[ticker] = bars_1m

                if bars_1m and ticker not in self.day_opens:
                    self.day_opens[ticker] = bars_1m[0].open

                self.bars_5m[ticker] = self._aggregate_to_5m(bars_1m, ticker)
            except Exception as e:
                logger.warning(
                    "bar_fetch_failed",
                    model=MODEL_NAME,
                    ticker=ticker,
                    error=str(e),
                )

    @staticmethod
    def _aggregate_to_5m(
        bars_1m: list[BarData], ticker: str,
    ) -> list[BarData]:
        """Aggregate 1-min bars into 5-min bars."""
        if not bars_1m:
            return []

        from collections import defaultdict

        buckets: dict[int, list[BarData]] = defaultdict(list)
        for bar in bars_1m:
            ts_epoch = int(bar.timestamp.timestamp())
            bucket = ts_epoch // 300
            buckets[bucket].append(bar)

        bars_5m: list[BarData] = []
        for bucket_key in sorted(buckets):
            chunk = buckets[bucket_key]
            bars_5m.append(BarData(
                ticker=ticker,
                timestamp=chunk[0].timestamp,
                timeframe=Timeframe.MIN_5,
                open=chunk[0].open,
                high=max(b.high for b in chunk),
                low=min(b.low for b in chunk if b.low > 0),
                close=chunk[-1].close,
                volume=sum(b.volume for b in chunk),
                vwap=(
                    sum(b.vwap * b.volume for b in chunk if b.vwap > 0)
                    / max(1, sum(b.volume for b in chunk))
                ),
                trade_count=sum(b.trade_count for b in chunk),
            ))

        return bars_5m

    def get_latest_price(self, ticker: str) -> float:
        """Get freshest price from 1-min bars."""
        bars = self.bars_1m.get(ticker, [])
        if bars:
            return bars[-1].close
        bars_5m = self.bars_5m.get(ticker, [])
        if bars_5m:
            return bars_5m[-1].close
        return 0.0

    # ── Signal Detection (3-of-5 Confluence) ─────────────────────

    def check_signals(self) -> list[VolScalpSignal]:
        """Run the 3-of-5 confluence filter.

        Conditions checked:
        1. IBS < 0.25 (oversold bar)
        2. RSI(3) < 25 (momentum snap)
        3. Below VWAP (selling pressure)
        4. Volume spike > 1.8x (capitulation volume)
        5. Intraday drop < -0.1% (real dip, not flat)

        Signal fires when 3+ conditions are met AND price is below VWAP.
        VWAP-below is the strongest single predictor, so it's always required.
        """
        signals: list[VolScalpSignal] = []

        for ticker in self.tickers:
            bars = self.bars_5m.get(ticker, [])
            if len(bars) < 6:
                continue

            # Already have a position in this ticker?
            if any(p.ticker == ticker for p in self.active_positions):
                continue

            # Max positions
            if len(self.active_positions) >= MAX_POSITIONS:
                break

            i = len(bars) - 1
            bar = bars[i]

            # ── Compute features ─────────────────────────────
            rng = bar.high - bar.low
            if rng <= 0 or bar.close <= 0:
                continue
            ibs = (bar.close - bar.low) / rng

            # RSI(3)
            if len(bars) < 4:
                continue
            c5m = [bars[j].close for j in range(max(0, i - 3), i + 1)]
            if len(c5m) < 4:
                continue
            gains = [max(0, c5m[k] - c5m[k - 1]) for k in range(1, len(c5m))]
            losses = [max(0, c5m[k - 1] - c5m[k]) for k in range(1, len(c5m))]
            ag = sum(gains) / len(gains)
            al = sum(losses) / len(losses)
            rsi3 = 100.0 - 100.0 / (1 + ag / al) if al > 0 else 100.0

            # VWAP
            cum_pv = 0.0
            cum_v = 0.0
            for b in bars:
                tp = (b.high + b.low + b.close) / 3
                cum_pv += tp * b.volume
                cum_v += b.volume
            vwap = cum_pv / cum_v if cum_v > 0 else bar.close
            vwap_gap_pct = (bar.close - vwap) / vwap if vwap > 0 else 0

            # Volume ratio
            lookback_start = max(0, i - 10)
            lookback_count = i - lookback_start
            if lookback_count <= 0:
                continue
            avg_vol = (
                sum(bars[j].volume for j in range(lookback_start, i))
                / lookback_count
            )
            vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 1.0

            # Intraday drop
            day_open = self.day_opens.get(ticker, bar.open)
            intraday_drop = (
                (bar.close - day_open) / day_open if day_open > 0 else 0
            )

            # ── HARD GATE: Must be below VWAP ────────────────
            if bar.close >= vwap:
                continue

            # ── Count conditions met ─────────────────────────
            conditions: list[str] = []

            if ibs < IBS_THRESHOLD:
                conditions.append("ibs_oversold")
            if rsi3 < RSI3_THRESHOLD:
                conditions.append("rsi3_snap")
            if vol_ratio > VOL_SPIKE_THRESHOLD:
                conditions.append("vol_surge")
            if intraday_drop < INTRADAY_DROP_THRESHOLD:
                conditions.append("intraday_dip")

            # Prior bar red (bonus condition)
            if i >= 2 and bars[i - 1].close < bars[i - 2].close:
                conditions.append("prior_red")

            # Need at least 3 conditions (plus VWAP gate already passed)
            if len(conditions) < 3:
                continue

            signal = VolScalpSignal(
                ticker=ticker,
                timestamp=bar.timestamp.isoformat(),
                conditions_met=conditions,
                ibs=ibs,
                rsi3=rsi3,
                vwap_gap_pct=vwap_gap_pct,
                vol_ratio=vol_ratio,
                intraday_drop_pct=intraday_drop,
                underlying_price=bar.close,
            )

            if signal.conviction < MIN_CONVICTION:
                continue

            signals.append(signal)
            logger.info(
                "vol_scalp_v1_signal",
                model=MODEL_NAME,
                ticker=ticker,
                conditions=conditions,
                count=len(conditions),
                ibs=round(ibs, 4),
                rsi3=round(rsi3, 2),
                vol_ratio=round(vol_ratio, 2),
                conviction=round(signal.conviction, 2),
                price=round(bar.close, 2),
            )

        return signals

    # ── Execution ────────────────────────────────────────────────

    async def execute_signal(
        self, signal: VolScalpSignal,
    ) -> AlpacaOrder | None:
        """Find best option and place order."""
        ticker = signal.ticker

        # Verify position limits from Alpaca
        positions = await self.alpaca.get_positions()
        if len(positions) >= MAX_POSITIONS:
            logger.info(
                "max_positions_reached",
                model=MODEL_NAME,
                current=len(positions),
            )
            return None

        # Find ATM call option
        option = await self.polygon.get_nearest_atm_option(
            ticker,
            signal.underlying_price,
            option_type="call",
            min_dte=OPTION_MIN_DTE,
            max_dte=OPTION_MAX_DTE,
        )

        if not option:
            logger.warning("no_option_found", model=MODEL_NAME, ticker=ticker)
            return None

        if option.bid < OPTION_MIN_BID:
            logger.info(
                "option_below_min_premium",
                model=MODEL_NAME,
                bid=option.bid,
                ticker=ticker,
            )
            return None

        if option.spread_pct > OPTION_MAX_SPREAD_PCT:
            logger.info(
                "spread_too_wide",
                model=MODEL_NAME,
                spread_pct=option.spread_pct,
                ticker=ticker,
            )
            return None

        # Position sizing
        account = await self.alpaca.get_account()
        equity = float(account.get("equity", 0))
        budget = equity * EQUITY_FRACTION
        premium = option.ask
        if premium <= 0:
            return None
        contracts = max(1, int(budget / (premium * 100)))

        logger.info(
            "executing_vol_scalp_v1",
            model=MODEL_NAME,
            ticker=ticker,
            conviction=signal.conviction,
            conditions=signal.conditions_met,
            contract=option.contract_symbol,
            strike=option.strike,
            bid=option.bid,
            ask=option.ask,
            contracts=contracts,
        )

        # Place order
        order = await self.alpaca.buy_option(
            symbol=option.contract_symbol,
            qty=contracts,
            order_type="limit",
            limit_price=round(option.ask, 2),
            model_name=MODEL_NAME,
            conviction=signal.conviction,
        )

        # Track position
        pos = VolScalpPosition(
            ticker=ticker,
            option_symbol=option.contract_symbol,
            qty=contracts,
            entry_price_underlying=signal.underlying_price,
            entry_price_option=option.ask,
            entry_time=signal.timestamp,
            order_id=order.order_id,
        )
        self.active_positions.append(pos)

        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "action": "entry",
            "ticker": ticker,
            "direction": "bullish",
            "conviction": signal.conviction,
            "conditions": signal.conditions_met,
            "contract": option.contract_symbol,
            "strike": option.strike,
            "expiration": option.expiration,
            "premium": option.ask,
            "contracts": contracts,
            "order_id": order.order_id,
            "status": order.status,
            "signal": signal.to_dict(),
        }
        self.trades_today.append(trade_record)

        # Email alert
        try:
            await send_trade_entry_alert(trade_record)
        except Exception as e:
            logger.warning("entry_email_failed", error=str(e))

        return order

    # ── Exit Monitoring ──────────────────────────────────────────

    async def check_exits(self) -> None:
        """Monitor positions for exit conditions.

        Exit rules:
        1. TP: underlying +0.3% from entry
        2. Trailing stop: option -5% from peak
        3. Time exit: 20 bars (100 min)
        4. Emergency stop: option -15% from entry
        """
        to_close: list[tuple[VolScalpPosition, str]] = []

        for pos in self.active_positions:
            pos.bars_held += 1

            current_underlying = self.get_latest_price(pos.ticker)
            if current_underlying <= 0:
                continue

            # Get option price from Alpaca
            alpaca_positions = await self.alpaca.get_positions()
            option_pos = next(
                (p for p in alpaca_positions if p.ticker == pos.option_symbol),
                None,
            )
            current_option = (
                option_pos.current_price if option_pos
                else pos.entry_price_option
            )

            # Update peak
            if current_option > pos.peak_option_price:
                pos.peak_option_price = current_option

            # ── EXIT 1: Take Profit ──────────────────────────
            underlying_move = (
                (current_underlying - pos.entry_price_underlying)
                / pos.entry_price_underlying
            )
            if underlying_move >= TP_UNDERLYING:
                to_close.append((pos, "take_profit"))
                continue

            # ── EXIT 2: Trailing Stop ────────────────────────
            if pos.peak_option_price > 0:
                drawdown = (
                    (pos.peak_option_price - current_option)
                    / pos.peak_option_price
                )
                if drawdown >= TRAIL_PCT:
                    to_close.append((pos, "trailing_stop"))
                    continue

            # ── EXIT 3: Time Exit ────────────────────────────
            if pos.bars_held >= MAX_HOLD_BARS:
                to_close.append((pos, "time_exit"))
                continue

            # ── EXIT 4: Emergency Stop ───────────────────────
            if pos.entry_price_option > 0:
                option_loss = (
                    (pos.entry_price_option - current_option)
                    / pos.entry_price_option
                )
                if option_loss >= EMERGENCY_STOP_PCT:
                    to_close.append((pos, "emergency_stop"))
                    continue

        # Execute exits
        for pos, reason in to_close:
            try:
                await self.alpaca.sell_option(
                    pos.option_symbol,
                    pos.qty,
                    model_name=MODEL_NAME,
                    reason=reason,
                )
                exit_record = {
                    "timestamp": datetime.now().isoformat(),
                    "model": MODEL_NAME,
                    "action": "exit",
                    "reason": reason,
                    "ticker": pos.ticker,
                    "contract": pos.option_symbol,
                    "qty": pos.qty,
                    "bars_held": pos.bars_held,
                    "entry_price_underlying": pos.entry_price_underlying,
                    "entry_price_option": pos.entry_price_option,
                    "peak_option_price": pos.peak_option_price,
                    "order_id": pos.order_id,
                }
                self.exits_today.append(exit_record)
                self.active_positions.remove(pos)

                try:
                    await send_trade_exit_alert(exit_record)
                except Exception as e:
                    logger.warning("exit_email_failed", error=str(e))

                logger.info(
                    "vol_scalp_v1_exit",
                    model=MODEL_NAME,
                    ticker=pos.ticker,
                    reason=reason,
                    bars_held=pos.bars_held,
                )
            except Exception as e:
                logger.error(
                    "vol_scalp_v1_exit_failed",
                    model=MODEL_NAME,
                    ticker=pos.ticker,
                    error=str(e),
                )

    # ── Logging ──────────────────────────────────────────────────

    def log_daily_summary(self) -> None:
        """Write daily summary to disk."""
        summary = {
            "date": date.today().isoformat(),
            "model": MODEL_NAME,
            "tickers": self.tickers,
            "scan_cycles": self._scan_count,
            "signals": self.signals_today,
            "trades": self.trades_today,
            "exits": self.exits_today,
            "total_signals": len(self.signals_today),
            "total_trades": len(self.trades_today),
            "total_exits": len(self.exits_today),
            "remaining_positions": len(self.active_positions),
        }

        log_file = self.log_dir / f"vol_scalp_v1_{date.today().isoformat()}.json"
        log_file.write_text(json.dumps(summary, indent=2, default=str))
        logger.info(
            "vol_scalp_v1_daily_summary",
            model=MODEL_NAME,
            path=str(log_file),
            signals=len(self.signals_today),
            trades=len(self.trades_today),
        )

    # ── Main Loop ────────────────────────────────────────────────

    async def run(self) -> None:
        """Main scanner loop."""
        logger.info(
            "vol_scalp_v1_starting",
            model=MODEL_NAME,
            tickers=self.tickers,
            max_positions=MAX_POSITIONS,
        )

        await self.load_daily_bars()

        while True:
            now = datetime.now(ET_OFFSET)
            current_time = now.time()

            if current_time < MARKET_OPEN or current_time > MARKET_CLOSE:
                if current_time > MARKET_CLOSE and self.signals_today:
                    self.log_daily_summary()
                    self.signals_today = []
                    self.trades_today = []
                    self.exits_today = []
                    self.day_opens = {}
                    self._scan_count = 0

                await asyncio.sleep(60)
                continue

            self._scan_count += 1

            try:
                # 1. Refresh bars
                await self.refresh_bars()

                # 2. Check exits
                if self.active_positions:
                    await self.check_exits()

                # 3. Scan for signals during entry window
                if ENTRY_START <= current_time <= ENTRY_END:
                    signals = self.check_signals()
                    for sig in signals:
                        self.signals_today.append(sig.to_dict())

                    # 4. Execute top conviction signal
                    for signal in sorted(
                        signals,
                        key=lambda s: s.conviction,
                        reverse=True,
                    ):
                        order = await self.execute_signal(signal)
                        if order:
                            logger.info(
                                "vol_scalp_v1_order_placed",
                                model=MODEL_NAME,
                                ticker=signal.ticker,
                                order_id=order.order_id,
                                conviction=signal.conviction,
                                conditions=signal.conditions_met,
                            )

                # 5. Log cycle (every 10th to reduce noise)
                if self._scan_count % 10 == 1:
                    positions = await self.alpaca.get_positions()
                    account = await self.alpaca.get_account()
                    logger.info(
                        "vol_scalp_v1_cycle",
                        model=MODEL_NAME,
                        cycle=self._scan_count,
                        positions=len(positions),
                        equity=account.get("equity"),
                        signals_today=len(self.signals_today),
                        trades_today=len(self.trades_today),
                    )
            except Exception as e:
                logger.error(
                    "vol_scalp_v1_cycle_error",
                    model=MODEL_NAME,
                    cycle=self._scan_count,
                    error=str(e),
                    error_type=type(e).__name__,
                )

            await asyncio.sleep(SIGNAL_SCAN_INTERVAL)


# ── Entry Point ──────────────────────────────────────────────────


def create_scanner(
    polygon: PolygonIntradayProvider,
    alpaca: AlpacaExecutor,
) -> VolumeScalperV1Scanner:
    """Factory: create scanner with pre-built providers (for unified orchestrator)."""
    return VolumeScalperV1Scanner(polygon, alpaca)


async def scanner_main(dry_run: bool = False) -> None:
    """Entry point for the accelerator scanner."""
    load_dotenv()

    polygon_key = os.getenv("POLYGON_API_KEY", "")
    alpaca_key = os.getenv("ALPACA_API_KEY_ID", "")
    alpaca_secret = os.getenv("ALPACA_API_SECRET_KEY", "")

    if not polygon_key or not alpaca_key or not alpaca_secret:
        logger.error(
            "missing_api_keys",
            model=MODEL_NAME,
            hint="Set POLYGON_API_KEY, ALPACA_API_KEY_ID, "
            "ALPACA_API_SECRET_KEY in .env",
        )
        return

    alpaca = AlpacaExecutor(alpaca_key, alpaca_secret, paper=True)

    # WebSocket-first data layer
    from flowedge.config.settings import get_settings
    from flowedge.scanner.data_feeds.ws_bars import WebSocketBarProvider
    from flowedge.scanner.flux.ws_consumer import MassiveDataFeed

    settings = get_settings()
    data_feed = None
    if settings.flux_use_websocket:
        data_feed = MassiveDataFeed(
            api_key=polygon_key,
            tickers=list(VOL_SCALP_TICKERS),
            ws_url=settings.flux_ws_url,
        )
        await data_feed.start()
        polygon = WebSocketBarProvider(data_feed, fallback_api_key=polygon_key)
    else:
        polygon = PolygonIntradayProvider(polygon_key)

    scanner = VolumeScalperV1Scanner(polygon, alpaca)

    data_mode = "WebSocket" if data_feed else "REST"
    print("=" * 65)
    print("FLOWEDGE VOLUME SCALPER v1 — HIGH-FREQUENCY SIGNAL GENERATOR")
    print("=" * 65)
    print(f"Data:     {data_mode} — bars from Massive WebSocket")
    print(f"Model:    {MODEL_NAME} (3-of-5 confluence)")
    print(f"Tickers:  {len(VOL_SCALP_TICKERS)} tickers across sectors")
    print(f"Filters:  IBS<{IBS_THRESHOLD} RSI3<{RSI3_THRESHOLD} "
          f"Vol>{VOL_SPIKE_THRESHOLD}x Drop<{INTRADAY_DROP_THRESHOLD*100:.1f}%")
    print("          + Below VWAP (always required)")
    print("          Need 3 of 5 conditions (vs 7/7 for scalp v2)")
    print(f"Exits:    TP={TP_UNDERLYING*100:.1f}% "
          f"Trail={TRAIL_PCT*100:.0f}% "
          f"MaxHold={MAX_HOLD_BARS}bars "
          f"Stop={EMERGENCY_STOP_PCT*100:.0f}%")
    print(f"Risk:     {EQUITY_FRACTION*100:.0f}% per trade, "
          f"max {MAX_POSITIONS} positions")
    print(f"Scan:     Every {SIGNAL_SCAN_INTERVAL}s, "
          f"entries {ENTRY_START.strftime('%H:%M')}-"
          f"{ENTRY_END.strftime('%H:%M')} ET")
    print("Expected: ~2-5 signals/day, ~50-60% WR")
    mode = "DRY RUN" if dry_run else "Alpaca paper trading"
    print(f"Execute:  {mode}")
    print("=" * 65)

    try:
        account = await alpaca.get_account()
        print(f"\nAlpaca: {account.get('status')} | "
              f"Equity: ${float(account.get('equity', 0)):,.0f} | "
              f"Buying Power: ${float(account.get('buying_power', 0)):,.0f}")
        print("\nVolume Scalper v1 running... (Ctrl+C to stop)\n")

        await scanner.run()
    except KeyboardInterrupt:
        print("\nVolume Scalper v1 stopped.")
    finally:
        scanner.log_daily_summary()
        await polygon.close()
        await alpaca.close()
        if data_feed:
            await data_feed.close()


if __name__ == "__main__":
    asyncio.run(scanner_main())
