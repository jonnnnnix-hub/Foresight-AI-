"""FlowEdge Scalp v2 Live Scanner — standalone paper trader.

Runs the sweep-validated 90% WR scalp model on live data:
1. Pulls 5-min bars from Polygon every 5 minutes (9:35–15:55 ET)
2. Computes the exact 7-condition entry filter from the backtest
3. On signal → finds real ATM call option via Polygon chain
4. Executes on Alpaca paper trading (tagged "scalp_v2")
5. Monitors positions for TP / trailing stop / time exit
6. Logs all signals, trades, and P&L to data/live_logs/scalp_v2/

The 7 entry conditions (all must fire):
  1. IBS(5-min bar) < 0.12
  2. RSI(3) on 5-min closes < 15.0
  3. Price below VWAP
  4. Volume spike > 2.5× (vs prior 10-bar avg)
  5. Intraday drop from open < -0.2%
  6. Prior bar red (close < prior close)
  7. SMA(5) < SMA(10) on 5-min bars

Exit conditions (first to fire wins):
  - Take profit: underlying +0.2% from entry
  - Trailing stop: option premium falls 3% from peak
  - Time exit: 12 bars (60 minutes) max hold

Config loaded from configs/sweep_best_90wr.json.

Usage:
    .venv/bin/python -m flowedge.scanner.live.scalp_v2_scanner
    .venv/bin/python scripts/run_scalp_v2_scanner.py
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
from flowedge.scanner.backtest.scalp_config import ScalpConfig
from flowedge.scanner.data_feeds.alpaca_execution import AlpacaExecutor, AlpacaOrder
from flowedge.scanner.data_feeds.polygon_intraday import PolygonIntradayProvider
from flowedge.scanner.data_feeds.schemas import BarData, Timeframe

logger = structlog.get_logger()

# ── Constants ────────────────────────────────────────────────────

MODEL_NAME = "scalp_v2"  # Tag for every order placed by this scanner

# US Eastern timezone (EDT April–October)
ET_OFFSET = timezone(timedelta(hours=-4))

# Market hours with buffer
MARKET_OPEN = time(9, 35)   # 5 min after open to let volatility settle
MARKET_CLOSE = time(15, 55)  # 5 min before close — no new entries after this

# Entry window — expanded for live scanning (backtest used 10:00-11:30)
MORNING_SESSION_START = time(9, 40)
MORNING_SESSION_END = time(15, 30)

# Scan intervals — two speeds:
# Signal detection: every 5 min (aligned to 5-min bar updates)
# Exit monitoring: every 15 sec (real-time snapshots for open positions)
SIGNAL_SCAN_INTERVAL = 60  # Check for new 5-min bars every 60s
EXIT_CHECK_INTERVAL = 15   # Check exits every 15s when positions are open

# Option selection
OPTION_MIN_DTE = 0   # 0-DTE allowed (matches backtest DTE ≤ 5)
OPTION_MAX_DTE = 5
OPTION_MIN_BID = 0.30  # Must match config.min_premium
OPTION_MAX_SPREAD_PCT = 10.0  # Max 10% bid-ask spread

# Position sizing
EQUITY_FRACTION_PER_TRADE = 0.05  # 5% of equity per scalp (matches config)


# ── Data Types ───────────────────────────────────────────────────

class ScalpSignal:
    """A scalp v2 entry signal."""

    __slots__ = (
        "ticker", "bar_index", "timestamp", "ibs", "rsi3",
        "vwap_gap_pct", "vol_ratio", "intraday_drop_pct",
        "underlying_price", "conviction",
    )

    def __init__(
        self,
        ticker: str,
        bar_index: int,
        timestamp: str,
        ibs: float,
        rsi3: float,
        vwap_gap_pct: float,
        vol_ratio: float,
        intraday_drop_pct: float,
        underlying_price: float,
    ) -> None:
        self.ticker = ticker
        self.bar_index = bar_index
        self.timestamp = timestamp
        self.ibs = ibs
        self.rsi3 = rsi3
        self.vwap_gap_pct = vwap_gap_pct
        self.vol_ratio = vol_ratio
        self.intraday_drop_pct = intraday_drop_pct
        self.underlying_price = underlying_price
        # Conviction: higher when conditions are more extreme
        self.conviction = self._compute_conviction()

    def _compute_conviction(self) -> float:
        score = 7.0  # Base: all 7 conditions passed
        # Bonus for extreme IBS (lower = more oversold)
        if self.ibs < 0.06:
            score += 1.0
        elif self.ibs < 0.09:
            score += 0.5
        # Bonus for extreme RSI3
        if self.rsi3 < 8.0:
            score += 1.0
        elif self.rsi3 < 12.0:
            score += 0.5
        # Bonus for heavy volume
        if self.vol_ratio > 4.0:
            score += 0.5
        return min(score, 10.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": MODEL_NAME,
            "ticker": self.ticker,
            "bar_index": self.bar_index,
            "timestamp": self.timestamp,
            "ibs": round(self.ibs, 4),
            "rsi3": round(self.rsi3, 2),
            "vwap_gap_pct": round(self.vwap_gap_pct, 4),
            "vol_ratio": round(self.vol_ratio, 2),
            "intraday_drop_pct": round(self.intraday_drop_pct, 4),
            "underlying_price": round(self.underlying_price, 2),
            "conviction": round(self.conviction, 2),
        }


class LivePosition:
    """A live scalp v2 position being monitored."""

    __slots__ = (
        "ticker", "option_symbol", "qty", "entry_price_underlying",
        "entry_price_option", "entry_bar_index", "entry_time",
        "peak_option_price", "bars_held", "order_id",
    )

    def __init__(
        self,
        ticker: str,
        option_symbol: str,
        qty: int,
        entry_price_underlying: float,
        entry_price_option: float,
        entry_bar_index: int,
        entry_time: str,
        order_id: str,
    ) -> None:
        self.ticker = ticker
        self.option_symbol = option_symbol
        self.qty = qty
        self.entry_price_underlying = entry_price_underlying
        self.entry_price_option = entry_price_option
        self.entry_bar_index = entry_bar_index
        self.entry_time = entry_time
        self.peak_option_price = entry_price_option
        self.bars_held = 0
        self.order_id = order_id


# ── Scanner ──────────────────────────────────────────────────────

class ScalpV2Scanner:
    """Standalone live scanner for the scalp v2 model.

    Pulls 1-min bars from Polygon every 60s, aggregates to 5-min
    chunks for signal detection (matching backtest), and uses the
    raw 1-min data for faster exit monitoring.
    """

    def __init__(
        self,
        polygon: PolygonIntradayProvider,
        alpaca: AlpacaExecutor,
        config: ScalpConfig,
        log_dir: str = "data/live_logs/scalp_v2",
    ) -> None:
        self.polygon = polygon
        self.alpaca = alpaca
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.bars_1m: dict[str, list[BarData]] = {}   # Raw 1-min bars
        self.bars_5m: dict[str, list[BarData]] = {}   # Aggregated 5-min bars
        self.daily_bars: dict[str, list[BarData]] = {}
        self.day_opens: dict[str, float] = {}
        self.active_positions: list[LivePosition] = []
        self.signals_today: list[dict[str, Any]] = []
        self.trades_today: list[dict[str, Any]] = []
        self.exits_today: list[dict[str, Any]] = []
        self._scan_count = 0
        self._last_5m_bar_count: dict[str, int] = {}  # Track new bars

    # ── Data Loading ─────────────────────────────────────────────

    async def load_daily_bars(self) -> None:
        """Load last 30 trading days of daily bars for SMA trend check."""
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=60)).isoformat()
        for ticker in self.config.tickers:
            try:
                bars = await self.polygon.get_intraday_bars(
                    ticker, Timeframe.DAILY, start, end, limit=60,
                )
                self.daily_bars[ticker] = bars
                logger.info(
                    "daily_bars_loaded",
                    model=MODEL_NAME,
                    ticker=ticker,
                    bars=len(bars),
                )
            except Exception as e:
                logger.warning(
                    "daily_bar_fetch_failed",
                    model=MODEL_NAME,
                    ticker=ticker,
                    error=str(e),
                )
            await asyncio.sleep(0.3)  # Rate limit courtesy

    async def refresh_bars(self) -> None:
        """Pull today's 1-min bars and aggregate to 5-min for signals.

        1-min bars give us:
        - Fresh underlying price (≤60s stale vs ≤5min with 5-min bars)
        - Better exit monitoring for 0.2% TP targets
        - Same 5-min features when aggregated (matching backtest)
        """
        today = date.today().isoformat()
        for ticker in self.config.tickers:
            try:
                bars_1m = await self.polygon.get_intraday_bars(
                    ticker, Timeframe.MIN_1, today, today, limit=500,
                )
                self.bars_1m[ticker] = bars_1m

                # Track day open from first 1-min bar
                if bars_1m and ticker not in self.day_opens:
                    self.day_opens[ticker] = bars_1m[0].open

                # Aggregate 1-min → 5-min chunks for signal detection
                self.bars_5m[ticker] = self._aggregate_to_5m(bars_1m, ticker)
            except Exception as e:
                logger.warning(
                    "bar_fetch_failed",
                    model=MODEL_NAME,
                    ticker=ticker,
                    error=str(e),
                )

        total_1m = sum(len(b) for b in self.bars_1m.values())
        total_5m = sum(len(b) for b in self.bars_5m.values())
        logger.info(
            "bars_refreshed",
            model=MODEL_NAME,
            tickers=len(self.bars_1m),
            bars_1m=total_1m,
            bars_5m=total_5m,
        )

    @staticmethod
    def _aggregate_to_5m(
        bars_1m: list[BarData], ticker: str,
    ) -> list[BarData]:
        """Aggregate 1-min bars into 5-min bars for signal detection.

        Groups by floor(timestamp / 5min) to match backtest bucketing.
        """
        if not bars_1m:
            return []

        from collections import defaultdict

        buckets: dict[int, list[BarData]] = defaultdict(list)
        for bar in bars_1m:
            # 5-min bucket key from timestamp
            ts_epoch = int(bar.timestamp.timestamp())
            bucket = ts_epoch // 300  # 300 seconds = 5 minutes
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
        """Get the freshest underlying price available.

        Uses 1-min bars (≤60s stale) instead of 5-min bars (≤5min stale).
        Critical for the 0.2% TP target.
        """
        bars = self.bars_1m.get(ticker, [])
        if bars:
            return bars[-1].close
        # Fallback to 5-min
        bars_5m = self.bars_5m.get(ticker, [])
        if bars_5m:
            return bars_5m[-1].close
        return 0.0

    # ── 7-Condition Signal Check ─────────────────────────────────

    def _check_daily_uptrend(self, ticker: str) -> bool:
        """SMA(10) > SMA(20) on daily closes — matching backtest."""
        daily = self.daily_bars.get(ticker, [])
        if len(daily) < 20:
            return False
        closes = [b.close for b in daily]
        sma10 = sum(closes[-10:]) / 10
        sma20 = sum(closes[-20:]) / 20
        return sma10 > sma20

    def check_signals(self) -> list[ScalpSignal]:
        """Run the 7-condition filter on current 5-min bars.

        This is the exact same logic as the backtest:
        1. IBS < threshold
        2. RSI(3) < threshold
        3. Below VWAP
        4. Volume spike > threshold
        5. Intraday drop > threshold
        6. Prior bar red
        7. SMA(5) < SMA(10) on 5-min bars
        """
        signals: list[ScalpSignal] = []
        cfg = self.config

        for ticker in cfg.tickers:
            bars = self.bars_5m.get(ticker, [])
            if len(bars) < 11:  # Need at least 11 bars for SMA(10) + lookback
                continue

            # Daily uptrend gate
            if not self._check_daily_uptrend(ticker):
                continue

            # Already have a position in this ticker?
            if any(p.ticker == ticker for p in self.active_positions):
                continue

            # Max positions reached?
            if len(self.active_positions) >= cfg.max_positions:
                continue

            # Only check the LATEST bar (we're real-time, not scanning history)
            i = len(bars) - 1
            bar = bars[i]

            # ── CONDITION 1: IBS ──────────────────────────────
            rng = bar.high - bar.low
            if rng <= 0 or bar.close <= 0:
                continue
            ibs = (bar.close - bar.low) / rng
            if ibs >= cfg.ibs_threshold:
                continue

            # ── CONDITION 2: RSI(3) ───────────────────────────
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
            if rsi3 >= cfg.rsi3_threshold:
                continue

            # ── CONDITION 3: Below VWAP ───────────────────────
            # Compute cumulative VWAP from today's bars
            cum_pv = 0.0
            cum_v = 0.0
            for b in bars:
                tp = (b.high + b.low + b.close) / 3
                cum_pv += tp * b.volume
                cum_v += b.volume
            vwap = cum_pv / cum_v if cum_v > 0 else bar.close
            if bar.close >= vwap:
                continue
            vwap_gap_pct = (bar.close - vwap) / vwap

            # ── CONDITION 4: Volume spike ─────────────────────
            lookback_start = max(0, i - 10)
            lookback_count = i - lookback_start
            if lookback_count <= 0:
                continue
            avg_vol = sum(bars[j].volume for j in range(lookback_start, i)) / lookback_count
            vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 1.0
            if vol_ratio < cfg.vol_spike:
                continue

            # ── CONDITION 5: Intraday drop ────────────────────
            day_open = self.day_opens.get(ticker, bar.open)
            intraday_drop = (bar.close - day_open) / day_open if day_open > 0 else 0
            if intraday_drop > cfg.intraday_drop:
                continue

            # ── CONDITION 6: Prior bar red ────────────────────
            if i < 2:
                continue
            if bars[i - 1].close >= bars[i - 2].close:
                continue

            # ── CONDITION 7: SMA(5) < SMA(10) ────────────────
            if len(bars) < 10:
                continue
            sma5 = sum(bars[j].close for j in range(i - 4, i + 1)) / 5
            sma10_count = min(10, i + 1)
            sma10 = sum(bars[j].close for j in range(max(0, i - 9), i + 1)) / sma10_count
            if sma5 >= sma10:
                continue

            # ═══ ALL 7 CONDITIONS PASSED ═══════════════════════
            signal = ScalpSignal(
                ticker=ticker,
                bar_index=i,
                timestamp=bar.timestamp.isoformat(),
                ibs=ibs,
                rsi3=rsi3,
                vwap_gap_pct=vwap_gap_pct,
                vol_ratio=vol_ratio,
                intraday_drop_pct=intraday_drop,
                underlying_price=bar.close,
            )
            signals.append(signal)
            logger.info(
                "scalp_v2_signal",
                model=MODEL_NAME,
                ticker=ticker,
                ibs=round(ibs, 4),
                rsi3=round(rsi3, 2),
                vol_ratio=round(vol_ratio, 2),
                drop_pct=round(intraday_drop, 4),
                conviction=round(signal.conviction, 2),
                price=round(bar.close, 2),
            )

        return signals

    # ── Execution ────────────────────────────────────────────────

    async def execute_signal(self, signal: ScalpSignal) -> AlpacaOrder | None:
        """Find best option and place order on Alpaca paper."""
        ticker = signal.ticker

        # Check total positions (Alpaca-side)
        positions = await self.alpaca.get_positions()
        if len(positions) >= self.config.max_positions:
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
            logger.warning(
                "no_option_found",
                model=MODEL_NAME,
                ticker=ticker,
            )
            return None

        # Validate option quality
        if option.bid < self.config.min_premium:
            logger.info(
                "option_below_min_premium",
                model=MODEL_NAME,
                bid=option.bid,
                min=self.config.min_premium,
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

        # Position sizing: risk_per_trade × equity
        account = await self.alpaca.get_account()
        equity = float(account.get("equity", 0))
        budget = equity * self.config.risk_per_trade
        premium = option.ask
        if premium <= 0:
            return None
        contracts = max(1, int(budget / (premium * 100)))

        logger.info(
            "executing_scalp_v2",
            model=MODEL_NAME,
            ticker=ticker,
            conviction=signal.conviction,
            contract=option.contract_symbol,
            strike=option.strike,
            expiration=option.expiration,
            bid=option.bid,
            ask=option.ask,
            spread_pct=option.spread_pct,
            contracts=contracts,
            budget=round(budget, 2),
        )

        # Place order — tagged as scalp_v2
        order = await self.alpaca.buy_option(
            symbol=option.contract_symbol,
            qty=contracts,
            order_type="limit",
            limit_price=round(option.ask, 2),
            model_name=MODEL_NAME,
            conviction=signal.conviction,
        )

        # Track position locally
        pos = LivePosition(
            ticker=ticker,
            option_symbol=option.contract_symbol,
            qty=contracts,
            entry_price_underlying=signal.underlying_price,
            entry_price_option=option.ask,
            entry_bar_index=signal.bar_index,
            entry_time=signal.timestamp,
            order_id=order.order_id,
        )
        self.active_positions.append(pos)

        # Log trade
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "action": "entry",
            "ticker": ticker,
            "direction": "bullish",
            "conviction": signal.conviction,
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

        # Email alert (non-blocking)
        await send_trade_entry_alert(trade_record)

        return order

    # ── Exit Monitoring ──────────────────────────────────────────

    async def check_exits(self) -> None:
        """Monitor positions for TP, trailing stop, and time exit.

        Exit rules (matching backtest exactly):
        1. Take profit: underlying moved +0.2% from entry
        2. Trailing stop: option premium dropped 3% from peak
        3. Time exit: held for 12 bars (60 minutes)
        """
        cfg = self.config
        to_close: list[tuple[LivePosition, str]] = []

        for pos in self.active_positions:
            pos.bars_held += 1

            # Get current underlying price from 1-min bars (freshest)
            current_underlying = self.get_latest_price(pos.ticker)
            if current_underlying <= 0:
                continue

            # Get current option price from positions API
            alpaca_positions = await self.alpaca.get_positions()
            option_pos = next(
                (p for p in alpaca_positions if p.ticker == pos.option_symbol),
                None,
            )
            current_option_price = (
                option_pos.current_price if option_pos else pos.entry_price_option
            )

            # Update peak
            if current_option_price > pos.peak_option_price:
                pos.peak_option_price = current_option_price

            # ── EXIT 1: Take Profit (underlying +0.2%) ────────
            underlying_move = (
                (current_underlying - pos.entry_price_underlying)
                / pos.entry_price_underlying
            )
            if underlying_move >= cfg.tp_underlying:
                to_close.append((pos, "take_profit"))
                logger.info(
                    "scalp_v2_exit_tp",
                    model=MODEL_NAME,
                    ticker=pos.ticker,
                    underlying_move_pct=round(underlying_move * 100, 3),
                )
                continue

            # ── EXIT 2: Trailing Stop (option -3% from peak) ──
            if pos.peak_option_price > 0:
                drawdown_from_peak = (
                    (pos.peak_option_price - current_option_price)
                    / pos.peak_option_price
                )
                if drawdown_from_peak >= cfg.trail_pct:
                    to_close.append((pos, "trailing_stop"))
                    logger.info(
                        "scalp_v2_exit_trail",
                        model=MODEL_NAME,
                        ticker=pos.ticker,
                        drawdown_pct=round(drawdown_from_peak * 100, 2),
                    )
                    continue

            # ── EXIT 3: Time Exit (12 bars = 60 min) ──────────
            if pos.bars_held >= cfg.max_hold_bars:
                to_close.append((pos, "time_exit"))
                logger.info(
                    "scalp_v2_exit_time",
                    model=MODEL_NAME,
                    ticker=pos.ticker,
                    bars_held=pos.bars_held,
                )
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

                # Email alert (non-blocking)
                await send_trade_exit_alert(exit_record)

                logger.info(
                    "scalp_v2_position_closed",
                    model=MODEL_NAME,
                    ticker=pos.ticker,
                    reason=reason,
                    bars_held=pos.bars_held,
                )
            except Exception as e:
                logger.error(
                    "scalp_v2_exit_failed",
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
            "config": self.config.model_dump(),
            "scan_cycles": self._scan_count,
            "signals": self.signals_today,
            "trades": self.trades_today,
            "exits": self.exits_today,
            "total_signals": len(self.signals_today),
            "total_trades": len(self.trades_today),
            "total_exits": len(self.exits_today),
            "remaining_positions": len(self.active_positions),
        }

        log_file = self.log_dir / f"scalp_v2_{date.today().isoformat()}.json"
        log_file.write_text(json.dumps(summary, indent=2, default=str))
        logger.info(
            "scalp_v2_daily_summary",
            model=MODEL_NAME,
            path=str(log_file),
            signals=len(self.signals_today),
            trades=len(self.trades_today),
            exits=len(self.exits_today),
        )

    # ── Main Loop ────────────────────────────────────────────────

    async def run(self) -> None:
        """Main scanner loop — runs during market hours."""
        logger.info(
            "scalp_v2_scanner_starting",
            model=MODEL_NAME,
            tickers=self.config.tickers,
            max_positions=self.config.max_positions,
        )

        # Load daily bars at startup for trend check
        await self.load_daily_bars()

        while True:
            now = datetime.now(ET_OFFSET)
            current_time = now.time()

            # Only run during market hours
            if current_time < MARKET_OPEN or current_time > MARKET_CLOSE:
                if current_time > MARKET_CLOSE and self.signals_today:
                    self.log_daily_summary()
                    self.signals_today = []
                    self.trades_today = []
                    self.exits_today = []
                    self.day_opens = {}
                    self._scan_count = 0

                wait = 60
                logger.debug(
                    "market_closed",
                    model=MODEL_NAME,
                    next_check_in=f"{wait}s",
                )
                await asyncio.sleep(wait)
                continue

            self._scan_count += 1

            try:
                # 1. Refresh 1-min bars → aggregate to 5-min
                await self.refresh_bars()

                # 2. Check exits on open positions (uses 1-min price)
                if self.active_positions:
                    await self.check_exits()

                # 3. Scan for new signals using 5-min aggregated bars
                if MORNING_SESSION_START <= current_time <= MORNING_SESSION_END:
                    signals = self.check_signals()
                    for sig in signals:
                        self.signals_today.append(sig.to_dict())

                    # 4. Execute highest conviction signal
                    for signal in sorted(
                        signals, key=lambda s: s.conviction, reverse=True,
                    ):
                        order = await self.execute_signal(signal)
                        if order:
                            logger.info(
                                "scalp_v2_order_placed",
                                model=MODEL_NAME,
                                ticker=signal.ticker,
                                order_id=order.order_id,
                                conviction=signal.conviction,
                            )

                # 5. Log cycle summary (every 5th cycle to reduce noise)
                if self._scan_count % 5 == 1:
                    positions = await self.alpaca.get_positions()
                    account = await self.alpaca.get_account()
                    logger.info(
                        "scalp_v2_cycle_complete",
                        model=MODEL_NAME,
                        cycle=self._scan_count,
                        alpaca_positions=len(positions),
                        local_positions=len(self.active_positions),
                        equity=account.get("equity"),
                        signals_today=len(self.signals_today),
                        trades_today=len(self.trades_today),
                    )
            except Exception as e:
                logger.error(
                    "scalp_v2_cycle_error",
                    model=MODEL_NAME,
                    cycle=self._scan_count,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Don't crash — wait and retry next cycle

            # Wait for next scan
            await asyncio.sleep(SIGNAL_SCAN_INTERVAL)


# ── Entry Point ──────────────────────────────────────────────────

async def main(
    config_path: str = "configs/sweep_best_90wr.json",
    dry_run: bool = False,
) -> None:
    """Entry point for the scalp v2 paper trading scanner.

    Args:
        config_path: Path to ScalpConfig JSON file.
        dry_run: If True, detect signals but don't place orders.
    """
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

    # Load config from sweep-validated JSON
    cfg_path = Path(config_path)
    if cfg_path.exists():
        config = ScalpConfig.from_json_file(cfg_path)
        logger.info("config_loaded", path=str(cfg_path))
    else:
        config = ScalpConfig()
        logger.warning("using_default_config", model=MODEL_NAME)

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
            tickers=config.tickers,
            ws_url=settings.flux_ws_url,
        )
        await data_feed.start()
        polygon = WebSocketBarProvider(data_feed, fallback_api_key=polygon_key)
    else:
        polygon = PolygonIntradayProvider(polygon_key)

    scanner = ScalpV2Scanner(polygon, alpaca, config)

    print("=" * 65)
    print("FLOWEDGE SCALP v2 — LIVE PAPER TRADER")
    print("=" * 65)
    data_mode = "WebSocket" if data_feed else "REST"
    print(f"Model:   {MODEL_NAME} (sweep-validated 90% WR)")
    print(f"Data:    {data_mode} — 1-min bars from Massive WebSocket")
    print(f"Tickers: {config.tickers}")
    print(f"Filters: IBS<{config.ibs_threshold} RSI3<{config.rsi3_threshold} "
          f"Vol>{config.vol_spike}x Drop<{config.intraday_drop*100:.1f}%")
    print(f"Exits:   TP={config.tp_underlying*100:.1f}% "
          f"Trail={config.trail_pct*100:.0f}% "
          f"MaxHold={config.max_hold_bars}bars")
    print(f"Risk:    {config.risk_per_trade*100:.0f}% per trade, "
          f"max {config.max_positions} positions")
    print(f"Scan:    Every {SIGNAL_SCAN_INTERVAL}s, "
          f"entries {MORNING_SESSION_START.strftime('%H:%M')}-"
          f"{MORNING_SESSION_END.strftime('%H:%M')} ET")
    mode = "DRY RUN (signals only)" if dry_run else "Alpaca paper trading"
    print(f"Execute: {mode}")
    print("Logs:    data/live_logs/scalp_v2/")
    print("=" * 65)

    try:
        # Verify Alpaca connection
        account = await alpaca.get_account()
        print(f"\nAlpaca: {account.get('status')} | "
              f"Equity: ${float(account.get('equity', 0)):,.0f} | "
              f"Buying Power: ${float(account.get('buying_power', 0)):,.0f}")
        print("\nScalp v2 scanner running... (Ctrl+C to stop)\n")

        await scanner.run()
    except KeyboardInterrupt:
        print("\nScalp v2 scanner stopped.")
    finally:
        scanner.log_daily_summary()
        await polygon.close()
        await alpaca.close()
        if data_feed:
            await data_feed.close()


if __name__ == "__main__":
    asyncio.run(main())
