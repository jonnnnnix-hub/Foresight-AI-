"""FlowEdge Production Scanner — runs all 3 models during market hours.

Executes every 5 minutes from 9:35 AM to 3:55 PM ET:
1. Pulls latest 5-min bars from Polygon for all tickers
2. Computes intraday features (IBS, RSI3, VWAP, gap, volume)
3. Runs Precision, Hybrid, and Rapid signal checks
4. If signal fires → looks up real options chain bid/ask
5. Executes on Alpaca paper trading
6. Monitors open positions for exit conditions
7. Logs everything to file + console

Usage:
    .venv/bin/python -m flowedge.scanner.live.scanner
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
from flowedge.scanner.data_feeds.feature_engine import (
    build_snapshot,
    check_hybrid_ibs,
    check_rapid_confluence,
)
from flowedge.scanner.data_feeds.polygon_intraday import PolygonIntradayProvider
from flowedge.scanner.data_feeds.schemas import (
    BarData,
    Timeframe,
)
from flowedge.scanner.flux.consumer import PolygonTradeConsumer
from flowedge.scanner.flux.engine import scan_flux_for_snapshot
from flowedge.scanner.flux.schemas import FlowBias, FLUXSignal

logger = structlog.get_logger()

# US Eastern timezone offset (ET = UTC-4 during EDT, UTC-5 during EST)
ET_OFFSET = timezone(timedelta(hours=-4))  # EDT (April)

# Market hours
MARKET_OPEN = time(9, 35)   # 5 min after open to let volatility settle
MARKET_CLOSE = time(15, 55)  # 5 min before close

# Scan interval
SCAN_INTERVAL_SECONDS = 300  # 5 minutes

# Models and their configs
# Tickers match proven backtest configs:
PRECISION_TICKERS = ["SPY"]  # v10.2: 80% WR
HYBRID_TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "META"]  # v7.2: 72.7% WR
RAPID_TICKERS = ["SPY", "QQQ", "XLK", "PLTR"]  # v5.2: 64.1% WR

ALL_TICKERS = sorted(set(
    PRECISION_TICKERS + HYBRID_TICKERS + RAPID_TICKERS
))

# Conviction thresholds
# Conviction thresholds match proven backtest configs:
PRECISION_MIN_CONV = 9.5  # v10.2 uses IBS < 0.10 + gap (fires rarely)
HYBRID_MIN_CONV = 9.5     # v7.2: 9.5+ = 76% WR vs 29% below
RAPID_MIN_CONV = 8.0      # v5.2: 8.0+ = 64% WR

# Position limits
MAX_TOTAL_POSITIONS = 5
MAX_PER_TICKER = 1

# Option selection
OPTION_MIN_DTE = 5
OPTION_MAX_DTE = 21
OPTION_MIN_BID = 0.50  # Don't trade illiquid options
OPTION_MAX_SPREAD_PCT = 8.0  # Max 8% bid-ask spread


class ProductionScanner:
    """Runs all 3 models and executes on Alpaca paper."""

    def __init__(
        self,
        polygon: PolygonIntradayProvider,
        alpaca: AlpacaExecutor,
        flux_consumer: PolygonTradeConsumer | None = None,
        log_dir: str = "data/live_logs",
    ) -> None:
        self.polygon = polygon
        self.alpaca = alpaca
        self.flux_consumer = flux_consumer
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.bars_cache: dict[str, list[BarData]] = {}
        self.daily_bars_cache: dict[str, list[BarData]] = {}
        self.active_orders: list[AlpacaOrder] = []
        self.signals_today: list[dict[str, Any]] = []
        self.trades_today: list[dict[str, Any]] = []
        # Track which model opened each position (option_symbol → model_name)
        self.position_models: dict[str, str] = {}
        # FLUX signals cache (refreshed each scan cycle)
        self.flux_cache: dict[str, FLUXSignal] = {}

    # ── Data Loading ──────────────────────────────────────────────

    async def refresh_bars(self) -> None:
        """Pull latest 5-min bars for all tickers."""
        today = date.today().isoformat()
        for ticker in ALL_TICKERS:
            try:
                bars = await self.polygon.get_intraday_bars(
                    ticker, Timeframe.MIN_5, today, today, limit=500,
                )
                self.bars_cache[ticker] = bars
            except Exception as e:
                logger.warning("bar_fetch_failed", ticker=ticker, error=str(e))

        logger.info(
            "bars_refreshed",
            tickers=len(self.bars_cache),
            total_bars=sum(len(b) for b in self.bars_cache.values()),
        )

    async def load_daily_bars(self) -> None:
        """Load last 30 days of daily bars for regime detection."""
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=60)).isoformat()
        for ticker in ALL_TICKERS:
            try:
                bars = await self.polygon.get_intraday_bars(
                    ticker, Timeframe.DAILY, start, end, limit=60,
                )
                self.daily_bars_cache[ticker] = bars
            except Exception as e:
                logger.warning(
                    "daily_bar_fetch_failed", ticker=ticker, error=str(e),
                )
            await asyncio.sleep(0.5)  # Rate limit courtesy

    # ── FLUX Order Flow ────────────────────────────────────────────

    async def refresh_flux(self) -> None:
        """Run FLUX order flow analysis for all tickers."""
        if not self.flux_consumer:
            return

        for ticker in ALL_TICKERS:
            try:
                # Compute 5-min price change for divergence detection
                bars = self.bars_cache.get(ticker, [])
                price_change = 0.0
                if len(bars) >= 2:
                    price_change = (
                        (bars[-1].close - bars[-2].close) / bars[-2].close
                    )

                flux_signal = await scan_flux_for_snapshot(
                    self.flux_consumer,
                    ticker,
                    price_change_pct=price_change,
                )
                self.flux_cache[ticker] = flux_signal

                if flux_signal.strength >= 5.0:
                    logger.info(
                        "flux_signal",
                        ticker=ticker,
                        strength=flux_signal.strength,
                        bias=flux_signal.bias.value,
                        divergence=flux_signal.divergence.value,
                        blocks=len(flux_signal.block_prints),
                    )
            except Exception as e:
                logger.warning("flux_scan_failed", ticker=ticker, error=str(e))

        logger.info("flux_refreshed", tickers=len(self.flux_cache))

    # ── Signal Scanning ──────────────────────────────────────────

    async def scan_signals(self) -> list[dict[str, Any]]:
        """Run all 3 models against current data."""
        signals: list[dict[str, Any]] = []

        for ticker in ALL_TICKERS:
            bars_5m = self.bars_cache.get(ticker, [])
            bars_daily = self.daily_bars_cache.get(ticker, [])

            if len(bars_5m) < 4:
                continue

            # Build snapshot
            snapshot = build_snapshot(
                ticker,
                bars_5m=bars_5m,
                bars_daily=bars_daily,
            )
            if not snapshot or not snapshot.is_tradeable:
                continue

            # Get FLUX data for this ticker (if available)
            flux = self.flux_cache.get(ticker)
            flux_boost = 0.0
            flux_veto = False
            if flux:
                # Boost conviction if FLUX confirms direction
                if flux.bias in (FlowBias.STRONG_BUY, FlowBias.BUY):
                    flux_boost = 0.5 if flux.strength >= 5.0 else 0.2
                # Veto if FLUX shows strong opposing flow
                elif (
                    flux.bias in (FlowBias.STRONG_SELL, FlowBias.SELL)
                    and flux.strength >= 7.0
                ):
                    flux_veto = True  # Strong sell pressure — skip entry

            if flux_veto:
                logger.info(
                    "flux_veto",
                    ticker=ticker,
                    flux_bias=flux.bias.value if flux else "",
                    flux_strength=flux.strength if flux else 0,
                )
                continue

            # Check Rapid model (highest frequency)
            if ticker in RAPID_TICKERS:
                rapid_sig = check_rapid_confluence(snapshot)
                if rapid_sig and rapid_sig["conviction"] >= RAPID_MIN_CONV:
                    rapid_sig["model"] = "rapid"
                    rapid_sig["conviction"] += flux_boost
                    rapid_sig["flux_strength"] = flux.strength if flux else 0.0
                    rapid_sig["flux_bias"] = flux.bias.value if flux else ""
                    signals.append(rapid_sig)

            # Check Hybrid model
            if ticker in HYBRID_TICKERS:
                hybrid_sig = check_hybrid_ibs(snapshot)
                if hybrid_sig and hybrid_sig["conviction"] >= HYBRID_MIN_CONV:
                    hybrid_sig["model"] = "hybrid"
                    hybrid_sig["conviction"] += flux_boost
                    hybrid_sig["flux_strength"] = flux.strength if flux else 0.0
                    hybrid_sig["flux_bias"] = flux.bias.value if flux else ""
                    signals.append(hybrid_sig)

            # Check Precision model (SPY only, strictest — daily IBS extreme)
            if (
                ticker in PRECISION_TICKERS
                and snapshot.ibs_daily < 0.10
                and snapshot.gap_pct < -0.003
            ):
                precision_sig: dict[str, Any] = {
                    "ticker": ticker,
                    "direction": "bullish",
                    "signal_type": "precision_ibs_extreme",
                    "model": "precision",
                    "conviction": 9.5 + flux_boost,
                    "ibs_daily": snapshot.ibs_daily,
                    "gap_pct": snapshot.gap_pct,
                    "flux_strength": flux.strength if flux else 0.0,
                    "flux_bias": flux.bias.value if flux else "",
                }
                signals.append(precision_sig)

        if signals:
            logger.info(
                "signals_detected",
                count=len(signals),
                models=[s["model"] for s in signals],
                tickers=[s["ticker"] for s in signals],
            )
        return signals

    # ── Execution ────────────────────────────────────────────────

    async def execute_signal(self, signal: dict[str, Any]) -> AlpacaOrder | None:
        """Execute a signal by finding the best option and placing an order."""
        ticker = signal["ticker"]
        direction = signal.get("direction", "bullish")
        model = signal.get("model", "unknown")
        conviction = signal.get("conviction", 0)

        # Check position limits
        positions = await self.alpaca.get_positions()
        if len(positions) >= MAX_TOTAL_POSITIONS:
            logger.info("max_positions_reached", current=len(positions))
            return None

        ticker_positions = [p for p in positions if ticker in p.ticker]
        if len(ticker_positions) >= MAX_PER_TICKER:
            logger.info("ticker_position_exists", ticker=ticker)
            return None

        # Find best option from chain
        account = await self.alpaca.get_account()
        current_price = 0.0
        if self.bars_cache.get(ticker):
            current_price = self.bars_cache[ticker][-1].close

        if current_price <= 0:
            return None

        option_type = "call" if direction == "bullish" else "put"
        option = await self.polygon.get_nearest_atm_option(
            ticker, current_price, option_type,
            min_dte=OPTION_MIN_DTE, max_dte=OPTION_MAX_DTE,
        )

        if not option:
            logger.warning("no_option_found", ticker=ticker, type=option_type)
            return None

        # Validate option quality
        if option.bid < OPTION_MIN_BID:
            logger.info("option_too_cheap", bid=option.bid, ticker=ticker)
            return None
        if option.spread_pct > OPTION_MAX_SPREAD_PCT:
            logger.info(
                "spread_too_wide",
                spread_pct=option.spread_pct,
                ticker=ticker,
            )
            return None

        # Size the position (8% of equity per trade)
        equity = float(account.get("equity", 0))
        budget = equity * 0.08
        premium = option.ask  # Buy at ask
        contracts = max(1, int(budget / (premium * 100)))

        logger.info(
            "executing_signal",
            model=model,
            ticker=ticker,
            direction=direction,
            conviction=conviction,
            contract=option.contract_symbol,
            strike=option.strike,
            expiration=option.expiration,
            bid=option.bid,
            ask=option.ask,
            spread_pct=option.spread_pct,
            delta=option.delta,
            iv=option.iv,
            contracts=contracts,
        )

        # Place the order
        order = await self.alpaca.buy_option(
            symbol=option.contract_symbol,
            qty=contracts,
            order_type="limit",
            limit_price=round(option.ask, 2),
            model_name=model,
            conviction=conviction,
        )

        self.active_orders.append(order)
        # Track which model owns this position
        self.position_models[option.contract_symbol] = model
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "ticker": ticker,
            "direction": direction,
            "conviction": conviction,
            "contract": option.contract_symbol,
            "strike": option.strike,
            "premium": option.ask,
            "contracts": contracts,
            "order_id": order.order_id,
            "status": order.status,
        }
        self.trades_today.append(trade_record)

        # Email alert (non-blocking)
        try:
            await send_trade_entry_alert(trade_record)
        except Exception as e:
            logger.warning("entry_email_failed", error=str(e))

        return order

    # ── Position Monitoring ──────────────────────────────────────

    async def check_exits(self) -> None:
        """Monitor open positions for exit conditions."""
        positions = await self.alpaca.get_positions()

        for pos in positions:
            pnl_pct = pos.unrealized_pnl_pct
            model = self.position_models.get(pos.ticker, "unknown")

            # Take profit: 20% for rapid, 50% for hybrid/precision
            if pnl_pct >= 20.0:
                logger.info(
                    "take_profit_triggered",
                    model=model,
                    ticker=pos.ticker,
                    pnl_pct=pnl_pct,
                )
                await self.alpaca.sell_option(
                    pos.ticker, pos.qty,
                    model_name=model, reason="tp",
                )
                exit_record = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "ticker": pos.ticker,
                    "reason": "take_profit",
                    "bars_held": 0,
                    "entry_price_option": pos.entry_price,
                    "peak_option_price": pos.current_price,
                }
                try:
                    await send_trade_exit_alert(exit_record)
                except Exception as e:
                    logger.warning("exit_email_failed", error=str(e))
                self.position_models.pop(pos.ticker, None)
                continue

            # Emergency stop: -40%
            if pnl_pct <= -40.0:
                logger.warning(
                    "emergency_stop_triggered",
                    model=model,
                    ticker=pos.ticker,
                    pnl_pct=pnl_pct,
                )
                await self.alpaca.sell_option(
                    pos.ticker, pos.qty,
                    model_name=model, reason="stop",
                )
                exit_record = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "ticker": pos.ticker,
                    "reason": "emergency_stop",
                    "bars_held": 0,
                    "entry_price_option": pos.entry_price,
                    "peak_option_price": pos.current_price,
                }
                try:
                    await send_trade_exit_alert(exit_record)
                except Exception as e:
                    logger.warning("exit_email_failed", error=str(e))
                self.position_models.pop(pos.ticker, None)
                continue

    # ── Logging ──────────────────────────────────────────────────

    def log_daily_summary(self) -> None:
        """Write daily summary + run self-learning weight update."""
        summary = {
            "date": date.today().isoformat(),
            "signals": self.signals_today,
            "trades": self.trades_today,
            "total_signals": len(self.signals_today),
            "total_trades": len(self.trades_today),
        }

        log_file = self.log_dir / f"scanner_{date.today().isoformat()}.json"
        log_file.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("daily_summary_saved", path=str(log_file))

        # Self-learning: update weights from today's closed trades
        if self.trades_today:
            from flowedge.scanner.backtest.learning_hook import post_backtest_learn
            post_backtest_learn(
                self.trades_today,
                model_name="live_scanner",
                min_trades=1,  # Update from any closed trade
            )

    # ── Main Loop ────────────────────────────────────────────────

    async def run(self) -> None:
        """Main scanner loop — runs during market hours."""
        logger.info("scanner_starting", tickers=ALL_TICKERS)

        # Load daily bars once at startup
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

                # Wait until next market open
                wait = 60
                logger.info("market_closed", next_check_in=f"{wait}s")
                await asyncio.sleep(wait)
                continue

            # Market is open — scan
            logger.info(
                "scan_cycle_start",
                time=current_time.isoformat(),
            )

            try:
                # 1. Refresh bars
                await self.refresh_bars()

                # 1b. Refresh FLUX order flow
                await self.refresh_flux()

                # 2. Check exits on open positions
                await self.check_exits()

                # 3. Scan for new signals (FLUX now informs conviction)
                signals = await self.scan_signals()
                self.signals_today.extend(signals)

                # 4. Execute top signals
                for signal in sorted(
                    signals,
                    key=lambda s: s.get("conviction", 0),
                    reverse=True,
                ):
                    order = await self.execute_signal(signal)
                    if order:
                        logger.info(
                            "order_placed",
                            model=signal["model"],
                            ticker=signal["ticker"],
                            order_id=order.order_id,
                        )

                # 5. Log current state
                positions = await self.alpaca.get_positions()
                account = await self.alpaca.get_account()
                logger.info(
                    "scan_cycle_complete",
                    signals=len(signals),
                    positions=len(positions),
                    equity=account.get("equity"),
                    signals_today=len(self.signals_today),
                    trades_today=len(self.trades_today),
                )
            except Exception as e:
                logger.error(
                    "scan_cycle_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Don't crash — wait and retry next cycle

            # Wait for next scan
            await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def main() -> None:
    """Entry point for the production scanner."""
    load_dotenv()

    polygon_key = os.getenv("POLYGON_API_KEY", "")
    alpaca_key = os.getenv("ALPACA_API_KEY_ID", "")
    alpaca_secret = os.getenv("ALPACA_API_SECRET_KEY", "")

    if not polygon_key or not alpaca_key or not alpaca_secret:
        logger.error(
            "missing_api_keys",
            hint="Set POLYGON_API_KEY, ALPACA_API_KEY_ID, "
            "ALPACA_API_SECRET_KEY in .env",
        )
        return

    polygon = PolygonIntradayProvider(polygon_key)
    alpaca = AlpacaExecutor(alpaca_key, alpaca_secret, paper=True)
    flux_consumer = PolygonTradeConsumer(polygon_key)

    scanner = ProductionScanner(polygon, alpaca, flux_consumer=flux_consumer)

    print("=" * 65)
    print("FLOWEDGE PRODUCTION SCANNER")
    print("=" * 65)
    print("Models:  Precision (SPY) | Hybrid (7 tickers) | Rapid (5 tickers) | FLUX (all)")
    print("FLUX:    Lee-Ready tape reading + L1 quote imbalance + block detection")
    print(f"Tickers: {ALL_TICKERS}")
    print(f"Scan:    Every {SCAN_INTERVAL_SECONDS}s during market hours (9:35-15:55 ET)")
    print("Execute: Alpaca paper trading ($100K account)")
    print("=" * 65)

    try:
        # Verify connection
        account = await alpaca.get_account()
        print(f"\nAlpaca: {account.get('status')} | "
              f"Equity: ${float(account.get('equity', 0)):,.0f} | "
              f"Buying Power: ${float(account.get('buying_power', 0)):,.0f}")
        print("\nScanner running... (Ctrl+C to stop)\n")

        await scanner.run()
    except KeyboardInterrupt:
        print("\nScanner stopped.")
    finally:
        scanner.log_daily_summary()
        await polygon.close()
        await alpaca.close()
        await flux_consumer.close()


if __name__ == "__main__":
    asyncio.run(main())
