"""Trident 🔱 — Live ETF 0DTE options scalper.

SPY / QQQ / IWM, both calls and puts, 0-90 min hold windows,
same-day only. Signal-optimized via combinatorial backtest.

Architecture mirrors scalp_v2_scanner.py:
  - Pulls 1-min bars from Polygon every 60s
  - Aggregates to 5-min for signal detection
  - Uses 1-min for exit monitoring (fresher prices)
  - Executes via Alpaca paper trading
  - Tags orders with 'trident_scalp' in client_order_id
  - Sends email alerts on entry/exit
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flowedge.notifications.email_alerts import (
    send_trade_entry_alert,
    send_trade_exit_alert,
)
from flowedge.scanner.backtest.trident.config import (
    Direction,
    TridentConfig,
)
from flowedge.scanner.backtest.trident.signals import (
    Bar,
    compute_all_signals,
    evaluate_signals,
)
from flowedge.scanner.data_feeds.alpaca_execution import AlpacaExecutor
from flowedge.scanner.data_feeds.polygon_intraday import PolygonIntradayProvider

logger = logging.getLogger("trident.live")

MODEL_NAME = "trident_scalp"
TRIDENT_TICKERS = ["SPY", "QQQ", "IWM"]

# ── Data directory ────────────────────────────────────────────────
LOG_DIR = Path("data/live_logs/trident")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Position tracking ────────────────────────────────────────────

@dataclass
class TridentPosition:
    """Tracks an active Trident options position."""

    ticker: str
    is_call: bool
    option_symbol: str
    entry_time: datetime
    entry_underlying: float
    entry_option_price: float
    contracts: int
    conviction: float
    signals_fired: int
    peak_option_price: float
    bars_held: int = 0
    order_id: str = ""


# ── Scanner ──────────────────────────────────────────────────────

class TridentScanner:
    """Live Trident scanner — detects signals and executes trades."""

    def __init__(
        self,
        config: TridentConfig | None = None,
        dry_run: bool = False,
    ) -> None:
        self.cfg = config or TridentConfig()
        self.dry_run = dry_run

        # Load credentials from env
        self._polygon_key = os.getenv("POLYGON_API_KEY", "")
        self._alpaca_key = os.getenv("TRIDENT_ALPACA_KEY_ID", "")
        self._alpaca_secret = os.getenv("TRIDENT_ALPACA_SECRET_KEY", "")

        if not self._polygon_key:
            raise ValueError("POLYGON_API_KEY required for live data")
        if not self._alpaca_key and not dry_run:
            raise ValueError(
                "TRIDENT_ALPACA_KEY_ID / TRIDENT_ALPACA_SECRET_KEY required"
            )

        # WebSocket-first data layer
        from flowedge.config.settings import get_settings
        settings = get_settings()
        self._data_feed = None
        self.polygon: Any = None  # WebSocketBarProvider | PolygonIntradayProvider
        if settings.flux_use_websocket:
            from flowedge.scanner.data_feeds.ws_bars import WebSocketBarProvider
            from flowedge.scanner.flux.ws_consumer import MassiveDataFeed
            self._data_feed = MassiveDataFeed(
                api_key=self._polygon_key,
                tickers=list(TRIDENT_TICKERS),
                ws_url=settings.flux_ws_url,
            )
            self.polygon = WebSocketBarProvider(
                self._data_feed, fallback_api_key=self._polygon_key,
            )
        else:
            self.polygon = PolygonIntradayProvider(self._polygon_key)
        self.alpaca: AlpacaExecutor | None = None
        if not dry_run:
            self.alpaca = AlpacaExecutor(
                self._alpaca_key, self._alpaca_secret, paper=True,
            )

        # State
        self.positions: dict[str, TridentPosition] = {}
        self.daily_bars_1m: dict[str, list[dict[str, Any]]] = {
            t: [] for t in TRIDENT_TICKERS
        }
        self.daily_bars_5m: dict[str, list[Bar]] = {
            t: [] for t in TRIDENT_TICKERS
        }
        self.daily_closes: dict[str, list[float]] = {}
        self.trade_count = 0
        self.last_trade_bar: dict[str, int] = {t: -999 for t in TRIDENT_TICKERS}

        # Log file
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        self.log_path = LOG_DIR / f"trident_{today}.jsonl"

    # ── Data loading ──────────────────────────────────────────

    async def load_daily_bars(self) -> None:
        """Load recent daily bars for trend detection."""
        for ticker in TRIDENT_TICKERS:
            try:
                bars = await self.polygon.get_intraday_bars(
                    ticker, "1", "day",
                    (datetime.now(UTC)).strftime("%Y-%m-%d"),
                    days_back=60,
                )
                self.daily_closes[ticker] = [
                    float(b.get("c", 0)) for b in bars if float(b.get("c", 0)) > 0
                ]
                logger.info(
                    "%s: loaded %d daily bars", ticker,
                    len(self.daily_closes.get(ticker, [])),
                )
            except Exception as exc:
                logger.warning("Failed to load daily bars for %s: %s", ticker, exc)
                self.daily_closes[ticker] = []

    async def refresh_bars(self) -> None:
        """Pull today's 1-min bars from Polygon and aggregate to 5-min."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        for ticker in TRIDENT_TICKERS:
            try:
                bars = await self.polygon.get_intraday_bars(
                    ticker, "1", "minute", today,
                )
                if not bars:
                    continue
                self.daily_bars_1m[ticker] = bars
                self.daily_bars_5m[ticker] = self._aggregate_to_5m(bars)
            except Exception as exc:
                logger.warning("Failed to refresh %s: %s", ticker, exc)

    def _aggregate_to_5m(self, bars_1m: list[dict[str, Any]]) -> list[Bar]:
        """Aggregate 1-min bars to 5-min bars."""
        if not bars_1m:
            return []
        window_ns = 5 * 60 * 1_000_000_000
        buckets: dict[int, list[dict[str, Any]]] = {}
        for b in bars_1m:
            ts = int(b.get("t", 0))
            if ts == 0:
                continue
            key = ts // window_ns
            buckets.setdefault(key, []).append(b)

        result: list[Bar] = []
        for key in sorted(buckets):
            chunk = buckets[key]
            result.append(Bar(
                ts=int(chunk[0].get("t", 0)),
                o=float(chunk[0].get("o", 0)),
                h=max(float(b.get("h", 0)) for b in chunk),
                lo=min(float(b.get("l", 0)) for b in chunk),
                c=float(chunk[-1].get("c", 0)),
                v=sum(int(b.get("v", 0)) for b in chunk),
            ))
        return result

    # ── Signal detection ──────────────────────────────────────

    async def scan_signals(self) -> list[dict[str, Any]]:
        """Scan all tickers for entry signals on the latest 5-min bar."""
        signals: list[dict[str, Any]] = []

        for ticker in TRIDENT_TICKERS:
            bars = self.daily_bars_5m.get(ticker, [])
            if len(bars) < 10:
                continue

            snapshots = compute_all_signals(
                bars,
                daily_closes=self.daily_closes.get(ticker),
                ema_fast_period=self.cfg.entry.ema_fast,
                ema_slow_period=self.cfg.entry.ema_slow,
            )
            if not snapshots:
                continue

            # Only check the latest bar
            snap = snapshots[-1]
            bar_idx = snap.bar_idx

            # Time gate
            now = datetime.now(UTC)
            et_hour = now.hour - 4  # approximate ET
            et_min = now.minute
            mins_since_open = (et_hour - 9) * 60 + (et_min - 30)
            if mins_since_open < self.cfg.time_filter.skip_first_n_minutes:
                continue
            if mins_since_open > (390 - self.cfg.time_filter.skip_last_n_minutes):
                continue

            # Cooldown
            if (bar_idx - self.last_trade_bar.get(ticker, -999)
                    < self.cfg.position.min_bars_between_trades):
                continue

            # Already have position in this ticker?
            if ticker in self.positions:
                continue

            # Max positions
            if len(self.positions) >= self.cfg.position.max_positions:
                continue

            call_count, put_count, conviction = evaluate_signals(
                snap, self.cfg.entry,
            )

            take_call = (
                call_count >= self.cfg.entry.min_signals_call
                and self.cfg.direction in (Direction.LONG, Direction.BOTH)
            )
            take_put = (
                put_count >= self.cfg.entry.min_signals_put
                and self.cfg.direction in (Direction.SHORT, Direction.BOTH)
            )

            if take_call and take_put:
                if call_count >= put_count:
                    take_put = False
                else:
                    take_call = False

            if not take_call and not take_put:
                continue

            is_call = take_call
            sigs = call_count if is_call else put_count

            signals.append({
                "ticker": ticker,
                "is_call": is_call,
                "conviction": conviction,
                "signals_fired": sigs,
                "price": snap.bar.c,
                "bar_idx": bar_idx,
                "rsi3": snap.rsi3,
                "ibs": snap.ibs,
                "vwap_dist": snap.vwap_distance_pct,
                "vol_ratio": snap.volume_ratio,
            })

        return signals

    # ── Trade execution ───────────────────────────────────────

    async def execute_signal(self, signal: dict[str, Any]) -> None:
        """Execute a trade for a detected signal."""
        ticker = signal["ticker"]
        is_call = signal["is_call"]
        opt_type = "call" if is_call else "put"
        conviction = signal["conviction"]
        price = signal["price"]

        logger.info(
            "SIGNAL: %s %s | price=%.2f | conviction=%.1f | sigs=%d",
            ticker, opt_type.upper(), price, conviction,
            signal["signals_fired"],
        )

        if self.dry_run:
            self._log_event("signal_dry_run", signal)
            return

        # Find nearest ATM option via Polygon
        try:
            option = await self.polygon.get_nearest_atm_option(
                ticker, price, opt_type,
                min_dte=self.cfg.options.min_dte,
                max_dte=self.cfg.options.max_dte,
            )
            if option is None:
                logger.warning("No %s option found for %s", opt_type, ticker)
                return

            opt_symbol = option.contract_symbol
            ask = option.ask or option.last
            if not ask or ask < self.cfg.options.min_premium:
                logger.warning(
                    "Option %s ask too low: %.2f", opt_symbol, ask or 0,
                )
                return

            # Check spread
            if option.bid and option.ask and option.bid > 0:
                spread_pct = (option.ask - option.bid) / option.ask
                if spread_pct > self.cfg.options.max_spread_pct:
                    logger.warning(
                        "Spread too wide for %s: %.1f%%",
                        opt_symbol, spread_pct * 100,
                    )
                    return

            # Position sizing
            if self.alpaca is None:
                logger.warning("No Alpaca executor configured (dry_run mode)")
                return
            account = await self.alpaca.get_account()
            equity = float(account.get("equity", 25000))
            max_spend = equity * self.cfg.position.risk_per_trade
            contracts = max(1, int(max_spend / (ask * 100)))

            # Place order
            conv_int = min(9, max(1, int(conviction)))
            order = await self.alpaca.buy_option(
                symbol=opt_symbol,
                qty=contracts,
                order_type="limit",
                limit_price=ask,
                model_name=MODEL_NAME,
                conviction=conv_int,
            )

            pos = TridentPosition(
                ticker=ticker,
                is_call=is_call,
                option_symbol=opt_symbol,
                entry_time=datetime.now(UTC),
                entry_underlying=price,
                entry_option_price=ask,
                contracts=contracts,
                conviction=conviction,
                signals_fired=signal["signals_fired"],
                peak_option_price=ask,
                order_id=order.order_id if order else "",
            )
            self.positions[ticker] = pos
            self.last_trade_bar[ticker] = signal["bar_idx"]

            # Email alert
            trade_record = {
                "model": MODEL_NAME,
                "ticker": ticker,
                "option_symbol": opt_symbol,
                "direction": opt_type,
                "entry_price": price,
                "option_price": ask,
                "contracts": contracts,
                "conviction": conviction,
            }
            await send_trade_entry_alert(trade_record)

            self._log_event("entry", {
                **signal, "option_symbol": opt_symbol,
                "contracts": contracts, "ask": ask,
            })
            logger.info(
                "ENTRY: %s %s %s x%d @ $%.2f | conv=%.1f",
                ticker, opt_type, opt_symbol, contracts, ask, conviction,
            )

        except Exception as exc:
            logger.error("Execution failed for %s: %s", ticker, exc)

    # ── Exit monitoring ───────────────────────────────────────

    async def check_exits(self) -> None:
        """Check exit conditions for all active positions."""
        if not self.positions:
            return
        if self.alpaca is None:
            return

        # Check EOD
        now = datetime.now(UTC)
        et_hour = now.hour - 4
        et_min = now.minute
        mins_since_open = (et_hour - 9) * 60 + (et_min - 30)
        force_eod = mins_since_open > (
            390 - self.cfg.exit.eod_close_minutes_before
        )

        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            try:
                # Get current option price from Alpaca positions
                positions = await self.alpaca.get_positions()
                opt_pos = None
                for p in positions:
                    if p.ticker == pos.option_symbol:
                        opt_pos = p
                        break

                if opt_pos is None:
                    # Position might have been filled/closed externally
                    logger.warning(
                        "Position %s not found in Alpaca, removing",
                        pos.option_symbol,
                    )
                    del self.positions[ticker]
                    continue

                current_price = opt_pos.current_price
                entry_price = pos.entry_option_price
                pos.bars_held += 1

                # Update peak
                if current_price > pos.peak_option_price:
                    pos.peak_option_price = current_price

                # Determine exit reason
                exit_reason = None
                if force_eod:
                    exit_reason = "eod"
                elif entry_price > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct >= self.cfg.exit.tp_pct:
                        exit_reason = "tp"
                    elif pnl_pct <= self.cfg.exit.sl_pct:
                        exit_reason = "sl"
                    elif (
                        self.cfg.exit.use_trailing
                        and pos.peak_option_price > entry_price
                    ):
                        peak = pos.peak_option_price
                        retrace = (peak - current_price) / peak if peak > 0 else 0
                        if retrace >= self.cfg.exit.trail_pct:
                            exit_reason = "trail"
                    if pos.bars_held >= self.cfg.exit.max_hold_bars:
                        exit_reason = "time"

                if exit_reason is None:
                    continue

                # Execute exit
                opt_type = "call" if pos.is_call else "put"
                await self.alpaca.sell_option(
                    symbol=pos.option_symbol,
                    qty=pos.contracts,
                    order_type="market",
                    model_name=MODEL_NAME,
                    reason=exit_reason,
                )

                pnl_pct = (
                    (current_price - entry_price) / entry_price * 100
                    if entry_price > 0 else 0
                )
                pnl_dollars = (
                    (current_price - entry_price) * 100 * pos.contracts
                )

                # Email alert
                exit_record = {
                    "model": MODEL_NAME,
                    "ticker": ticker,
                    "option_symbol": pos.option_symbol,
                    "direction": opt_type,
                    "entry_price": pos.entry_underlying,
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "pnl_dollars": pnl_dollars,
                    "reason": exit_reason,
                    "hold_bars": pos.bars_held,
                }
                await send_trade_exit_alert(exit_record)

                self._log_event("exit", {
                    "ticker": ticker,
                    "option_symbol": pos.option_symbol,
                    "direction": opt_type,
                    "exit_reason": exit_reason,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "pnl_dollars": pnl_dollars,
                    "hold_bars": pos.bars_held,
                })

                logger.info(
                    "EXIT [%s]: %s %s | %.1f%% ($%.2f) | %d bars | %s",
                    exit_reason, ticker, opt_type,
                    pnl_pct, pnl_dollars, pos.bars_held,
                    pos.option_symbol,
                )

                del self.positions[ticker]
                self.trade_count += 1

            except Exception as exc:
                logger.error("Exit check failed for %s: %s", ticker, exc)

    # ── Logging ───────────────────────────────────────────────

    def _log_event(self, event: str, data: dict[str, Any]) -> None:
        """Append a JSON event to the daily log file."""
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            **data,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ── Main loop ─────────────────────────────────────────────

    async def run(self) -> None:
        """Main scanner loop — runs until market close."""
        logger.info("=" * 60)
        logger.info("Trident Live Scanner starting")
        logger.info("Tickers: %s", TRIDENT_TICKERS)
        logger.info("Dry run: %s", self.dry_run)
        logger.info("Direction: %s", self.cfg.direction.value)
        logger.info("=" * 60)

        # Start WebSocket data feed if configured
        if self._data_feed:
            await self._data_feed.start()

        await self.load_daily_bars()

        scan_interval = 60  # seconds
        exit_interval = 15  # seconds

        while True:
            now = datetime.now(UTC)
            et_hour = now.hour - 4
            et_min = now.minute
            mins_since_open = (et_hour - 9) * 60 + (et_min - 30)

            # Pre-market or post-market — sleep
            if mins_since_open < -5 or mins_since_open > 395:
                logger.info("Outside trading hours, sleeping 60s...")
                await asyncio.sleep(60)
                continue

            # Refresh bars and scan
            await self.refresh_bars()

            # Check exits first (higher priority)
            await self.check_exits()

            # Scan for new signals
            signals = await self.scan_signals()
            for sig in signals:
                await self.execute_signal(sig)

            # Status log every 5 minutes
            if mins_since_open % 5 == 0:
                logger.info(
                    "Status: %d positions, %d trades today, %d min into session",
                    len(self.positions), self.trade_count, mins_since_open,
                )

            # Sleep — shorter interval when holding positions
            interval = (
                exit_interval if self.positions else scan_interval
            )
            await asyncio.sleep(interval)


# ── Entry point ───────────────────────────────────────────────────

async def scanner_main(dry_run: bool = False) -> None:
    """Entry point for the Trident live scanner."""
    # Load config — use best optimized config if available
    config_path = Path("data/trident_backtest_results")
    cfg = TridentConfig(name="trident_live")

    # Try to load champion config from optimization
    champion_files = sorted(
        config_path.glob("trident_opt_*.json"), reverse=True,
    )
    if champion_files:
        try:
            data = json.loads(champion_files[0].read_text())
            best = data.get("best_config", {})
            if best:
                cfg = TridentConfig(**best)
                logger.info(
                    "Loaded champion config: %s", cfg.name,
                )
        except Exception as exc:
            logger.warning("Failed to load champion config: %s", exc)

    scanner = TridentScanner(config=cfg, dry_run=dry_run)
    await scanner.run()
