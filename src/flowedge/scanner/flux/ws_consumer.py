"""MassiveDataFeed — central real-time data layer via WebSocket.

Single persistent WebSocket connection to Massive/Polygon that streams:
  - T.* — Trade ticks (for FLUX Lee-Ready classification)
  - Q.* — NBBO quotes (for FLUX midpoint + L1 imbalance)
  - AM.* — Aggregate minute bars (replaces REST bar polling for ALL bots)

Replaces ~11,000-21,000 daily REST calls with one WebSocket connection.
All data reads are instant synchronous buffer reads — zero HTTP overhead.

Usage:
    feed = MassiveDataFeed(api_key, tickers)
    await feed.start()
    # ... later, in scan cycle:
    bars = feed.get_bars("SPY", count=60)        # last 60 1-min bars
    trades = feed.get_trades("SPY", window_minutes=5)
    price = feed.get_latest_price("SPY")
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from typing import Any

import structlog

from flowedge.scanner.data_feeds.schemas import BarData, Timeframe
from flowedge.scanner.flux.schemas import NBBOQuote, TradeTick

logger = structlog.get_logger()

# Trade conditions to exclude (same as REST consumer)
_EXCLUDE_CONDITIONS = frozenset({
    2, 7, 10, 15, 16, 22, 29, 33, 38, 52, 53,
})

# Max buffer sizes per ticker
_MAX_TRADE_BUFFER = 20_000   # ~15 min at 1000 trades/min
_MAX_QUOTE_BUFFER = 5_000
_MAX_BAR_BUFFER = 500        # ~8 hours of 1-min bars

# Buffer window for trades/quotes: keep 15 minutes
_BUFFER_WINDOW_NS = 15 * 60 * 1_000_000_000


class MassiveDataFeed:
    """Central real-time data layer via Massive/Polygon WebSocket.

    Streams trades (T.*), quotes (Q.*), and minute bars (AM.*) for
    all scanner tickers. Maintains rolling in-memory buffers so all
    bots can read data instantly without HTTP calls.
    """

    def __init__(
        self,
        api_key: str,
        tickers: list[str],
        ws_url: str = "wss://socket.massive.com/stocks",
    ) -> None:
        self._api_key = api_key
        self._tickers = [t.upper() for t in tickers]
        self._ws_url = ws_url
        self._ws: Any = None
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._connected = False
        self._authenticated = False

        # Rolling buffers per ticker
        self._trade_buffers: dict[str, deque[TradeTick]] = {
            t: deque(maxlen=_MAX_TRADE_BUFFER) for t in self._tickers
        }
        self._quote_buffers: dict[str, deque[NBBOQuote]] = {
            t: deque(maxlen=_MAX_QUOTE_BUFFER) for t in self._tickers
        }
        self._bar_buffers: dict[str, deque[BarData]] = {
            t: deque(maxlen=_MAX_BAR_BUFFER) for t in self._tickers
        }

        # Last known prices per ticker (from most recent trade)
        self._last_prices: dict[str, float] = {}

        # Stats
        self._trades_received = 0
        self._quotes_received = 0
        self._bars_received = 0
        self._errors = 0

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the WebSocket connection as a background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "ws_consumer_starting",
            url=self._ws_url,
            tickers=self._tickers,
        )

    async def stop(self) -> None:
        """Stop the WebSocket connection cleanly."""
        self._running = False
        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self._connected = False
        self._authenticated = False
        logger.info(
            "ws_consumer_stopped",
            trades_total=self._trades_received,
            quotes_total=self._quotes_received,
        )

    async def close(self) -> None:
        """Alias for stop() — matches REST consumer interface."""
        await self.stop()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._authenticated

    # ── Data Access (synchronous reads from buffer) ──────────────

    def get_trades(
        self, ticker: str, window_minutes: int = 5,
    ) -> list[TradeTick]:
        """Return buffered trades for the last N minutes.

        This is a synchronous read — the WebSocket fills the buffer
        continuously in the background.
        """
        ticker = ticker.upper()
        buf = self._trade_buffers.get(ticker)
        if not buf:
            return []

        cutoff_ns = _now_ns() - (window_minutes * 60 * 1_000_000_000)
        return [t for t in buf if t.timestamp >= cutoff_ns]

    def get_quotes(
        self, ticker: str, window_minutes: int = 5,
    ) -> list[NBBOQuote]:
        """Return buffered quotes for the last N minutes."""
        ticker = ticker.upper()
        buf = self._quote_buffers.get(ticker)
        if not buf:
            return []

        cutoff_ns = _now_ns() - (window_minutes * 60 * 1_000_000_000)
        return [q for q in buf if q.timestamp >= cutoff_ns]

    def get_bars(
        self, ticker: str, count: int = 60,
    ) -> list[BarData]:
        """Return the last N 1-minute bars from the buffer.

        Equivalent to PolygonIntradayProvider.get_today_bars() but
        instant (synchronous buffer read, no HTTP call).
        """
        ticker = ticker.upper()
        buf = self._bar_buffers.get(ticker)
        if not buf:
            return []
        # Return last `count` bars
        bars = list(buf)
        return bars[-count:] if len(bars) > count else bars

    def get_latest_price(self, ticker: str) -> float:
        """Return the last known trade price for a ticker."""
        return self._last_prices.get(ticker.upper(), 0.0)

    def get_latest_bar(self, ticker: str) -> BarData | None:
        """Return the most recent 1-minute bar."""
        ticker = ticker.upper()
        buf = self._bar_buffers.get(ticker)
        return buf[-1] if buf else None

    def add_ticker(self, ticker: str) -> None:
        """Add a ticker to track (requires re-subscribe)."""
        ticker = ticker.upper()
        if ticker not in self._tickers:
            self._tickers.append(ticker)
            self._trade_buffers[ticker] = deque(maxlen=_MAX_TRADE_BUFFER)
            self._quote_buffers[ticker] = deque(maxlen=_MAX_QUOTE_BUFFER)
            self._bar_buffers[ticker] = deque(maxlen=_MAX_BAR_BUFFER)

    # ── WebSocket Loop ──────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main WebSocket loop with auto-reconnect."""
        while self._running:
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors += 1
                self._connected = False
                self._authenticated = False
                logger.warning(
                    "ws_connection_error",
                    error=str(e),
                    reconnect_in=5,
                    total_errors=self._errors,
                )
                await asyncio.sleep(5)

    async def _connect_and_stream(self) -> None:
        """Connect, authenticate, subscribe, and stream messages."""
        try:
            import websockets
        except ImportError:
            logger.error(
                "websockets_not_installed",
                hint="pip install websockets",
            )
            self._running = False
            return

        logger.info("ws_connecting", url=self._ws_url)
        async with websockets.connect(
            self._ws_url,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._connected = True
            logger.info("ws_connected")

            # Wait for connection confirmation
            msg = await ws.recv()
            data = json.loads(msg)
            if isinstance(data, list) and data:
                status = data[0].get("status", "")
                if status == "connected":
                    logger.info("ws_connection_confirmed")

            # Authenticate
            await ws.send(json.dumps({
                "action": "auth",
                "params": self._api_key,
            }))

            auth_msg = await ws.recv()
            auth_data = json.loads(auth_msg)
            if isinstance(auth_data, list) and auth_data:
                auth_status = auth_data[0].get("status", "")
                if auth_status == "auth_success":
                    self._authenticated = True
                    logger.info("ws_authenticated")
                else:
                    logger.error(
                        "ws_auth_failed",
                        status=auth_status,
                        message=auth_data[0].get("message", ""),
                    )
                    self._running = False
                    return

            # Subscribe to trades, quotes, and minute bars for all tickers
            trade_channels = ",".join(f"T.{t}" for t in self._tickers)
            quote_channels = ",".join(f"Q.{t}" for t in self._tickers)
            bar_channels = ",".join(f"AM.{t}" for t in self._tickers)
            subs = f"{trade_channels},{quote_channels},{bar_channels}"

            await ws.send(json.dumps({
                "action": "subscribe",
                "params": subs,
            }))
            logger.info(
                "ws_subscribed",
                tickers=self._tickers,
                channels=len(self._tickers) * 3,
                types=["T", "Q", "AM"],
            )

            # Stream messages
            async for raw_msg in ws:
                if not self._running:
                    break
                try:
                    messages = json.loads(raw_msg)
                    if not isinstance(messages, list):
                        messages = [messages]
                    for msg in messages:
                        self._handle_message(msg)
                except json.JSONDecodeError:
                    continue

    # ── Message Handling ────────────────────────────────────────

    def _handle_message(self, msg: dict[str, Any]) -> None:
        """Route a single message to the appropriate buffer."""
        ev = msg.get("ev", "")
        if ev == "T":
            self._handle_trade(msg)
        elif ev == "Q":
            self._handle_quote(msg)
        elif ev == "AM":
            self._handle_bar(msg)

    def _handle_trade(self, msg: dict[str, Any]) -> None:
        """Parse a trade message and add to buffer."""
        sym = msg.get("sym", "")
        if sym not in self._trade_buffers:
            return

        conditions = msg.get("c", []) or []
        if any(c in _EXCLUDE_CONDITIONS for c in conditions):
            return

        size = int(msg.get("s", 0))
        if size <= 0:
            return

        # Massive timestamp is Unix milliseconds → convert to nanoseconds
        ts_ms = int(msg.get("t", 0))
        ts_ns = ts_ms * 1_000_000

        tick = TradeTick(
            price=float(msg.get("p", 0)),
            size=size,
            timestamp=ts_ns,
            conditions=conditions,
            exchange=int(msg.get("x", 0)),
        )

        self._trade_buffers[sym].append(tick)
        self._last_prices[sym] = tick.price
        self._trades_received += 1

        # Periodic stats
        if self._trades_received % 10_000 == 0:
            self._prune_buffers()
            logger.debug(
                "ws_trades_buffered",
                total=self._trades_received,
                buffer_sizes={
                    t: len(b) for t, b in self._trade_buffers.items() if b
                },
            )

    def _handle_quote(self, msg: dict[str, Any]) -> None:
        """Parse a quote message and add to buffer."""
        sym = msg.get("sym", "")
        if sym not in self._quote_buffers:
            return

        bid = float(msg.get("bp", 0))
        ask = float(msg.get("ap", 0))
        if bid <= 0 or ask <= 0 or ask < bid:
            return

        ts_ms = int(msg.get("t", 0))
        ts_ns = ts_ms * 1_000_000

        quote = NBBOQuote(
            bid=bid,
            bid_size=int(msg.get("bs", 0)),
            ask=ask,
            ask_size=int(msg.get("as", 0)),
            timestamp=ts_ns,
        )

        self._quote_buffers[sym].append(quote)
        self._quotes_received += 1

    def _handle_bar(self, msg: dict[str, Any]) -> None:
        """Parse an aggregate minute bar (AM) and add to buffer.

        AM message format:
        {"ev":"AM", "sym":"SPY", "v":12345, "o":150.85, "c":152.90,
         "h":153.17, "l":150.50, "a":151.87, "s":1611082800000, "e":1611082860000}
        """
        sym = msg.get("sym", "")
        if sym not in self._bar_buffers:
            return

        from datetime import datetime

        ts_ms = int(msg.get("s", 0))
        bar = BarData(
            ticker=sym,
            timestamp=datetime.fromtimestamp(ts_ms / 1000),
            timeframe=Timeframe.MIN_1,
            open=float(msg.get("o", 0)),
            high=float(msg.get("h", 0)),
            low=float(msg.get("l", 0)),
            close=float(msg.get("c", 0)),
            volume=int(msg.get("v", 0)),
            vwap=float(msg.get("a", 0)),
            trade_count=int(msg.get("n", 0)),
        )

        self._bar_buffers[sym].append(bar)
        self._last_prices[sym] = bar.close
        self._bars_received += 1

        if self._bars_received % 100 == 0:
            logger.debug(
                "ws_bars_buffered",
                total=self._bars_received,
                buffer_sizes={
                    t: len(b) for t, b in self._bar_buffers.items() if b
                },
            )

    def _prune_buffers(self) -> None:
        """Remove entries older than the buffer window."""
        cutoff = _now_ns() - _BUFFER_WINDOW_NS
        for buf in self._trade_buffers.values():
            while buf and buf[0].timestamp < cutoff:
                buf.popleft()
        for qbuf in self._quote_buffers.values():
            while qbuf and qbuf[0].timestamp < cutoff:
                qbuf.popleft()


# Backward-compatible alias
MassiveWebSocketConsumer = MassiveDataFeed


def _now_ns() -> int:
    """Current time in nanoseconds."""
    return int(time.time() * 1_000_000_000)
