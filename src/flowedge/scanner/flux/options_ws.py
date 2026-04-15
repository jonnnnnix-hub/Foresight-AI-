"""Real-time options WebSocket consumer via Massive/Polygon.

Streams per-second option aggregates (A.*) for near-ATM contracts on
watchlist underlyings. Provides instant option price snapshots that
replace REST-based option chain lookups.

Requires Options Advanced plan ($199/mo) on Massive.
URL: wss://socket.massive.com/options
Auth key: MASSIVE_OPTIONS_WS_KEY env var (separate from stocks key).

Architecture:
  - ONE connection for all option contracts
  - Rolling buffers per underlying (last 60 seconds of bars)
  - Scanners call get_best_option(ticker, "call"/"put") for instant ATM lookup
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import structlog

logger = structlog.get_logger()

_BUFFER_SIZE = 120  # ~2 min of per-second bars per contract


@dataclass
class OptionBar:
    """Per-second option aggregate from WebSocket."""

    symbol: str         # O:SPY260415C00550000
    underlying: str     # SPY
    strike: float
    is_call: bool
    expiration: str     # 2026-04-15
    dte: int

    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    timestamp_ms: int   # Unix milliseconds

    @property
    def mid(self) -> float:
        return self.close  # Best available realtime price

    @property
    def age_seconds(self) -> float:
        return (time.time() * 1000 - self.timestamp_ms) / 1000


def _parse_opra_symbol(sym: str) -> dict[str, Any] | None:
    """Parse OPRA symbol like O:SPY260415C00550000."""
    try:
        if sym.startswith("O:"):
            sym = sym[2:]

        # Find where the date starts (first digit after ticker)
        i = 0
        while i < len(sym) and not sym[i].isdigit():
            i += 1
        if i == 0 or i >= len(sym) - 9:
            return None

        ticker = sym[:i]
        date_str = sym[i:i + 6]  # YYMMDD
        cp = sym[i + 6]          # C or P
        strike_raw = sym[i + 7:] # 00550000

        expiration = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
        strike = int(strike_raw) / 1000.0
        exp_date = date.fromisoformat(expiration)
        dte = (exp_date - date.today()).days

        return {
            "ticker": ticker,
            "expiration": expiration,
            "is_call": cp == "C",
            "strike": strike,
            "dte": dte,
        }
    except Exception:
        return None


class OptionsWebSocketConsumer:
    """Real-time options WebSocket: streams per-second aggregates.

    Usage:
        consumer = OptionsWebSocketConsumer(
            api_key=os.getenv("MASSIVE_OPTIONS_WS_KEY"),
            underlyings=["SPY", "QQQ", "AAPL", ...],
        )
        await consumer.start()

        # Get best ATM call for SPY at current price
        bar = consumer.get_best_option("SPY", 550.0, "call", max_dte=5)
        if bar:
            print(f"Best: {bar.symbol} @ ${bar.close} vol={bar.volume}")
    """

    def __init__(
        self,
        api_key: str = "",
        underlyings: list[str] | None = None,
        ws_url: str = "wss://socket.massive.com/options",
        max_dte: int = 7,
    ) -> None:
        self._api_key = api_key or os.getenv("MASSIVE_OPTIONS_WS_KEY", "")
        self._underlyings = [t.upper() for t in (underlyings or [])]
        self._ws_url = ws_url
        self._max_dte = max_dte
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._connected = False

        # Buffers: {underlying: {opra_symbol: deque[OptionBar]}}
        self._bars: dict[str, dict[str, deque[OptionBar]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=_BUFFER_SIZE)),
        )

        # Latest price per underlying (from most recent option trade)
        self._latest_underlying_prices: dict[str, float] = {}

        # Stats
        self._bars_received = 0
        self._errors = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def start(self) -> None:
        if not self._api_key:
            logger.warning("options_ws_no_key", hint="Set MASSIVE_OPTIONS_WS_KEY")
            return
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "options_ws_starting",
            url=self._ws_url,
            underlyings=self._underlyings,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(
            "options_ws_stopped",
            bars_received=self._bars_received,
            errors=self._errors,
        )

    async def close(self) -> None:
        await self.stop()

    # ── Data access ───────────────────────────────────────────

    def get_best_option(
        self,
        underlying: str,
        current_price: float,
        option_type: str = "call",
        max_dte: int = 5,
        max_spread_pct: float = 15.0,
    ) -> OptionBar | None:
        """Find the nearest ATM option with recent activity.

        Returns the OptionBar closest to current_price that:
        - Is the correct type (call/put)
        - Has DTE <= max_dte
        - Had activity in the last 60 seconds
        """
        is_call = option_type.lower() == "call"
        underlying = underlying.upper()
        contract_bars = self._bars.get(underlying, {})

        if not contract_bars:
            return None

        cutoff_ms = int((time.time() - 60) * 1000)
        best: OptionBar | None = None
        best_distance = 999.0

        for _sym, bar_deque in contract_bars.items():
            if not bar_deque:
                continue
            latest = bar_deque[-1]

            # Filter: correct type, DTE, freshness
            if latest.is_call != is_call:
                continue
            if latest.dte > max_dte or latest.dte < 0:
                continue
            if latest.timestamp_ms < cutoff_ms:
                continue

            distance = abs(latest.strike - current_price)
            if distance < best_distance:
                best_distance = distance
                best = latest

        return best

    def get_option_bars(
        self, underlying: str, symbol: str,
    ) -> list[OptionBar]:
        """Get recent bars for a specific option contract."""
        return list(self._bars.get(underlying.upper(), {}).get(symbol, []))

    def get_active_contracts(self, underlying: str) -> list[str]:
        """List OPRA symbols with recent activity for an underlying."""
        cutoff_ms = int((time.time() - 120) * 1000)
        result = []
        for sym, bar_deque in self._bars.get(underlying.upper(), {}).items():
            if bar_deque and bar_deque[-1].timestamp_ms >= cutoff_ms:
                result.append(sym)
        return result

    # ── WebSocket connection ──────────────────────────────────

    async def _run_loop(self) -> None:
        """Persistent connection loop with auto-reconnect."""
        while self._running:
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors += 1
                self._connected = False
                logger.warning(
                    "options_ws_error",
                    error=str(e)[:200],
                    reconnect_in=5,
                    total_errors=self._errors,
                )
                await asyncio.sleep(5)

    async def _connect_and_stream(self) -> None:
        import websockets

        logger.info("options_ws_connecting", url=self._ws_url)
        async with websockets.connect(
            self._ws_url, ping_interval=30, ping_timeout=10, close_timeout=5,
        ) as ws:
            self._connected = True
            logger.info("options_ws_connected")

            # Connection confirmation
            msg = await ws.recv()
            data = json.loads(msg)
            if isinstance(data, list) and data:
                if data[0].get("status") == "connected":
                    logger.info("options_ws_confirmed")

            # Authenticate
            await ws.send(json.dumps({
                "action": "auth",
                "params": self._api_key,
            }))
            auth_msg = await ws.recv()
            auth_data = json.loads(auth_msg)
            if isinstance(auth_data, list) and auth_data:
                if auth_data[0].get("status") != "auth_success":
                    logger.error(
                        "options_ws_auth_failed",
                        message=auth_data[0].get("message", ""),
                    )
                    self._running = False
                    return
            logger.info("options_ws_authenticated")

            # Subscribe to per-second aggregates for all underlyings
            # A.O:{underlying}* captures all strikes/expirations
            subs = ",".join(f"A.O:{t}*" for t in self._underlyings)
            await ws.send(json.dumps({
                "action": "subscribe",
                "params": subs,
            }))
            logger.info(
                "options_ws_subscribed",
                underlyings=self._underlyings,
                pattern="A.O:{ticker}*",
            )

            # Stream
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
                    pass

    def _handle_message(self, msg: dict[str, Any]) -> None:
        ev = msg.get("ev", "")
        if ev not in ("A", "AM"):
            return

        sym = msg.get("sym", "")
        if not sym:
            return

        parsed = _parse_opra_symbol(sym)
        if not parsed:
            return

        # Filter by DTE
        if parsed["dte"] > self._max_dte or parsed["dte"] < 0:
            return

        bar = OptionBar(
            symbol=sym,
            underlying=parsed["ticker"],
            strike=parsed["strike"],
            is_call=parsed["is_call"],
            expiration=parsed["expiration"],
            dte=parsed["dte"],
            open=float(msg.get("o", 0)),
            high=float(msg.get("h", 0)),
            low=float(msg.get("l", 0)),
            close=float(msg.get("c", 0)),
            volume=int(msg.get("v", 0)),
            vwap=float(msg.get("vw", msg.get("a", 0))),
            timestamp_ms=int(msg.get("s", 0)),
        )

        self._bars[parsed["ticker"]][sym].append(bar)
        self._bars_received += 1

    def get_stats(self) -> dict[str, Any]:
        total_contracts = sum(
            len(contracts) for contracts in self._bars.values()
        )
        return {
            "connected": self._connected,
            "bars_received": self._bars_received,
            "errors": self._errors,
            "underlyings_tracked": len(self._bars),
            "total_contracts": total_contracts,
        }
