"""WebSocket streaming engine — pushes live scan results to connected clients.

Runs scan cycles in the background and broadcasts results to all
connected WebSocket clients via FastAPI WebSocket endpoint.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import structlog
from fastapi import WebSocket

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.catalyst.engine import scan_catalysts
from flowedge.scanner.flux.consumer import PolygonTradeConsumer
from flowedge.scanner.flux.engine import scan_flux
from flowedge.scanner.iv_rank.engine import scan_iv
from flowedge.scanner.providers.registry import ProviderRegistry
from flowedge.scanner.scorer.engine import score_lottos
from flowedge.scanner.uoa.engine import scan_uoa

logger = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info("ws_client_connected", total=len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)
        logger.info("ws_client_disconnected", total=len(self._connections))

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send data to all connected clients."""
        message = json.dumps(data, default=str)
        dead: list[WebSocket] = []

        async with self._lock:
            connections = list(self._connections)

        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self._connections:
                        self._connections.remove(ws)

    @property
    def client_count(self) -> int:
        return len(self._connections)


# Global manager instance
manager = ConnectionManager()


async def run_streaming_scanner(
    tickers: list[str],
    interval_seconds: int = 60,
    min_score: float = 0.0,
    settings: Settings | None = None,
) -> None:
    """Run continuous scanner loop, broadcasting results via WebSocket.

    This runs as a background task in the FastAPI app.
    """
    settings = settings or get_settings()

    while True:
        if manager.client_count == 0:
            await asyncio.sleep(5)
            continue

        try:
            registry = ProviderRegistry(settings)
            flux_consumer = PolygonTradeConsumer(settings.polygon_api_key)
            try:
                uoa_signals = await scan_uoa(registry, tickers, settings)
                iv_signals = await scan_iv(registry, tickers, settings)
                catalyst_signals = await scan_catalysts(
                    registry, tickers, settings
                )
                flux_signals = await scan_flux(
                    flux_consumer, tickers, settings,
                )
                result = score_lottos(
                    uoa_signals, iv_signals, catalyst_signals, settings,
                    flux_signals=flux_signals,
                )

                filtered = [
                    o for o in result.top_opportunities
                    if o.composite_score >= min_score
                ]

                payload = {
                    "type": "scan_update",
                    "timestamp": datetime.now().isoformat(),
                    "scan_id": result.scan_id,
                    "tickers_scanned": result.tickers_scanned,
                    "opportunities": [
                        o.model_dump(mode="json") for o in filtered
                    ],
                }

                await manager.broadcast(payload)
                logger.info(
                    "ws_broadcast",
                    clients=manager.client_count,
                    opportunities=len(filtered),
                )

            finally:
                await registry.close_all()
                await flux_consumer.close()

        except Exception as e:
            logger.error("streaming_scan_error", error=str(e))
            await manager.broadcast({
                "type": "error",
                "timestamp": datetime.now().isoformat(),
                "message": str(e),
            })

        await asyncio.sleep(interval_seconds)
