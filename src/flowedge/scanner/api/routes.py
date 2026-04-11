"""Scanner API routes — options flow, IV, catalysts, and lotto scoring."""

from __future__ import annotations

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from flowedge.config.settings import get_settings
from flowedge.scanner.catalyst.engine import scan_catalysts
from flowedge.scanner.iv_rank.engine import scan_iv
from flowedge.scanner.providers.registry import ProviderRegistry
from flowedge.scanner.schemas.catalyst import CatalystSignal
from flowedge.scanner.schemas.flow import UOASignal
from flowedge.scanner.schemas.iv import IVSignal
from flowedge.scanner.schemas.options import OptionsChain
from flowedge.scanner.schemas.signals import LottoOpportunity, ScannerResult
from flowedge.scanner.scorer.engine import score_lottos
from flowedge.scanner.uoa.engine import scan_uoa

scanner_router = APIRouter(prefix="/scanner", tags=["scanner"])


class ScanRequest(BaseModel):
    """Request to scan tickers."""

    tickers: list[str] = Field(min_length=1)
    scan_types: list[str] = Field(
        default=["uoa", "iv", "catalyst"],
        description="Which scanners to run",
    )
    min_score: float = Field(default=0.0, ge=0.0, le=10.0)


@scanner_router.get("/health")
async def scanner_health() -> dict[str, str]:
    """Scanner health check."""
    return {"status": "ok", "module": "scanner"}


@scanner_router.post("/scan", response_model=ScannerResult)
async def run_scan(request: ScanRequest) -> ScannerResult:
    """Run full scanner pipeline on given tickers."""
    settings = get_settings()
    registry = ProviderRegistry(settings)

    try:
        uoa_signals: list[UOASignal] = []
        iv_signals: list[IVSignal] = []
        catalyst_signals: list[CatalystSignal] = []

        if "uoa" in request.scan_types:
            uoa_signals = await scan_uoa(registry, request.tickers, settings)
        if "iv" in request.scan_types:
            iv_signals = await scan_iv(registry, request.tickers, settings)
        if "catalyst" in request.scan_types:
            catalyst_signals = await scan_catalysts(
                registry, request.tickers, settings
            )

        result = score_lottos(uoa_signals, iv_signals, catalyst_signals, settings)

        # Filter by min score
        if request.min_score > 0:
            result.opportunities = [
                o for o in result.opportunities
                if o.composite_score >= request.min_score
            ]

        return result
    finally:
        await registry.close_all()


@scanner_router.get("/flow/{ticker}", response_model=list[UOASignal])
async def get_flow(ticker: str) -> list[UOASignal]:
    """Get unusual options flow for a ticker."""
    settings = get_settings()
    registry = ProviderRegistry(settings)
    try:
        return await scan_uoa(registry, [ticker.upper()], settings)
    finally:
        await registry.close_all()


@scanner_router.get("/iv/{ticker}", response_model=IVSignal | None)
async def get_iv(ticker: str) -> IVSignal | None:
    """Get IV rank and regime for a ticker."""
    settings = get_settings()
    registry = ProviderRegistry(settings)
    try:
        signals = await scan_iv(registry, [ticker.upper()], settings)
        return signals[0] if signals else None
    finally:
        await registry.close_all()


@scanner_router.get("/catalysts", response_model=list[CatalystSignal])
async def get_catalysts(
    tickers: list[str] = Query(default=[]),
    days_ahead: int = Query(default=14, ge=1, le=90),
) -> list[CatalystSignal]:
    """Get upcoming catalysts for tickers."""
    settings = get_settings()
    # Override lookforward
    settings.catalyst_lookforward_days = days_ahead
    registry = ProviderRegistry(settings)
    try:
        return await scan_catalysts(registry, [t.upper() for t in tickers], settings)
    finally:
        await registry.close_all()


@scanner_router.get("/chain/{ticker}", response_model=OptionsChain)
async def get_chain(
    ticker: str,
    source: str = Query(default="tradier", pattern="^(tradier|polygon)$"),
) -> OptionsChain:
    """Get live options chain for a ticker."""
    settings = get_settings()
    registry = ProviderRegistry(settings)
    try:
        provider = registry.get_options_chain_provider(preferred=source)
        return await provider.get_options_chain(ticker.upper())
    finally:
        await registry.close_all()


@scanner_router.get("/opportunities", response_model=list[LottoOpportunity])
async def get_opportunities(
    tickers: list[str] = Query(default=[]),
    min_score: float = Query(default=5.0, ge=0.0, le=10.0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[LottoOpportunity]:
    """Get top lotto opportunities across tickers."""
    settings = get_settings()
    registry = ProviderRegistry(settings)
    try:
        uoa_signals = await scan_uoa(
            registry, [t.upper() for t in tickers] if tickers else None, settings
        )
        iv_tickers = [t.upper() for t in tickers] if tickers else [
            s.ticker for s in uoa_signals[:20]
        ]
        iv_signals = await scan_iv(registry, iv_tickers, settings)
        catalyst_signals = await scan_catalysts(registry, iv_tickers, settings)

        result = score_lottos(uoa_signals, iv_signals, catalyst_signals, settings)
        filtered = [
            o for o in result.top_opportunities
            if o.composite_score >= min_score
        ]
        return filtered[:limit]
    finally:
        await registry.close_all()


@scanner_router.get("/gex/{ticker}")
async def get_gex(ticker: str) -> dict:  # type: ignore[type-arg]
    """Get gamma exposure profile for a ticker."""
    from flowedge.scanner.gex.engine import compute_gex_profile

    settings = get_settings()
    registry = ProviderRegistry(settings)
    try:
        profile = await compute_gex_profile(ticker.upper(), registry, settings)
        return profile.model_dump(mode="json")
    finally:
        await registry.close_all()


@scanner_router.get("/portfolio")
async def get_portfolio() -> dict:  # type: ignore[type-arg]
    """Get paper trading portfolio state."""
    from flowedge.scanner.paper_trading.engine import AlpacaPaperTrader

    trader = AlpacaPaperTrader()
    try:
        portfolio = await trader.get_portfolio()
        return portfolio.model_dump(mode="json")
    finally:
        await trader.close()


@scanner_router.post("/backtest")
async def run_backtest_api(
    request: ScanRequest,
) -> dict:  # type: ignore[type-arg]
    """Run a backtest on given tickers."""
    from flowedge.scanner.backtest.engine import run_backtest

    result = await run_backtest(
        tickers=[t.upper() for t in request.tickers],
    )
    return result.model_dump(mode="json")


@scanner_router.get("/momentum/{ticker}")
async def get_momentum(ticker: str) -> dict:  # type: ignore[type-arg]
    """Get multi-timeframe momentum analysis."""
    from flowedge.scanner.momentum.engine import analyze_momentum

    signal = await analyze_momentum(ticker.upper())
    return signal.model_dump(mode="json")


@scanner_router.post("/interpret")
async def interpret_api(
    request: ScanRequest,
) -> list[dict]:  # type: ignore[type-arg]
    """AI-interpret top opportunities for given tickers."""
    from flowedge.scanner.interpreter.engine import interpret_batch

    settings = get_settings()
    registry = ProviderRegistry(settings)
    try:
        uoa = await scan_uoa(registry, request.tickers, settings)
        iv = await scan_iv(registry, request.tickers, settings)
        cat = await scan_catalysts(registry, request.tickers, settings)
        result = score_lottos(uoa, iv, cat, settings)
        theses = await interpret_batch(result.top_opportunities, settings=settings)
        return [t.model_dump(mode="json") for t in theses]
    finally:
        await registry.close_all()


@scanner_router.websocket("/ws")
async def websocket_stream(ws: WebSocket) -> None:
    """WebSocket endpoint for real-time scan updates."""
    from flowedge.scanner.streaming.engine import manager

    await manager.connect(ws)
    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await ws.receive_text()
            # Client can send ticker updates
            if data:
                await ws.send_text('{"type": "ack"}')
    except WebSocketDisconnect:
        await manager.disconnect(ws)
