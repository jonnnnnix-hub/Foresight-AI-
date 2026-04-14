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

    tickers: list[str] = Field(
        min_length=1,
        max_length=50,
        description="Tickers to scan (max 50)",
    )
    scan_types: list[str] = Field(
        default=["uoa", "iv", "catalyst", "flux"],
        description="Which scanners to run",
    )
    min_score: float = Field(default=0.0, ge=0.0, le=10.0)


class BacktestRequest(BaseModel):
    """Request to run a backtest."""

    tickers: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Tickers to backtest (max 20)",
    )
    lookback_days: int = Field(default=90, ge=7, le=365)
    max_hold_days: int = Field(default=10, ge=1, le=30)
    take_profit_pct: float = Field(default=100.0, ge=10.0, le=1000.0)
    stop_loss_pct: float = Field(default=-80.0, ge=-99.0, le=-10.0)


@scanner_router.get("/health")
async def scanner_health() -> dict[str, str]:
    """Scanner health check."""
    return {"status": "ok", "module": "scanner"}


@scanner_router.get("/status")
async def scanner_status() -> dict:  # type: ignore[type-arg]
    """Full provider connectivity status."""
    from flowedge.scanner.healthcheck import check_all_providers

    statuses = await check_all_providers()
    connected = sum(1 for s in statuses if s.healthy)
    return {
        "connected": connected,
        "total": len(statuses),
        "providers": [s.to_dict() for s in statuses],
    }


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

        # FLUX order flow scan
        flux_signals = None
        if "flux" in request.scan_types and settings.polygon_api_key:
            from flowedge.scanner.flux.consumer import PolygonTradeConsumer
            from flowedge.scanner.flux.engine import scan_flux

            flux_consumer = PolygonTradeConsumer(settings.polygon_api_key)
            try:
                flux_signals = await scan_flux(
                    flux_consumer, request.tickers, settings,
                )
            finally:
                await flux_consumer.close()

        # Fetch market-wide regime from UW Market Tide
        market_tide = None
        try:
            flow_provider = registry.get_flow_provider()
            market_tide = await flow_provider.get_market_tide()  # type: ignore[attr-defined]
        except (AttributeError, Exception):
            pass

        result = score_lottos(
            uoa_signals, iv_signals, catalyst_signals, settings,
            market_tide=market_tide,
            flux_signals=flux_signals,
        )

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


@scanner_router.get("/flux/{ticker}")
async def get_flux(ticker: str) -> dict:  # type: ignore[type-arg]
    """Get FLUX order flow signal for a ticker."""
    from flowedge.scanner.flux.consumer import PolygonTradeConsumer
    from flowedge.scanner.flux.engine import scan_flux_for_snapshot

    settings = get_settings()
    consumer = PolygonTradeConsumer(settings.polygon_api_key)
    try:
        signal = await scan_flux_for_snapshot(consumer, ticker.upper(), settings)
        return signal.model_dump(mode="json")
    finally:
        await consumer.close()


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
    request: BacktestRequest,
) -> dict:  # type: ignore[type-arg]
    """Run a backtest on given tickers (max 20, 7-365d lookback)."""
    from flowedge.scanner.backtest.engine import run_backtest

    result = await run_backtest(
        tickers=[t.upper() for t in request.tickers],
        lookback_days=request.lookback_days,
        max_hold_days=request.max_hold_days,
        take_profit_pct=request.take_profit_pct,
        stop_loss_pct=request.stop_loss_pct,
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


@scanner_router.get("/learning/weights")
async def get_learning_weights() -> dict:  # type: ignore[type-arg]
    """Get current adaptive scoring weights."""
    from flowedge.scanner.learning.adaptive import load_weights

    weights = load_weights()
    return weights.model_dump(mode="json")


@scanner_router.post("/learning/cycle")
async def run_learning_cycle_api(
    dry_run: bool = False,
    max_losses: int = 15,
) -> dict:  # type: ignore[type-arg]
    """Run a learning cycle to refine the scoring model."""
    from flowedge.scanner.learning.feedback import run_learning_cycle

    refinement = await run_learning_cycle(dry_run=dry_run, max_losses=max_losses)
    if refinement is None:
        return {"status": "insufficient_data"}
    return {
        "status": "complete" if not dry_run else "dry_run",
        "cycle_id": refinement.cycle_id,
        "losses_analyzed": refinement.losses_analyzed,
        "insights": len(refinement.insights),
        "weight_changes": len(refinement.weight_adjustments),
        "new_rules": len(refinement.new_rules),
        "failure_distribution": refinement.failure_distribution,
        "rationale": refinement.rationale[:300],
    }


@scanner_router.get("/learning/history")
async def get_learning_history() -> list:  # type: ignore[type-arg]
    """Get learning cycle history."""
    import json
    from pathlib import Path

    history_file = Path("./data/learning/feedback_log.json")
    if not history_file.exists():
        return []
    try:
        return json.loads(history_file.read_text())  # type: ignore[no-any-return]
    except Exception:
        return []


@scanner_router.post("/learning/pipeline")
async def run_full_pipeline_api() -> dict:  # type: ignore[type-arg]
    """Run simulation + learning cycle as a single pipeline."""
    from flowedge.scanner.learning.feedback import run_full_pipeline

    result = await run_full_pipeline()
    return dict(result)


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
