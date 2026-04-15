"""Unified Options Process — single async event loop running all options scanners.

Solves the WebSocket connection conflict: Massive/Polygon allows exactly 1
concurrent WebSocket per API key per asset class. Instead of 4 processes
each opening their own connection (→ crash loop), one process shares a
single MassiveDataFeed across all scanners.

Architecture:
  MassiveDataFeed (1 WebSocket) ──┬── ScalpV2Scanner (Acct 1)
         │                        ├── VolumeScalperV1Scanner (Acct 1)
         │                        ├── ProductionScanner (Acct 2)
         ▼                        └── TridentScanner (Acct 3)
  FlowStateManager ──► OratsCacheLayer ──► ScalpSignalScorer
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import structlog
from dotenv import load_dotenv

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.data_feeds.alpaca_execution import AlpacaExecutor
from flowedge.scanner.data_feeds.polygon_intraday import PolygonIntradayProvider
from flowedge.scanner.data_feeds.ws_bars import WebSocketBarProvider
from flowedge.scanner.flux.flow_state import FlowStateManager
from flowedge.scanner.flux.ws_consumer import MassiveDataFeed
from flowedge.scanner.providers.orats import OratsProvider
from flowedge.scanner.providers.orats_cache import OratsCacheLayer

logger = structlog.get_logger()

# US Eastern timezone
ET = timezone(timedelta(hours=-4))  # EDT (April-October)

# All unique tickers across all scanners
SCALP_V2_CONFIG = "configs/sweep_best_90wr.json"

# Tickers from each scanner (union for WebSocket subscription)
PRODUCTION_TICKERS = ["AAPL", "IWM", "META", "PLTR", "QQQ", "SPY", "XLK"]
VOL_SCALP_TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "META",
    "GOOGL", "AMZN", "PLTR", "SOFI", "AMD", "XLK", "XLF", "COST", "WMT",
]
TRIDENT_TICKERS = ["SPY", "QQQ", "IWM"]


def _load_scalp_v2_tickers() -> list[str]:
    """Load ticker list from scalp_v2 config JSON."""
    import json
    cfg_path = Path(SCALP_V2_CONFIG)
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f).get("tickers", [])
    return ["META", "AMD", "MSTR", "SQ", "UNH", "RDDT", "BA", "IWM"]


# ── Error-isolated scanner runner ─────────────────────────────


async def _safe_run(
    name: str,
    coro_fn: object,
    max_restarts: int = 5,
) -> None:
    """Run a scanner with auto-restart on crash. One failure doesn't kill others."""
    restarts = 0
    while restarts < max_restarts:
        try:
            logger.info("scanner_starting", scanner=name, restart=restarts)
            await coro_fn()  # type: ignore[operator]
            logger.info("scanner_exited_cleanly", scanner=name)
            return
        except asyncio.CancelledError:
            logger.info("scanner_cancelled", scanner=name)
            return
        except Exception as e:
            restarts += 1
            logger.error(
                "scanner_crashed",
                scanner=name,
                error=str(e),
                restart=restarts,
                max=max_restarts,
            )
            if restarts < max_restarts:
                await asyncio.sleep(10)  # Brief pause before restart

    logger.error("scanner_max_restarts", scanner=name, restarts=restarts)


# ── ORATS enrichment loop ─────────────────────────────────────


async def _orats_enrichment_loop(
    orats: OratsCacheLayer,
    flow_mgr: FlowStateManager,
    interval: float = 60.0,
) -> None:
    """Background: refresh ORATS data for tickers with active flow signals."""
    logger.info("orats_enrichment_loop_started", interval=interval)
    while True:
        try:
            for ticker, state in flow_mgr.get_all_states().items():
                if state.triggers.any_active:
                    await orats.get_live_enrichment(ticker)

            stats = orats.get_stats()
            if stats["total_api_calls"] > 0 and stats["total_api_calls"] % 50 == 0:
                logger.info("orats_cache_stats", **stats)

        except Exception as e:
            logger.warning("orats_enrichment_error", error=str(e))

        await asyncio.sleep(interval)


# ── Main orchestrator ─────────────────────────────────────────


async def run_unified_options() -> None:
    """Single process running all 4 options scanners with shared data layer.

    1. ONE MassiveDataFeed (WebSocket) for all tickers
    2. ONE OratsCacheLayer with trigger-gated enrichment
    3. ONE FlowStateManager tracking per-ticker flow
    4. Each scanner gets shared providers, runs its own scan loop
    """
    load_dotenv()
    settings = get_settings()

    # ── API Keys ──────────────────────────────────────────────
    polygon_key = os.getenv("POLYGON_API_KEY", "")
    if not polygon_key:
        logger.error("missing_polygon_key")
        return

    # Account 1: scalp_v2 + vol_scalp_v1 (shared)
    acct1_key = os.getenv("ALPACA_API_KEY_ID", "")
    acct1_secret = os.getenv("ALPACA_API_SECRET_KEY", "")

    # Account 2: production scanner
    acct2_key = os.getenv("ALPACA_PROD_KEY_ID", "")
    acct2_secret = os.getenv("ALPACA_PROD_SECRET_KEY", "")

    # Account 3: trident
    acct3_key = os.getenv("TRIDENT_ALPACA_KEY_ID", "")
    acct3_secret = os.getenv("TRIDENT_ALPACA_SECRET_KEY", "")

    # ── Union of all tickers ──────────────────────────────────
    scalp_v2_tickers = _load_scalp_v2_tickers()
    all_tickers = sorted(set(
        scalp_v2_tickers
        + VOL_SCALP_TICKERS
        + PRODUCTION_TICKERS
        + TRIDENT_TICKERS
    ))

    # ── Shared data layer ─────────────────────────────────────

    # 1. ONE MassiveDataFeed (WebSocket)
    data_feed: MassiveDataFeed | None = None
    if settings.flux_use_websocket and polygon_key:
        data_feed = MassiveDataFeed(
            api_key=polygon_key,
            tickers=all_tickers,
            ws_url=settings.flux_ws_url,
        )
        await data_feed.start()
        polygon = WebSocketBarProvider(data_feed, fallback_api_key=polygon_key)
        data_mode = "WebSocket"
        logger.info(
            "websocket_started",
            tickers=len(all_tickers),
            url=settings.flux_ws_url,
        )
    else:
        polygon = PolygonIntradayProvider(polygon_key)
        data_mode = "REST"
        logger.info("using_rest_polling")

    # 2. ONE OratsCacheLayer
    orats_cache: OratsCacheLayer | None = None
    if settings.orats_api_key:
        orats_provider = OratsProvider(settings)
        orats_cache = OratsCacheLayer(orats_provider, settings)
        logger.info("orats_cache_initialized")
    else:
        logger.warning("no_orats_key_set")

    # 3. ONE FlowStateManager
    flow_mgr: FlowStateManager | None = None
    if data_feed:
        flow_mgr = FlowStateManager(data_feed, settings)
        flow_mgr.set_tickers(all_tickers)

    # ── Create Alpaca executors ───────────────────────────────

    alpaca_acct1 = AlpacaExecutor(acct1_key, acct1_secret, paper=True) if acct1_key else None
    alpaca_acct2 = AlpacaExecutor(acct2_key, acct2_secret, paper=True) if acct2_key else None
    alpaca_acct3 = AlpacaExecutor(acct3_key, acct3_secret, paper=True) if acct3_key else None

    # ── Create scanners with SHARED providers ─────────────────

    tasks: list[tuple[str, object]] = []

    # Scalp V2
    if alpaca_acct1:
        from flowedge.scanner.live.scalp_v2_scanner import create_scanner as create_scalp_v2
        scalp_v2 = create_scalp_v2(polygon, alpaca_acct1, SCALP_V2_CONFIG)
        tasks.append(("scalp_v2", scalp_v2.run))

    # Volume Scalper V1
    if alpaca_acct1:
        from flowedge.scanner.live.volume_scalper_v1_scanner import create_scanner as create_vol_scalp
        vol_scalp = create_vol_scalp(polygon, alpaca_acct1)
        tasks.append(("vol_scalp_v1", vol_scalp.run))

    # Production Scanner
    if alpaca_acct2:
        from flowedge.scanner.live.scanner import create_scanner as create_production
        flux_consumer = data_feed if data_feed else None
        if not flux_consumer and polygon_key:
            from flowedge.scanner.flux.consumer import PolygonTradeConsumer
            flux_consumer = PolygonTradeConsumer(polygon_key)
        production = create_production(polygon, alpaca_acct2, flux_consumer)
        tasks.append(("production", production.run))

    # Trident
    if alpaca_acct3:
        from flowedge.scanner.live.trident_scanner import TridentScanner
        trident = TridentScanner(polygon=polygon, alpaca=alpaca_acct3, data_feed=data_feed)
        tasks.append(("trident", trident.run))

    # ── Pre-market warmup ─────────────────────────────────────

    if orats_cache:
        await orats_cache.warm_cache(all_tickers)

    # ── Startup banner ────────────────────────────────────────

    print("=" * 70)
    print("FLOWEDGE UNIFIED OPTIONS SCANNER — DUAL SOURCE ARCHITECTURE")
    print("=" * 70)
    print(f"Data:       {data_mode} → {len(all_tickers)} tickers")
    print(f"ORATS:      {'Enabled (cached)' if orats_cache else 'Disabled'}")
    print(f"FlowState:  {'Enabled (30s updates)' if flow_mgr else 'Disabled'}")
    print(f"Scanners:   {len(tasks)} active")
    for name, _ in tasks:
        print(f"  - {name}")
    accounts = []
    if alpaca_acct1:
        accounts.append("Acct1 (scalp_v2 + vol_scalp)")
    if alpaca_acct2:
        accounts.append("Acct2 (production)")
    if alpaca_acct3:
        accounts.append("Acct3 (trident)")
    print(f"Accounts:   {', '.join(accounts)}")
    print("=" * 70)

    # ── Run everything concurrently ───────────────────────────

    try:
        coros = [_safe_run(name, fn) for name, fn in tasks]

        # Add background services
        if flow_mgr:
            coros.append(_safe_run("flow_state", flow_mgr.update_loop))
        if orats_cache and flow_mgr:
            coros.append(_safe_run(
                "orats_enrichment",
                lambda: _orats_enrichment_loop(orats_cache, flow_mgr),
            ))

        await asyncio.gather(*coros)

    except KeyboardInterrupt:
        print("\nUnified scanner stopped.")
    finally:
        if data_feed:
            await data_feed.close()
        for name, _ in tasks:
            logger.info("scanner_shutdown", scanner=name)
