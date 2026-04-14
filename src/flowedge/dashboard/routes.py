"""Dashboard API routes — serves both HTML pages and JSON endpoints."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from flowedge.council.daily_review import (
    get_review_trends,
    list_reviews,
    load_review,
    run_daily_review,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()

# ── Alpaca account configs (loaded from env) ────────────────────
# Account 1: Scalp v2 (.env)
# Account 2: Legacy production models (.env.production)
_ACCOUNTS: dict[str, dict[str, str]] = {}


def _get_account_config(account_id: str) -> dict[str, str]:
    """Get Alpaca credentials for a given account ID.

    Account 1 = scalp v2 + vol scalper (from .env)
    Account 2 = legacy models (from .env: ALPACA_PROD_KEY_ID / ALPACA_PROD_SECRET_KEY)
    Account 3 = Trident (from .env: TRIDENT_ALPACA_KEY_ID / TRIDENT_ALPACA_SECRET_KEY)

    Falls back to loading .env.production if prod keys not in env.
    """
    if account_id == "1":
        return {
            "key": os.getenv("ALPACA_API_KEY_ID", ""),
            "secret": os.getenv("ALPACA_API_SECRET_KEY", ""),
            "label": "Scalp v2 + Vol Scalper v1",
        }
    if account_id == "2":
        # Try explicit prod env vars first, then fallback to .env.production file
        key = os.getenv("ALPACA_PROD_KEY_ID", "")
        secret = os.getenv("ALPACA_PROD_SECRET_KEY", "")
        if not key or not secret:
            # Load from .env.production file
            prod_env = Path(".env.production")
            if prod_env.exists():
                for line in prod_env.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() == "ALPACA_API_KEY_ID":
                        key = v.strip()
                    elif k.strip() == "ALPACA_API_SECRET_KEY":
                        secret = v.strip()
        return {
            "key": key,
            "secret": secret,
            "label": "Legacy Models",
        }
    if account_id == "3":
        return {
            "key": os.getenv("TRIDENT_ALPACA_KEY_ID", ""),
            "secret": os.getenv("TRIDENT_ALPACA_SECRET_KEY", ""),
            "label": "Trident",
        }
    return {"key": "", "secret": "", "label": "Unknown"}


async def _fetch_alpaca_data(
    account_id: str,
) -> dict[str, Any]:
    """Fetch account, positions, and orders from an Alpaca account."""
    from flowedge.scanner.data_feeds.alpaca_execution import AlpacaExecutor

    config = _get_account_config(account_id)
    if not config["key"] or not config["secret"]:
        return {
            "error": f"Account {account_id} not configured",
            "account": {}, "positions": [], "orders": [],
        }

    executor = AlpacaExecutor(config["key"], config["secret"], paper=True)
    try:
        account = await executor.get_account()
        positions = await executor.get_positions()
        # Fetch today's orders
        client = await executor._ensure_client()
        resp = await client.get(
            f"{executor._base_url}/v2/orders",
            params={"status": "all", "limit": "50", "direction": "desc"},
        )
        orders_raw = resp.json() if resp.status_code < 400 else []

        return {
            "account": account,
            "positions": [
                {
                    "ticker": p.ticker,
                    "qty": p.qty,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                }
                for p in positions
            ],
            "orders": [
                {
                    "id": o.get("id", ""),
                    "symbol": o.get("symbol", ""),
                    "side": o.get("side", ""),
                    "qty": o.get("qty", ""),
                    "status": o.get("status", ""),
                    "submitted_at": o.get("submitted_at", ""),
                    "filled_avg_price": o.get("filled_avg_price"),
                    "limit_price": o.get("limit_price"),
                    "client_order_id": o.get("client_order_id", ""),
                }
                for o in (orders_raw if isinstance(orders_raw, list) else [])
            ],
        }
    except Exception as e:
        return {"error": str(e), "account": {}, "positions": [], "orders": []}
    finally:
        await executor.close()


# ── HTML Pages ───────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page showing the latest council review."""
    reviews = list_reviews()
    latest_review = None
    if reviews:
        with contextlib.suppress(Exception):
            latest_review = load_review(reviews[0])

    trends = get_review_trends(limit=30)
    # Serialize trends to JSON string for safe Jinja2 rendering
    trends_json = json.dumps(
        [json.loads(t.model_dump_json()) for t in trends],
        default=str,
    )

    # Pre-serialize config for safe Jinja2 rendering
    config_json = ""
    if latest_review and latest_review.config_used:
        config_json = json.dumps(latest_review.config_used, indent=2, default=str)

    # Starlette 1.0: TemplateResponse(request, name, context)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "review": latest_review,
            "trends": trends,
            "trends_json": Markup(trends_json),
            "config_json": config_json,
            "review_count": len(reviews),
        },
    )


@router.get("/review/{review_id}", response_class=HTMLResponse)
async def review_detail(request: Request, review_id: str):
    """Detailed view of a specific council review."""
    for path in list_reviews():
        if review_id in path.name:
            review = load_review(path)
            config_json = json.dumps(review.config_used, indent=2, default=str)
            return templates.TemplateResponse(
                request,
                "review_detail.html",
                {"review": review, "config_json": config_json},
            )
    raise HTTPException(status_code=404, detail=f"Review {review_id} not found")


@router.get("/history", response_class=HTMLResponse)
async def review_history(request: Request):
    """History of all council reviews."""
    reviews = []
    for path in list_reviews()[:50]:
        with contextlib.suppress(Exception):
            reviews.append(load_review(path))

    return templates.TemplateResponse(
        request,
        "history.html",
        {"reviews": reviews},
    )


# ── JSON API Endpoints ──────────────────────────────────────────

@router.get("/api/reviews", response_class=JSONResponse)
async def api_list_reviews():
    """List all reviews as JSON."""
    reviews = []
    for path in list_reviews()[:30]:
        try:
            r = load_review(path)
            reviews.append({
                "review_id": r.review_id,
                "review_date": r.review_date.isoformat(),
                "status": r.status.value,
                "overall_health": r.overall_health,
                "trades": r.cumulative_trades,
                "win_rate": r.cumulative_wr,
                "pnl": r.cumulative_pnl,
                "recommendations": len(r.top_recommendations),
            })
        except Exception:
            pass
    return JSONResponse(content=reviews)


@router.get("/api/review/{review_id}", response_class=JSONResponse)
async def api_review_detail(review_id: str):
    """Get a specific review as JSON."""
    for path in list_reviews():
        if review_id in path.name:
            review = load_review(path)
            return JSONResponse(content=json.loads(review.model_dump_json()))
    raise HTTPException(status_code=404, detail="Review not found")


@router.get("/api/trends", response_class=JSONResponse)
async def api_trends():
    """Get review trend data for charts."""
    trends = get_review_trends(limit=30)
    return JSONResponse(
        content=[json.loads(t.model_dump_json()) for t in trends]
    )


@router.post("/api/run-review", response_class=JSONResponse)
async def api_run_review(
    config_path: str | None = None,
    run_backtest: bool = False,
):
    """Trigger a new council review.

    By default loads the latest saved backtest result.
    Set run_backtest=true to run a fresh backtest first.
    """
    try:
        review = run_daily_review(
            config_path=config_path,
            run_backtest=run_backtest,
        )
        return JSONResponse(
            content={
                "review_id": review.review_id,
                "status": review.status.value,
                "health": review.overall_health,
                "recommendations": len(review.top_recommendations),
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


_flux_ws_consumer = None  # Module-level shared WebSocket consumer


@router.get("/api/flux", response_class=JSONResponse)
async def api_flux_signals():
    """Get FLUX order flow signals for all scanner tickers."""
    global _flux_ws_consumer
    from flowedge.scanner.flux.engine import scan_flux

    settings_mod = __import__("flowedge.config.settings", fromlist=["get_settings"])
    settings = settings_mod.get_settings()
    if not settings.polygon_api_key:
        return JSONResponse(content={"error": "Polygon API key not configured"})

    tickers = ["SPY", "QQQ", "IWM", "AAPL", "META", "PLTR", "XLK"]

    # Use WebSocket consumer (shared, persistent) if enabled
    if settings.flux_use_websocket:
        from flowedge.scanner.flux.ws_consumer import MassiveWebSocketConsumer

        if _flux_ws_consumer is None or not _flux_ws_consumer.is_connected:
            _flux_ws_consumer = MassiveWebSocketConsumer(
                api_key=settings.polygon_api_key,
                tickers=tickers,
                ws_url=settings.flux_ws_url,
            )
            await _flux_ws_consumer.start()
            # Allow buffer to fill briefly on first call
            import asyncio
            await asyncio.sleep(2)

        signals = await scan_flux(_flux_ws_consumer, tickers, settings)
    else:
        from flowedge.scanner.flux.consumer import PolygonTradeConsumer
        consumer = PolygonTradeConsumer(settings.polygon_api_key)
        try:
            signals = await scan_flux(consumer, tickers, settings)
        finally:
            await consumer.close()

    return JSONResponse(content=[
        {
            "ticker": s.ticker,
            "strength": s.strength,
            "bias": s.bias.value,
            "divergence": s.divergence.value,
            "aggression": s.delta_5m.aggression_ratio if s.delta_5m else 0.5,
            "net_delta": s.delta_5m.net_delta if s.delta_5m else 0,
            "blocks": len(s.block_prints),
            "quote_imbalance": (
                s.quote_imbalance.avg_imbalance if s.quote_imbalance else 0.0
            ),
            "rationale": s.rationale,
        }
        for s in signals
    ])


@router.get("/api/health", response_class=JSONResponse)
async def api_health():
    """Health check endpoint."""
    reviews = list_reviews()
    return JSONResponse(
        content={
            "status": "ok",
            "reviews_count": len(reviews),
            "latest_review": reviews[0].name if reviews else None,
        }
    )


# ── Live P&L Page + API ───────────────────────────────────────────

@router.get("/live", response_class=HTMLResponse)
async def live_pnl_page(request: Request):
    """Live P&L dashboard showing both paper trading accounts."""
    return templates.TemplateResponse(request, "live_pnl.html", {})


@router.get("/api/live/account/{account_id}", response_class=JSONResponse)
async def api_live_account(account_id: str):
    """Fetch live account data, positions, and orders for a specific account.

    account_id: "1" = Scalp v2, "2" = Legacy production models
    """
    if account_id not in ("1", "2", "3"):
        raise HTTPException(status_code=400, detail="account_id must be 1, 2, or 3")
    data = await _fetch_alpaca_data(account_id)
    return JSONResponse(content=data)
