"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from flowedge.api.auth import require_api_key
from flowedge.api.routes import router
from flowedge.config.logging import setup_logging
from flowedge.config.settings import get_settings
from flowedge.scanner.api.routes import scanner_router
from flowedge.scanner.healthcheck import check_all_providers, log_provider_status
from flowedge.ui.charts import charts_router
from flowedge.ui.dashboard import dashboard_router
from flowedge.ui.performance import perf_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup health checks when the server starts."""
    statuses = await check_all_providers()
    log_provider_status(statuses)
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="FlowEdge NEXUS API",
        version="0.2.0",
        description="Options scanner and repo analysis platform",
        lifespan=lifespan,
        dependencies=[Depends(require_api_key)],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    app.include_router(scanner_router, prefix="/api/v1")
    app.include_router(dashboard_router)
    app.include_router(charts_router)
    app.include_router(perf_router)
    return app


app = create_app()
