"""FastAPI application factory."""

from fastapi import FastAPI

from flowedge.api.routes import router
from flowedge.config.logging import setup_logging
from flowedge.config.settings import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="FlowEdge Repo Intelligence API",
        version="0.1.0",
        description="Analyze GitHub repos for trading system evaluation",
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
