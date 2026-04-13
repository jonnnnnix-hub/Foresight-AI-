"""FastAPI admin dashboard for the Council review system.

Run:
    uvicorn flowedge.dashboard.app:app --reload --port 8050
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from flowedge.dashboard.routes import router

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(
    title="FlowEdge Council Dashboard",
    description="Post-market specialist review panel for scalp model oversight",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(router)
