"""API key authentication dependency for FastAPI."""

from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from flowedge.config.settings import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """Validate the X-API-Key header against the configured API_KEY.

    If no API_KEY is set in settings, authentication is skipped (dev mode).
    """
    settings = get_settings()
    if not settings.api_key:
        return ""
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key
