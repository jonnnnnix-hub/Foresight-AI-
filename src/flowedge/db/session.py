"""Async database session factory."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from flowedge.config.settings import get_settings


def get_engine() -> AsyncEngine:
    """Create async engine from settings."""
    settings = get_settings()
    return create_async_engine(settings.database_url, echo=False)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return an async session factory."""
    engine = get_engine()
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session
