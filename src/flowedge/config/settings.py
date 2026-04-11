"""Application settings loaded from environment."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    anthropic_api_key: str = Field(default="")
    database_url: str = Field(
        default="postgresql+asyncpg://flowedge:flowedge@localhost:5432/flowedge"
    )
    log_level: str = Field(default="INFO")
    clone_base_dir: Path = Field(default=Path("./repos"))
    max_file_size_kb: int = Field(default=512)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
