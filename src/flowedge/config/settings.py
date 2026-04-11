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

    # Scanner provider API keys
    polygon_api_key: str = Field(default="")
    tradier_api_key: str = Field(default="")
    tradier_base_url: str = Field(default="https://api.tradier.com")
    alpaca_api_key_id: str = Field(default="")
    alpaca_api_secret_key: str = Field(default="")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets")
    unusual_whales_api_key: str = Field(default="")
    unusual_whales_base_url: str = Field(
        default="https://api.unusualwhales.com"
    )
    orats_api_key: str = Field(default="")
    orats_base_url: str = Field(default="https://api.orats.io")
    fmp_api_key: str = Field(default="")
    fmp_base_url: str = Field(default="https://financialmodelingprep.com")
    sec_edgar_user_agent: str = Field(
        default="FlowEdge/1.0 (contact@flowedge.io)"
    )
    taapi_api_key: str = Field(default="")
    taapi_base_url: str = Field(default="https://api.taapi.io")

    # Scanner tuning
    scanner_http_timeout: float = Field(default=30.0)
    uoa_volume_oi_threshold: float = Field(default=3.0)
    uoa_min_premium: float = Field(default=25_000.0)
    iv_rank_low_threshold: float = Field(default=30.0)
    iv_rank_high_threshold: float = Field(default=70.0)
    catalyst_lookforward_days: int = Field(default=14)
    catalyst_lookback_insider_days: int = Field(default=90)
    lotto_score_uoa_weight: float = Field(default=0.35)
    lotto_score_iv_weight: float = Field(default=0.30)
    lotto_score_catalyst_weight: float = Field(default=0.35)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
