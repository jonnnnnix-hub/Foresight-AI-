"""Application settings loaded from environment."""

from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings.

    Reads from .env file, with a workaround for empty env vars
    in the shell overriding non-empty .env file values.
    """

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @model_validator(mode="before")
    @classmethod
    def _fill_empty_from_dotenv(cls, values: dict) -> dict:  # type: ignore[type-arg]
        """If an env var is empty string but .env has a value, use .env."""
        try:
            dotenv = dotenv_values(".env")
        except Exception:
            return values
        for key, file_val in dotenv.items():
            lower_key = key.lower()
            if lower_key in values and values[lower_key] == "" and file_val:
                values[lower_key] = file_val
        return values

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

    # Massive S3 flat files
    massive_access_key: str = Field(default="")
    massive_secret_key: str = Field(default="")
    massive_endpoint: str = Field(default="https://files.massive.com")
    massive_bucket: str = Field(default="flatfiles")

    # Scanner tuning
    scanner_http_timeout: float = Field(default=30.0)
    uoa_volume_oi_threshold: float = Field(default=3.0)
    uoa_min_premium: float = Field(default=25_000.0)
    iv_rank_low_threshold: float = Field(default=30.0)
    iv_rank_high_threshold: float = Field(default=70.0)
    catalyst_lookforward_days: int = Field(default=14)
    catalyst_lookback_insider_days: int = Field(default=90)
    lotto_score_uoa_weight: float = Field(default=0.30)
    lotto_score_iv_weight: float = Field(default=0.25)
    lotto_score_catalyst_weight: float = Field(default=0.25)
    lotto_score_flux_weight: float = Field(default=0.20)

    # FLUX engine tuning
    flux_block_min_multiple: float = Field(
        default=10.0,
        description="Block print threshold: trades > N * avg size",
    )
    flux_aggression_strong: float = Field(
        default=0.65,
        description="Aggression ratio threshold for strong signal",
    )
    flux_divergence_min_pct: float = Field(
        default=0.05,
        description="Minimum delta fraction of volume to count",
    )
    flux_use_websocket: bool = Field(
        default=True,
        description="Use Massive WebSocket for live data (single shared connection)",
    )
    flux_ws_url: str = Field(
        default="wss://delayed.polygon.io/stocks",
        description="Polygon stocks WebSocket URL (delayed on Developer plan)",
    )
    flux_ws_delayed_url: str = Field(
        default="wss://delayed.polygon.io/stocks",
        description="Polygon 15-min delayed WebSocket URL",
    )

    # ORATS cache tuning
    orats_cache_cores_ttl: int = Field(
        default=300,
        description="TTL for ORATS cores/summaries/ivrank cache (seconds)",
    )
    orats_cache_strikes_ttl: int = Field(
        default=45,
        description="TTL for ORATS live strikes/summaries cache (seconds)",
    )
    orats_trigger_min_flux_strength: float = Field(
        default=5.0,
        description="Minimum FLUX strength to trigger ORATS enrichment",
    )

    # Scalp signal scorer
    scalp_scorer_entry_min: float = Field(
        default=6.5,
        description="Minimum composite score for scalp entry (0-10)",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
