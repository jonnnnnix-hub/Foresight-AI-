"""ORATS Cache Layer — TTL-based caching with trigger-gated enrichment.

Wraps OratsProvider to minimize API calls ($399/mo plan) while keeping
data fresh enough for real-time scalp decisions.

Cache tiers:
  - cores / summaries / ivrank: 300s (changes slowly)
  - live_strikes / live_summaries: 45s (real-time intraday)

Trigger gating: Only calls ORATS when FLUX detects meaningful flow
(strength >= threshold, volume spike, block prints, or quote imbalance).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.flux.schemas import FLUXSignal

logger = structlog.get_logger()

# ── Cache TTL defaults (seconds) ──────────────────────────────

_TTL_SLOW = 300     # Cores, summaries, IV rank
_TTL_FAST = 45      # Live strikes, live summaries
_RATE_LIMIT = 10    # Max ORATS API calls per minute


# ── ORATSContext: typed enrichment output ─────────────────────


@dataclass
class ORATSContext:
    """IV surface + analytics for a single underlying at a point in time."""

    ticker: str
    fetched_at: float = 0.0  # time.monotonic()

    # IV surface
    atm_iv: float = 0.0
    iv_rank: float = 0.0           # 0-100
    iv_percentile: float = 0.0     # 0-100
    iv_slope: float = 0.0          # Skew from summaries

    # Historical vol context
    hv_20d: float = 0.0
    hv_60d: float = 0.0
    iv_hv_spread: float = 0.0     # IV - HV (positive = vol premium)

    # Term structure
    iv_10d: float = 0.0
    iv_30d: float = 0.0
    iv_60d: float = 0.0
    contango: float = 0.0         # iv_30d - iv_10d (positive = normal)

    # Earnings
    days_to_earnings: int | None = None
    in_earnings_blackout: bool = False  # Within 2 days of earnings

    # Per-strike data (populated on trigger)
    live_strikes: list[dict[str, Any]] = field(default_factory=list)


# ── Cache entry ───────────────────────────────────────────────


@dataclass
class _CacheEntry:
    data: Any
    timestamp: float  # time.monotonic()


# ── OratsCacheLayer ───────────────────────────────────────────


class OratsCacheLayer:
    """Caching wrapper around OratsProvider with trigger gating.

    Usage:
        orats = OratsCacheLayer(OratsProvider(settings), settings)
        await orats.warm_cache(["SPY", "QQQ", ...])  # Pre-market

        # During market: get enrichment for a ticker
        ctx = await orats.get_enrichment("SPY")

        # Trigger-gated: only call live strikes if FLUX fires
        if orats.should_enrich(flux_signal):
            ctx = await orats.get_live_enrichment("SPY")
    """

    def __init__(self, provider: Any, settings: Settings | None = None) -> None:
        self._provider = provider
        self._settings = settings or get_settings()
        self._cache: dict[str, _CacheEntry] = {}

        # Rate limiting
        self._call_timestamps: list[float] = []
        self._total_calls = 0
        self._cache_hits = 0
        self._cache_misses = 0

        # TTL config
        self._ttl_slow = getattr(
            self._settings, "orats_cache_cores_ttl", _TTL_SLOW,
        )
        self._ttl_fast = getattr(
            self._settings, "orats_cache_strikes_ttl", _TTL_FAST,
        )
        self._trigger_min_strength = getattr(
            self._settings, "orats_trigger_min_flux_strength", 5.0,
        )

    # ── Cache helpers ─────────────────────────────────────────

    def _cache_key(self, endpoint: str, ticker: str) -> str:
        return f"{endpoint}:{ticker}"

    def _get_cached(self, key: str, ttl: float) -> Any | None:
        entry = self._cache.get(key)
        if entry and (time.monotonic() - entry.timestamp) < ttl:
            self._cache_hits += 1
            return entry.data
        return None

    def _set_cached(self, key: str, data: Any) -> None:
        self._cache[key] = _CacheEntry(data=data, timestamp=time.monotonic())

    def _rate_check(self) -> bool:
        """Return True if we can make another API call."""
        now = time.monotonic()
        # Prune calls older than 60s
        self._call_timestamps = [
            t for t in self._call_timestamps if now - t < 60
        ]
        return len(self._call_timestamps) < _RATE_LIMIT

    def _record_call(self) -> None:
        self._call_timestamps.append(time.monotonic())
        self._total_calls += 1
        self._cache_misses += 1

    # ── Trigger gating ────────────────────────────────────────

    def should_enrich(self, flux_signal: FLUXSignal | None) -> bool:
        """Return True if FLUX signal warrants ORATS enrichment.

        Triggers: high FLUX strength, block prints, or strong imbalance.
        """
        if flux_signal is None:
            return False
        if flux_signal.strength >= self._trigger_min_strength:
            return True
        if flux_signal.block_prints:
            return True
        if (
            flux_signal.quote_imbalance
            and abs(flux_signal.quote_imbalance.avg_imbalance) > 0.20
        ):
            return True
        return False

    # ── Data fetching with cache ──────────────────────────────

    async def _fetch_cores(self, ticker: str) -> dict[str, Any]:
        key = self._cache_key("cores", ticker)
        cached = self._get_cached(key, self._ttl_slow)
        if cached is not None:
            return cached
        if not self._rate_check():
            logger.debug("orats_rate_limited", ticker=ticker, endpoint="cores")
            return {}
        self._record_call()
        try:
            data = await self._provider.get_cores(ticker)
            self._set_cached(key, data)
            return data
        except Exception as e:
            logger.warning("orats_cores_failed", ticker=ticker, error=str(e))
            return {}

    async def _fetch_iv_rank(self, ticker: str) -> dict[str, Any]:
        key = self._cache_key("ivrank", ticker)
        cached = self._get_cached(key, self._ttl_slow)
        if cached is not None:
            return cached
        if not self._rate_check():
            return {}
        self._record_call()
        try:
            data = await self._provider.get_iv_rank(ticker)
            result = {
                "iv_rank": data.iv_rank,
                "iv_percentile": data.iv_percentile,
                "current_iv": data.current_iv,
                "hv_20": data.hv_20,
                "hv_60": data.hv_60,
                "iv_hv_spread": data.iv_hv_spread,
            }
            self._set_cached(key, result)
            return result
        except Exception as e:
            logger.warning("orats_ivrank_failed", ticker=ticker, error=str(e))
            return {}

    async def _fetch_summaries(self, ticker: str) -> dict[str, Any]:
        key = self._cache_key("summaries", ticker)
        cached = self._get_cached(key, self._ttl_slow)
        if cached is not None:
            return cached
        if not self._rate_check():
            return {}
        self._record_call()
        try:
            points = await self._provider.get_historical_iv(ticker, days=252)
            result: dict[str, Any] = {}
            for pt in points:
                if pt.days_to_expiration == 10:
                    result["iv_10d"] = pt.iv
                elif pt.days_to_expiration == 30:
                    result["iv_30d"] = pt.iv
                elif pt.days_to_expiration == 60:
                    result["iv_60d"] = pt.iv
            self._set_cached(key, result)
            return result
        except Exception as e:
            logger.warning("orats_summaries_failed", ticker=ticker, error=str(e))
            return {}

    async def _fetch_live_strikes(self, ticker: str) -> list[dict[str, Any]]:
        key = self._cache_key("live_strikes", ticker)
        cached = self._get_cached(key, self._ttl_fast)
        if cached is not None:
            return cached
        if not self._rate_check():
            return []
        self._record_call()
        try:
            data = await self._provider.get_live_strikes(ticker)
            self._set_cached(key, data)
            return data
        except Exception as e:
            logger.warning("orats_live_strikes_failed", ticker=ticker, error=str(e))
            return []

    async def _fetch_earnings(self, ticker: str) -> dict[str, Any]:
        key = self._cache_key("earnings", ticker)
        cached = self._get_cached(key, self._ttl_slow)
        if cached is not None:
            return cached
        if not self._rate_check():
            return {}
        self._record_call()
        try:
            move = await self._provider.get_expected_move(ticker)
            result: dict[str, Any] = {}
            if move:
                days = (move.event_date - date.today()).days
                result["days_to_earnings"] = days
                result["in_blackout"] = 0 <= days <= 2
            self._set_cached(key, result)
            return result
        except Exception as e:
            logger.warning("orats_earnings_failed", ticker=ticker, error=str(e))
            return {}

    # ── Public API ────────────────────────────────────────────

    async def get_enrichment(self, ticker: str) -> ORATSContext:
        """Get ORATS context using cached slow-tier data.

        Safe to call frequently — only makes API calls on cache miss.
        """
        cores = await self._fetch_cores(ticker)
        iv_data = await self._fetch_iv_rank(ticker)
        summaries = await self._fetch_summaries(ticker)
        earnings = await self._fetch_earnings(ticker)

        iv_10d = summaries.get("iv_10d", 0.0)
        iv_30d = summaries.get("iv_30d", 0.0)

        return ORATSContext(
            ticker=ticker,
            fetched_at=time.monotonic(),
            atm_iv=float(iv_data.get("current_iv", 0)),
            iv_rank=float(iv_data.get("iv_rank", 0)),
            iv_percentile=float(iv_data.get("iv_percentile", 0)),
            hv_20d=float(iv_data.get("hv_20", 0) or 0),
            hv_60d=float(iv_data.get("hv_60", 0) or 0),
            iv_hv_spread=float(iv_data.get("iv_hv_spread", 0) or 0),
            iv_10d=iv_10d,
            iv_30d=iv_30d,
            iv_60d=summaries.get("iv_60d", 0.0),
            contango=iv_30d - iv_10d if iv_10d and iv_30d else 0.0,
            days_to_earnings=earnings.get("days_to_earnings"),
            in_earnings_blackout=earnings.get("in_blackout", False),
        )

    async def get_live_enrichment(self, ticker: str) -> ORATSContext:
        """Get ORATS context including fresh live strikes.

        Call this only when should_enrich() returns True.
        """
        ctx = await self.get_enrichment(ticker)
        ctx.live_strikes = await self._fetch_live_strikes(ticker)
        return ctx

    async def warm_cache(self, tickers: list[str]) -> None:
        """Pre-market cache warmup: fetch cores + IV rank for all tickers."""
        logger.info(
            "orats_cache_warming",
            tickers=len(tickers),
        )
        for ticker in tickers:
            await self._fetch_cores(ticker)
            await self._fetch_iv_rank(ticker)
            await self._fetch_earnings(ticker)

        logger.info(
            "orats_cache_warmed",
            tickers=len(tickers),
            api_calls=self._total_calls,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return cache performance stats for monitoring."""
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(self._cache_hits / total, 3) if total else 0.0,
            "total_api_calls": self._total_calls,
            "cache_entries": len(self._cache),
        }
