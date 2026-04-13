"""Match scalp signals to real OPRA option contracts.

Given an underlying price and nanosecond timestamp at signal time,
finds the nearest-ATM call option from cached OPRA minute bars and
returns its real OHLCV for entry, hold, and exit pricing.

No Black-Scholes.  No estimated spreads.  Real market data only.
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

CACHE_DIR = Path("data/flat_files_s3")

# Maximum timestamp gap (nanoseconds) when matching a signal to an
# option bar.  120 seconds covers the 5-min aggregation window.
TIMESTAMP_TOLERANCE_NS = 120_000_000_000  # 120 seconds


@dataclass
class MatchedContract:
    """A real option contract matched to a scalp signal."""

    contract_symbol: str
    underlying: str
    strike: float
    expiration: str  # ISO date string
    dte: int
    bars: list[dict[str, Any]] = field(default_factory=list)
    total_volume: int = 0


class OptionsMatcher:
    """Look up real option bars from the OPRA cache.

    Loads cached option data per-underlying per-day and provides
    contract matching and time-series lookup for the scalp backtest.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or CACHE_DIR
        # In-memory cache: (underlying, date_str) -> {contract: [bars]}
        self._day_cache: dict[
            tuple[str, str], dict[str, list[dict[str, Any]]]
        ] = {}

    # ── Load / Index ───────────────────────────────────────────

    def load_day(
        self,
        underlying: str,
        date_str: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Load and index one day of cached option bars.

        Returns:
            Dict mapping contract symbol to sorted list of bars.
        """
        key = (underlying, date_str)
        if key in self._day_cache:
            return self._day_cache[key]

        path = (
            self._cache_dir
            / underlying
            / "options_1min"
            / f"{underlying}_options_1min_{date_str}.json"
        )
        if not path.exists():
            self._day_cache[key] = {}
            return {}

        raw: list[dict[str, Any]] = json.loads(path.read_text())

        by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for bar in raw:
            sym = bar.get("contract", "")
            if sym:
                by_contract[sym].append(bar)

        # Sort each contract's bars by timestamp
        for sym in by_contract:
            by_contract[sym].sort(key=lambda b: int(b.get("ts", 0)))

        self._day_cache[key] = dict(by_contract)
        return self._day_cache[key]

    # ── Contract Matching ──────────────────────────────────────

    def find_best_contract(
        self,
        underlying: str,
        date_str: str,
        underlying_price: float,
        signal_ts_ns: int,
        max_dte: int = 5,
    ) -> MatchedContract | None:
        """Find the best near-ATM call contract at signal time.

        Selection criteria (in priority order):
        1. Must have a bar within ``TIMESTAMP_TOLERANCE_NS`` of signal.
        2. Closest strike to ``underlying_price`` (nearest-ATM).
        3. Higher total day volume breaks ties.

        Args:
            underlying: Ticker (e.g. ``"PLTR"``).
            date_str: ISO date string (e.g. ``"2025-03-15"``).
            underlying_price: Underlying close at signal bar.
            signal_ts_ns: Signal timestamp in nanoseconds.
            max_dte: Maximum days to expiration.

        Returns:
            ``MatchedContract`` with all day bars, or ``None``.
        """
        day_data = self.load_day(underlying, date_str)
        if not day_data:
            return None

        candidates: list[tuple[float, int, str, list[dict[str, Any]]]] = []

        for sym, bars in day_data.items():
            if not bars:
                continue

            # Extract contract metadata from first bar
            first = bars[0]
            strike = float(first.get("strike", 0))
            dte = int(first.get("dte", 999))
            cp = first.get("option_type", "C")

            if cp != "C":
                continue
            if dte < 0 or dte > max_dte:
                continue

            # Check for bar near signal time
            bar_at_signal = self.get_bar_at_time(bars, signal_ts_ns)
            if bar_at_signal is None:
                continue

            # Signal bar must have a non-zero price
            if float(bar_at_signal.get("c", 0)) <= 0:
                continue

            atm_distance = abs(strike - underlying_price)
            day_volume = sum(int(b.get("v", 0)) for b in bars)

            candidates.append((atm_distance, -day_volume, sym, bars))

        if not candidates:
            logger.debug(
                "no_matching_contract",
                underlying=underlying,
                date=date_str,
                price=underlying_price,
                contracts_checked=len(day_data),
            )
            return None

        # Sort: closest ATM first, then highest volume
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, neg_vol, best_sym, best_bars = candidates[0]

        first = best_bars[0]
        return MatchedContract(
            contract_symbol=best_sym,
            underlying=underlying,
            strike=float(first.get("strike", 0)),
            expiration=str(first.get("expiration", "")),
            dte=int(first.get("dte", 0)),
            bars=best_bars,
            total_volume=-neg_vol,
        )

    # ── Time-Series Lookup ─────────────────────────────────────

    @staticmethod
    def get_bar_at_time(
        bars: list[dict[str, Any]],
        target_ts_ns: int,
        tolerance_ns: int = TIMESTAMP_TOLERANCE_NS,
    ) -> dict[str, Any] | None:
        """Find the bar closest to ``target_ts_ns`` within tolerance.

        Uses binary search on the sorted bar list.
        """
        if not bars:
            return None

        timestamps = [int(b.get("ts", 0)) for b in bars]
        idx = bisect.bisect_left(timestamps, target_ts_ns)

        best: dict[str, Any] | None = None
        best_dist = tolerance_ns + 1

        for i in (idx - 1, idx):
            if 0 <= i < len(bars):
                dist = abs(timestamps[i] - target_ts_ns)
                if dist < best_dist:
                    best_dist = dist
                    best = bars[i]

        return best

    @staticmethod
    def get_bars_after(
        bars: list[dict[str, Any]],
        start_ts_ns: int,
        count: int,
    ) -> list[dict[str, Any]]:
        """Return up to ``count`` bars starting at or after ``start_ts_ns``."""
        timestamps = [int(b.get("ts", 0)) for b in bars]
        idx = bisect.bisect_left(timestamps, start_ts_ns)
        return bars[idx : idx + count]

    @staticmethod
    def aggregate_to_5min(
        bars_1min: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Aggregate 1-minute option bars into 5-minute OHLCV bars.

        Groups by 5-minute windows starting from the first bar's
        timestamp, preserving the first bar's timestamp as ``ts``.
        """
        if not bars_1min:
            return []

        window_ns = 5 * 60 * 1_000_000_000  # 5 minutes
        chunks: list[dict[str, Any]] = []
        current_chunk: list[dict[str, Any]] = []
        chunk_start = int(bars_1min[0].get("ts", 0))

        for bar in bars_1min:
            ts = int(bar.get("ts", 0))
            if ts >= chunk_start + window_ns and current_chunk:
                chunks.append(_merge_chunk(current_chunk))
                current_chunk = []
                chunk_start = ts

            current_chunk.append(bar)

        if current_chunk:
            chunks.append(_merge_chunk(current_chunk))

        return chunks


def _merge_chunk(bars: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge a list of 1-min bars into one 5-min bar."""
    return {
        "ts": bars[0].get("ts", ""),
        "contract": bars[0].get("contract", ""),
        "underlying": bars[0].get("underlying", ""),
        "strike": bars[0].get("strike", 0),
        "expiration": bars[0].get("expiration", ""),
        "option_type": bars[0].get("option_type", "C"),
        "dte": bars[0].get("dte", 0),
        "date": bars[0].get("date", ""),
        "o": float(bars[0].get("o", 0)),
        "h": max(float(b.get("h", 0)) for b in bars),
        "l": min(
            float(b.get("l", 0)) for b in bars if float(b.get("l", 0)) > 0
        )
        if any(float(b.get("l", 0)) > 0 for b in bars)
        else 0.0,
        "c": float(bars[-1].get("c", 0)),
        "v": sum(int(b.get("v", 0)) for b in bars),
    }
