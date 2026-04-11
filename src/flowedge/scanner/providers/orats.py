"""Orats provider — IV rank, historical IV, live strikes, and expected moves."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from flowedge.config.settings import Settings
from flowedge.scanner.providers.base import IVDataProvider
from flowedge.scanner.schemas.catalyst import ExpectedMove
from flowedge.scanner.schemas.iv import IVRankData, TermStructurePoint


class OratsProvider(IVDataProvider):
    """Orats IV data, live strikes, and earnings expected move provider.

    Leverages the full ORATS API including:
    - datav2/ivrank — IV rank and percentile
    - datav2/summaries — IV across tenors
    - datav2/cores — ATM IV, greeks summary, market data
    - datav2/strikes — EOD per-strike data with full greeks
    - datav2/monies/implied — IV by expiration
    - datav2/hist/earnings — historical earnings and expected moves
    - datav2/live/strikes — real-time intraday option strikes
    - datav2/live/summaries — real-time IV summaries
    """

    name = "orats"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._base_url = settings.orats_base_url
        self._token = settings.orats_api_key

    def _params(self, **kwargs: str) -> dict[str, str]:
        """Build params dict with token authentication."""
        return {"token": self._token, **kwargs}

    async def health_check(self) -> bool:
        try:
            await self._get(
                f"{self._base_url}/datav2/ivrank",
                params=self._params(ticker="AAPL"),
            )
            return True
        except Exception:
            return False

    async def get_iv_rank(self, ticker: str) -> IVRankData:
        """Fetch current IV rank and percentile."""
        data = await self._get(
            f"{self._base_url}/datav2/ivrank",
            params=self._params(ticker=ticker),
        )

        results = data.get("data", [])
        if not results:
            return IVRankData(ticker=ticker, iv_rank=0.0)

        row = results[0] if isinstance(results, list) else results

        current_iv = float(row.get("iv", row.get("ivMean", 0)))
        hv_20 = row.get("hv20", row.get("orHv20d"))
        iv_rank_1y = float(row.get("ivRank1y", row.get("ivRank", 0)))
        iv_pct_1y = float(row.get("ivPct1y", row.get("ivPct", 0)))

        return IVRankData(
            ticker=ticker,
            iv_rank=iv_rank_1y * 100 if iv_rank_1y <= 1.0 else iv_rank_1y,
            iv_percentile=iv_pct_1y * 100 if iv_pct_1y <= 1.0 else iv_pct_1y,
            current_iv=current_iv,
            hv_20=float(hv_20) if hv_20 is not None else None,
            hv_60=float(row["hv60"]) if row.get("hv60") is not None else None,
            iv_hv_spread=(
                round(current_iv - float(hv_20), 4)
                if hv_20 is not None
                else None
            ),
            iv_52w_high=float(row.get("ivHigh1y", 0)),
            iv_52w_low=float(row.get("ivLow1y", 0)),
            fetched_at=datetime.now(),
            source="orats",
        )

    async def get_historical_iv(
        self, ticker: str, days: int = 252
    ) -> list[TermStructurePoint]:
        """Fetch IV term structure from Orats summaries."""
        data = await self._get(
            f"{self._base_url}/datav2/summaries",
            params=self._params(ticker=ticker),
        )

        points: list[TermStructurePoint] = []
        results = data.get("data", [])

        if results:
            row = results[0] if isinstance(results, list) else results
            tenor_map = {
                "iv10d": 10,
                "iv20d": 20,
                "iv30d": 30,
                "iv60d": 60,
                "iv90d": 90,
                "iv6m": 180,
                "iv1y": 365,
            }
            today = date.today()
            for field, dte in tenor_map.items():
                val = row.get(field)
                if val is not None:
                    points.append(
                        TermStructurePoint(
                            expiration=today + timedelta(days=dte),
                            iv=float(val),
                            days_to_expiration=dte,
                        )
                    )

        return sorted(points, key=lambda p: p.days_to_expiration)

    async def get_expected_move(
        self, ticker: str
    ) -> ExpectedMove | None:
        """Fetch expected earnings move from historical earnings data."""
        data = await self._get(
            f"{self._base_url}/datav2/hist/earnings",
            params=self._params(ticker=ticker),
        )

        results = data.get("data", [])
        if not results:
            return None

        latest = results[-1] if isinstance(results, list) else results
        earn_date_str = latest.get("earnDate", "")
        if not earn_date_str:
            return None

        try:
            earn_date = date.fromisoformat(earn_date_str)
        except ValueError:
            return None

        return ExpectedMove(
            ticker=ticker,
            event_date=earn_date,
            expected_move_pct=float(latest.get("straddlePct", 0)),
            expected_move_dollars=float(latest.get("straddleUsd", 0)),
            straddle_price=float(latest.get("straddlePrice", 0)),
            source="orats",
        )

    # ---- Premium ORATS endpoints ($399/mo plan) ----

    async def get_live_strikes(
        self, ticker: str
    ) -> list[dict[str, Any]]:
        """Fetch real-time intraday option strikes.

        Returns per-strike: bid, ask, mid IV, greeks, volume, OI.
        """
        data = await self._get(
            f"{self._base_url}/datav2/live/strikes",
            params=self._params(ticker=ticker),
        )
        results = data.get("data", [])
        return results if isinstance(results, list) else []

    async def get_live_summary(self, ticker: str) -> dict[str, Any]:
        """Fetch real-time intraday IV summary."""
        data = await self._get(
            f"{self._base_url}/datav2/live/summaries",
            params=self._params(ticker=ticker),
        )
        results = data.get("data", [])
        if results and isinstance(results, list):
            result: dict[str, Any] = results[0]
            return result
        return {}

    async def get_cores(self, ticker: str) -> dict[str, Any]:
        """Fetch core metrics — ATM IV, market cap, greeks summary."""
        data = await self._get(
            f"{self._base_url}/datav2/cores",
            params=self._params(ticker=ticker),
        )
        results = data.get("data", [])
        if results and isinstance(results, list):
            result: dict[str, Any] = results[0]
            return result
        return {}

    async def get_strikes(self, ticker: str) -> list[dict[str, Any]]:
        """Fetch EOD per-strike data with full greeks and pricing."""
        data = await self._get(
            f"{self._base_url}/datav2/strikes",
            params=self._params(ticker=ticker),
        )
        results = data.get("data", [])
        return results if isinstance(results, list) else []

    async def get_monies_implied(
        self, ticker: str
    ) -> list[dict[str, Any]]:
        """Fetch implied moneyness data by expiration."""
        data = await self._get(
            f"{self._base_url}/datav2/monies/implied",
            params=self._params(ticker=ticker),
        )
        results = data.get("data", [])
        return results if isinstance(results, list) else []

    async def get_historical_earnings(
        self, ticker: str
    ) -> list[dict[str, Any]]:
        """Fetch full historical earnings data with moves and straddle pricing."""
        data = await self._get(
            f"{self._base_url}/datav2/hist/earnings",
            params=self._params(ticker=ticker),
        )
        results = data.get("data", [])
        return results if isinstance(results, list) else []
