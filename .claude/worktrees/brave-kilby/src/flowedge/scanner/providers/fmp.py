"""FMP (Financial Modeling Prep) provider — earnings calendar."""

from __future__ import annotations

from datetime import date

from flowedge.config.settings import Settings
from flowedge.scanner.providers.base import EarningsProvider
from flowedge.scanner.schemas.catalyst import EarningsEvent


class FMPProvider(EarningsProvider):
    """Financial Modeling Prep earnings calendar provider."""

    name = "fmp"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._base_url = settings.fmp_base_url
        self._api_key = settings.fmp_api_key

    async def health_check(self) -> bool:
        try:
            today = date.today()
            await self._get(
                f"{self._base_url}/stable/earnings-calendar",
                params={
                    "from": today.isoformat(),
                    "to": today.isoformat(),
                    "apikey": self._api_key,
                },
            )
            return True
        except Exception:
            return False

    async def get_earnings_calendar(
        self, from_date: date, to_date: date
    ) -> list[EarningsEvent]:
        """Fetch earnings calendar from FMP."""
        data = await self._get(
            f"{self._base_url}/stable/earnings-calendar",
            params={
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "apikey": self._api_key,
            },
        )

        events: list[EarningsEvent] = []

        # FMP returns a list directly
        items = data if isinstance(data, list) else data.get("data", [])

        for item in items:
            if not isinstance(item, dict):
                continue
            date_str = item.get("date", "")
            try:
                report_date = date.fromisoformat(date_str) if date_str else None
            except ValueError:
                continue

            if report_date is None:
                continue

            # Determine time of day
            time_str = str(item.get("time", "")).lower()
            if "bmo" in time_str or "before" in time_str:
                tod = "bmo"
            elif "amc" in time_str or "after" in time_str:
                tod = "amc"
            else:
                tod = time_str or ""

            events.append(
                EarningsEvent(
                    ticker=item.get("symbol", ""),
                    report_date=report_date,
                    fiscal_quarter=item.get("fiscalDateEnding", ""),
                    eps_estimate=(
                        float(item["epsEstimated"])
                        if item.get("epsEstimated") is not None
                        else None
                    ),
                    revenue_estimate=(
                        float(item["revenueEstimated"])
                        if item.get("revenueEstimated") is not None
                        else None
                    ),
                    time_of_day=tod,
                    source="fmp",
                )
            )

        return events
