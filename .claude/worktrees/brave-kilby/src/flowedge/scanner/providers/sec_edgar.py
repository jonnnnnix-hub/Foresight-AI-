"""SEC EDGAR provider — Form 4 insider trade filings."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any
from xml.etree import ElementTree

import structlog

from flowedge.config.settings import Settings
from flowedge.scanner.providers.base import InsiderTradeProvider
from flowedge.scanner.schemas.catalyst import InsiderTrade

logger = structlog.get_logger()

# SEC rate limit: 10 requests per second
_SEC_SEMAPHORE = asyncio.Semaphore(10)


class SECEdgarProvider(InsiderTradeProvider):
    """SEC EDGAR Form 4 insider trade provider."""

    name = "sec_edgar"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._base_url = "https://efts.sec.gov/LATEST"
        self._headers = {
            "User-Agent": settings.sec_edgar_user_agent,
            "Accept": "application/json",
        }

    async def health_check(self) -> bool:
        try:
            async with _SEC_SEMAPHORE:
                await self._get(
                    f"{self._base_url}/search-index",
                    params={"q": "AAPL", "forms": "4", "dateRange": "custom",
                            "startdt": date.today().isoformat(),
                            "enddt": date.today().isoformat()},
                    headers=self._headers,
                )
            return True
        except Exception:
            return False

    async def get_insider_trades(
        self, ticker: str, days_back: int = 90
    ) -> list[InsiderTrade]:
        """Fetch Form 4 insider trades from SEC EDGAR."""
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        async with _SEC_SEMAPHORE:
            data = await self._get(
                f"{self._base_url}/search-index",
                params={
                    "q": f'"{ticker}"',
                    "forms": "4",
                    "dateRange": "custom",
                    "startdt": start_date.isoformat(),
                    "enddt": end_date.isoformat(),
                },
                headers=self._headers,
            )

        trades: list[InsiderTrade] = []
        hits = data.get("hits", {}).get("hits", [])

        for hit in hits:
            source = hit.get("_source", {})
            filing_date_str = source.get("file_date", "")
            try:
                filing_date = (
                    date.fromisoformat(filing_date_str)
                    if filing_date_str
                    else date.today()
                )
            except ValueError:
                filing_date = date.today()

            # Try to get entity info
            names = source.get("display_names", [])
            entity_name = names[0] if names else ""

            # Parse the Form 4 XML if available
            file_url = source.get("file_url", "")
            if file_url:
                xml_trades = await self._parse_form4_xml(
                    ticker, file_url, filing_date
                )
                trades.extend(xml_trades)
            else:
                # Fallback: create a basic record from search metadata
                trades.append(
                    InsiderTrade(
                        ticker=ticker,
                        insider_name=entity_name,
                        filing_date=filing_date,
                        transaction_date=filing_date,
                        source="sec_edgar",
                    )
                )

        return trades

    async def _parse_form4_xml(
        self,
        ticker: str,
        file_url: str,
        filing_date: date,
    ) -> list[InsiderTrade]:
        """Parse a Form 4 XML filing for transaction details."""
        trades: list[InsiderTrade] = []
        try:
            client = await self._get_client()
            async with _SEC_SEMAPHORE:
                resp = await client.get(
                    f"https://www.sec.gov{file_url}",
                    headers=self._headers,
                )
                resp.raise_for_status()

            root = ElementTree.fromstring(resp.text)

            # Get reporting owner info
            owner_elem = root.find(".//reportingOwner")
            owner_name = ""
            owner_title = ""
            if owner_elem is not None:
                name_elem = owner_elem.find(".//rptOwnerName")
                owner_name = (name_elem.text or "") if name_elem is not None else ""
                title_elem = owner_elem.find(
                    ".//officerTitle"
                )
                owner_title = (title_elem.text or "") if title_elem is not None else ""

            # Parse non-derivative transactions
            for txn in root.findall(".//nonDerivativeTransaction"):
                trades.append(
                    self._parse_transaction(
                        txn, ticker, owner_name, owner_title, filing_date
                    )
                )

        except Exception as e:
            logger.debug("form4_parse_error", url=file_url, error=str(e))

        return trades

    @staticmethod
    def _parse_transaction(
        txn: Any,
        ticker: str,
        owner_name: str,
        owner_title: str,
        filing_date: date,
    ) -> InsiderTrade:
        """Parse a single transaction element from Form 4 XML."""

        def _text(parent: Any, tag: str) -> str:
            elem = parent.find(f".//{tag}")
            return elem.text if elem is not None and elem.text else ""

        txn_date_str = _text(txn, "transactionDate/value")
        try:
            txn_date = date.fromisoformat(txn_date_str) if txn_date_str else filing_date
        except ValueError:
            txn_date = filing_date

        shares_str = _text(txn, "transactionAmounts/transactionShares/value")
        price_str = _text(txn, "transactionAmounts/transactionPricePerShare/value")
        code = _text(txn, "transactionCoding/transactionCode")

        shares = int(float(shares_str)) if shares_str else 0
        price = float(price_str) if price_str else 0.0

        return InsiderTrade(
            ticker=ticker,
            insider_name=owner_name,
            title=owner_title,
            transaction_type=code,
            shares=shares,
            price_per_share=price,
            total_value=round(abs(shares * price), 2),
            filing_date=filing_date,
            transaction_date=txn_date,
            source="sec_edgar",
        )
