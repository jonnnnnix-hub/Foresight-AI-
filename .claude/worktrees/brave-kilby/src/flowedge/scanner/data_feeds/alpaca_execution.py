"""Alpaca Markets — paper trading execution layer.

Handles:
- Real-time quote fetching
- Paper order placement (market, limit)
- Position management
- Account status
- P&L tracking

Requires ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.
Uses paper trading endpoint by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


@dataclass
class AlpacaOrder:
    """An order placed on Alpaca."""

    order_id: str
    ticker: str
    side: str  # "buy" or "sell"
    qty: int
    order_type: str  # "market", "limit"
    limit_price: float | None = None
    status: str = "pending"
    filled_price: float = 0.0
    filled_at: str = ""
    model_name: str = ""  # Which model generated this
    signal_conviction: float = 0.0


@dataclass
class AlpacaPosition:
    """A current position on Alpaca."""

    ticker: str
    qty: int
    side: str
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class AlpacaExecutor:
    """Paper trading execution via Alpaca Markets API.

    Usage:
        executor = AlpacaExecutor(api_key, secret_key)
        # Check account
        account = await executor.get_account()
        # Place order
        order = await executor.buy_option("SPY250418C00580000", qty=1)
        # Check positions
        positions = await executor.get_positions()
        # Close position
        await executor.close_position("SPY250418C00580000")
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._base_url = ALPACA_PAPER_URL if paper else "https://api.alpaca.markets"
        self._data_url = ALPACA_DATA_URL
        self._session: Any = None

    async def _ensure_client(self) -> Any:
        if self._session is None:
            import httpx
            self._session = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "APCA-API-KEY-ID": self._api_key,
                    "APCA-API-SECRET-KEY": self._secret_key,
                },
            )
        return self._session

    async def close(self) -> None:
        if self._session:
            await self._session.aclose()
            self._session = None

    async def _get(self, url: str) -> dict[str, Any]:
        client = await self._ensure_client()
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def _post(self, url: str, data: dict[str, Any]) -> dict[str, Any]:
        client = await self._ensure_client()
        resp = await client.post(url, json=data)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    async def _delete(self, url: str) -> dict[str, Any]:
        client = await self._ensure_client()
        resp = await client.delete(url)
        if resp.status_code == 204:
            return {"status": "closed"}
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    # ── Account ────────────────────────────────────────────────────

    async def get_account(self) -> dict[str, Any]:
        """Get account status: equity, buying power, P&L."""
        data = await self._get(f"{self._base_url}/v2/account")
        logger.info(
            "alpaca_account",
            equity=data.get("equity"),
            buying_power=data.get("buying_power"),
            status=data.get("status"),
        )
        return data

    # ── Orders ─────────────────────────────────────────────────────

    async def buy_option(
        self,
        symbol: str,
        qty: int = 1,
        order_type: str = "market",
        limit_price: float | None = None,
        model_name: str = "",
        conviction: float = 0.0,
    ) -> AlpacaOrder:
        """Place a buy order for an option contract.

        Args:
            symbol: OCC option symbol (e.g., "SPY250418C00580000")
            qty: Number of contracts
            order_type: "market" or "limit"
            limit_price: Required for limit orders
            model_name: Which model triggered this
            conviction: Signal conviction score
        """
        order_data: dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy",
            "type": order_type,
            "time_in_force": "day",
        }
        if order_type == "limit" and limit_price is not None:
            order_data["limit_price"] = str(round(limit_price, 2))

        result = await self._post(f"{self._base_url}/v2/orders", order_data)

        order = AlpacaOrder(
            order_id=result.get("id", ""),
            ticker=symbol,
            side="buy",
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            status=result.get("status", "unknown"),
            model_name=model_name,
            signal_conviction=conviction,
        )
        logger.info(
            "alpaca_order_placed",
            symbol=symbol,
            qty=qty,
            order_type=order_type,
            status=order.status,
            model=model_name,
        )
        return order

    async def sell_option(
        self,
        symbol: str,
        qty: int = 1,
        order_type: str = "market",
    ) -> AlpacaOrder:
        """Sell (close) an option position."""
        order_data: dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "sell",
            "type": order_type,
            "time_in_force": "day",
        }
        result = await self._post(f"{self._base_url}/v2/orders", order_data)
        return AlpacaOrder(
            order_id=result.get("id", ""),
            ticker=symbol,
            side="sell",
            qty=qty,
            order_type=order_type,
            status=result.get("status", "unknown"),
        )

    # ── Positions ──────────────────────────────────────────────────

    async def get_positions(self) -> list[AlpacaPosition]:
        """Get all current positions."""
        data = await self._get(f"{self._base_url}/v2/positions")
        positions: list[AlpacaPosition] = []
        if isinstance(data, list):
            for p in data:
                positions.append(AlpacaPosition(
                    ticker=p.get("symbol", ""),
                    qty=int(p.get("qty", 0)),
                    side=p.get("side", "long"),
                    entry_price=float(p.get("avg_entry_price", 0)),
                    current_price=float(p.get("current_price", 0)),
                    market_value=float(p.get("market_value", 0)),
                    unrealized_pnl=float(p.get("unrealized_pl", 0)),
                    unrealized_pnl_pct=float(p.get("unrealized_plpc", 0)) * 100,
                ))
        return positions

    async def close_position(self, symbol: str) -> dict[str, Any]:
        """Close a specific position by symbol."""
        result = await self._delete(
            f"{self._base_url}/v2/positions/{symbol}",
        )
        logger.info("alpaca_position_closed", symbol=symbol)
        return result

    async def close_all_positions(self) -> dict[str, Any]:
        """Close all positions (emergency)."""
        result = await self._delete(f"{self._base_url}/v2/positions")
        logger.warning("alpaca_all_positions_closed")
        return result

    # ── Orders Status ──────────────────────────────────────────────

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Check status of a specific order."""
        return await self._get(f"{self._base_url}/v2/orders/{order_id}")

    async def get_open_orders(self) -> list[dict[str, Any]]:
        """Get all open (unfilled) orders."""
        data = await self._get(f"{self._base_url}/v2/orders?status=open")
        return data if isinstance(data, list) else []

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a pending order."""
        return await self._delete(f"{self._base_url}/v2/orders/{order_id}")
