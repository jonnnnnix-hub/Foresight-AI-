"""Paper trading engine — auto-submits orders for top scanner opportunities.

Uses Alpaca paper trading API to:
1. Monitor scanner results
2. Submit buy orders for qualifying opportunities
3. Track positions and P&L
4. Auto-exit based on take-profit / stop-loss / expiry rules
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.paper_trading.schemas import (
    OrderSide,
    OrderStatus,
    PaperOrder,
    PaperPortfolio,
    PaperPosition,
)
from flowedge.scanner.schemas.signals import LottoOpportunity

logger = structlog.get_logger()


class AlpacaPaperTrader:
    """Alpaca paper trading client.

    Submits equity orders (options paper trading requires
    separate Alpaca approval — we use equity as proxy for now).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._base_url = self._settings.alpaca_base_url
        self._headers = {
            "APCA-API-KEY-ID": self._settings.alpaca_api_key_id,
            "APCA-API-SECRET-KEY": self._settings.alpaca_api_secret_key,
            "Content-Type": "application/json",
        }
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get_account(self) -> dict[str, Any]:
        """Fetch paper trading account info."""
        client = await self._get_client()
        resp = await client.get(
            f"{self._base_url}/v2/account",
            headers=self._headers,
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    async def get_positions(self) -> list[PaperPosition]:
        """Fetch current positions."""
        client = await self._get_client()
        resp = await client.get(
            f"{self._base_url}/v2/positions",
            headers=self._headers,
        )
        resp.raise_for_status()
        positions: list[PaperPosition] = []

        for pos in resp.json():
            if not isinstance(pos, dict):
                continue
            positions.append(
                PaperPosition(
                    ticker=str(pos.get("symbol", "")),
                    symbol=str(pos.get("symbol", "")),
                    qty=int(pos.get("qty", 0)),
                    avg_entry=float(pos.get("avg_entry_price", 0)),
                    current_price=float(pos.get("current_price", 0)),
                    market_value=float(pos.get("market_value", 0)),
                    unrealized_pnl=float(pos.get("unrealized_pl", 0)),
                    unrealized_pnl_pct=float(
                        pos.get("unrealized_plpc", 0)
                    )
                    * 100,
                )
            )
        return positions

    async def submit_order(
        self,
        symbol: str,
        qty: int = 1,
        side: str = "buy",
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> PaperOrder:
        """Submit a paper trading order."""
        payload: dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": "day",
        }
        if limit_price and order_type == "limit":
            payload["limit_price"] = str(limit_price)

        client = await self._get_client()
        resp = await client.post(
            f"{self._base_url}/v2/orders",
            headers=self._headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        return PaperOrder(
            order_id=str(data.get("id", "")),
            ticker=symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            qty=qty,
            symbol=symbol,
            order_type=order_type,
            limit_price=limit_price,
            status=OrderStatus.SUBMITTED,
            submitted_at=datetime.now(),
        )

    async def get_portfolio(self) -> PaperPortfolio:
        """Get complete portfolio state."""
        account = await self.get_account()
        positions = await self.get_positions()

        cash = float(account.get("cash", 0))
        portfolio_value = float(account.get("portfolio_value", 0))
        initial = float(account.get("last_equity", portfolio_value))
        total_pnl = portfolio_value - initial

        return PaperPortfolio(
            account_id=str(account.get("id", "")),
            cash=cash,
            portfolio_value=portfolio_value,
            positions=positions,
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=(
                round(total_pnl / initial * 100, 2) if initial > 0 else 0.0
            ),
            updated_at=datetime.now(),
        )


async def auto_trade_opportunities(
    opportunities: list[LottoOpportunity],
    max_positions: int = 5,
    position_size_pct: float = 2.0,
    min_score: float = 6.0,
    settings: Settings | None = None,
) -> list[PaperOrder]:
    """Auto-submit paper orders for top opportunities.

    Args:
        opportunities: Ranked lotto opportunities from scanner.
        max_positions: Maximum concurrent positions.
        position_size_pct: % of portfolio per position.
        min_score: Minimum composite score to trade.
    """
    settings = settings or get_settings()
    trader = AlpacaPaperTrader(settings)
    orders: list[PaperOrder] = []

    try:
        portfolio = await trader.get_portfolio()
        current_positions = len(portfolio.positions)
        available_slots = max_positions - current_positions

        if available_slots <= 0:
            logger.info(
                "max_positions_reached",
                current=current_positions,
                max=max_positions,
            )
            return orders

        # Filter and rank
        qualified = [
            o for o in opportunities if o.composite_score >= min_score
        ]
        qualified.sort(key=lambda o: o.composite_score, reverse=True)

        position_size = portfolio.portfolio_value * (position_size_pct / 100)

        for opp in qualified[:available_slots]:
            # Use equity as proxy (options paper trading requires separate setup)
            # Buy the underlying ticker with position_size dollars
            try:
                # Calculate share qty from position size
                # We'd need current price — approximate from UOA signal
                approx_price = 100.0  # Default
                if opp.uoa_signal and opp.uoa_signal.alerts:
                    approx_price = opp.uoa_signal.alerts[0].underlying_price

                if approx_price <= 0:
                    continue

                qty = max(1, int(position_size / approx_price))

                order = await trader.submit_order(
                    symbol=opp.ticker,
                    qty=qty,
                    side="buy",
                    order_type="market",
                )
                order.signal_score = opp.composite_score
                order.notes = (
                    f"Scanner score {opp.composite_score:.1f} | "
                    f"{opp.suggested_direction.value}"
                )
                orders.append(order)

                logger.info(
                    "paper_order_submitted",
                    ticker=opp.ticker,
                    qty=qty,
                    score=opp.composite_score,
                )

            except Exception as e:
                logger.warning(
                    "paper_order_failed",
                    ticker=opp.ticker,
                    error=str(e),
                )

    finally:
        await trader.close()

    return orders
