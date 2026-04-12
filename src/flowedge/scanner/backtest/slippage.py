"""Options slippage and bid-ask spread model.

Current backtest assumes perfect fills at theoretical BS price — unrealistic.
Real options have:
- Bid-ask spreads (wider for OTM, low volume, low OI)
- Slippage on entry (buy at ask) and exit (sell at bid)
- Impact on small accounts (market orders move price)

This module models realistic fill prices based on:
1. Option moneyness (OTM options have wider spreads)
2. Underlying liquidity (SPY=tight, IWM=wider)
3. Premium level (cheap options have proportionally wider spreads)
4. Time of day effects (wider at open/close)

Typical spreads:
- SPY ATM: $0.01-0.03 (very tight)
- SPY 2% OTM: $0.03-0.08
- Single stock ATM: $0.05-0.15
- Single stock 3% OTM: $0.10-0.50
- Low liquidity stock 5% OTM: $0.30-1.00+

Impact on a $2.00 option:
- SPY: ~1.5% round-trip cost
- Liquid single stock: ~5% round-trip cost
- Illiquid: ~15%+ round-trip cost
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Spread multipliers by underlying liquidity tier
LIQUIDITY_TIERS: dict[str, float] = {
    # Tier 1: Ultra-liquid ETFs (tightest spreads)
    "SPY": 0.5,
    "QQQ": 0.6,
    "IWM": 0.7,
    # Tier 2: Mega-cap liquid names
    "AAPL": 0.8,
    "MSFT": 0.8,
    "AMZN": 0.85,
    "GOOGL": 0.85,
    "META": 0.85,
    "NVDA": 0.9,
    "TSLA": 0.9,
    # Tier 3: Large-cap (default)
    "AMD": 1.0,
}
DEFAULT_LIQUIDITY = 1.0


@dataclass
class SlippageModel:
    """Configurable slippage model for option backtesting.

    Attributes:
        base_spread_pct: Base half-spread as % of premium (one side).
        otm_spread_multiplier: How much wider spreads get per 1% OTM.
        cheap_option_floor: Minimum absolute half-spread in dollars.
        market_impact_pct: Additional slippage from market impact.
        enabled: If False, returns zero slippage (for comparison).
    """

    base_spread_pct: float = 0.015  # 1.5% of premium (one side, calibrated)
    otm_spread_multiplier: float = 0.8  # 1% OTM → 0.8x wider (not 1.5x)
    cheap_option_floor: float = 0.02  # $0.02 minimum half-spread
    market_impact_pct: float = 0.003  # 0.3% market impact (small accounts)
    enabled: bool = True


def estimate_half_spread(
    premium: float,
    otm_pct: float,
    ticker: str,
    model: SlippageModel | None = None,
) -> float:
    """Estimate the half bid-ask spread for an option.

    The half-spread is the cost of crossing from mid to ask (entry)
    or mid to bid (exit). Total round-trip cost = 2 * half_spread.

    Args:
        premium: Theoretical (mid) option premium.
        otm_pct: How far out-of-the-money (0.02 = 2% OTM).
        ticker: Underlying ticker for liquidity lookup.
        model: Slippage model parameters.

    Returns:
        Half-spread in dollars per contract.
    """
    if model is None:
        model = SlippageModel()
    if not model.enabled:
        return 0.0

    # Base spread from premium
    base = premium * model.base_spread_pct

    # OTM multiplier: deeper OTM → wider spreads
    # 2% OTM → 3x multiplier, 5% OTM → 7.5x
    otm_mult = 1.0 + (otm_pct * 100) * model.otm_spread_multiplier
    spread = base * otm_mult

    # Liquidity tier adjustment
    liq = LIQUIDITY_TIERS.get(ticker, DEFAULT_LIQUIDITY)
    spread *= liq

    # Cheap option floor — can't have sub-penny spreads
    spread = max(spread, model.cheap_option_floor)

    # Market impact
    impact = premium * model.market_impact_pct
    spread += impact

    return round(spread, 4)


def apply_entry_slippage(
    theoretical_premium: float,
    otm_pct: float,
    ticker: str,
    model: SlippageModel | None = None,
) -> float:
    """Get the fill price for an option entry (buy at ask).

    Returns:
        Fill price (theoretical + half-spread).
    """
    half = estimate_half_spread(theoretical_premium, otm_pct, ticker, model)
    return round(theoretical_premium + half, 4)


def apply_exit_slippage(
    theoretical_premium: float,
    otm_pct: float,
    ticker: str,
    model: SlippageModel | None = None,
) -> float:
    """Get the fill price for an option exit (sell at bid).

    Returns:
        Fill price (theoretical - half-spread), floored at 0.01.
    """
    half = estimate_half_spread(theoretical_premium, otm_pct, ticker, model)
    return round(max(0.01, theoretical_premium - half), 4)


def compute_roundtrip_cost(
    entry_premium: float,
    exit_premium: float,
    otm_pct: float,
    ticker: str,
    contracts: int = 1,
    model: SlippageModel | None = None,
) -> dict[str, float]:
    """Compute full round-trip slippage cost for a trade.

    Returns:
        Dict with entry_fill, exit_fill, entry_cost, exit_cost,
        total_slippage_dollars, slippage_pct_of_cost.
    """
    entry_fill = apply_entry_slippage(entry_premium, otm_pct, ticker, model)
    exit_fill = apply_exit_slippage(exit_premium, otm_pct, ticker, model)

    entry_cost = entry_fill * contracts * 100
    exit_proceeds = exit_fill * contracts * 100

    # Compare to theoretical
    theo_entry = entry_premium * contracts * 100
    theo_exit = exit_premium * contracts * 100

    slippage_entry = entry_cost - theo_entry
    slippage_exit = theo_exit - exit_proceeds
    total_slippage = slippage_entry + slippage_exit

    slippage_pct = (total_slippage / theo_entry * 100) if theo_entry > 0 else 0.0

    return {
        "entry_fill": entry_fill,
        "exit_fill": exit_fill,
        "entry_cost": round(entry_cost, 2),
        "exit_proceeds": round(exit_proceeds, 2),
        "slippage_entry": round(slippage_entry, 2),
        "slippage_exit": round(slippage_exit, 2),
        "total_slippage": round(total_slippage, 2),
        "slippage_pct": round(slippage_pct, 2),
    }


def estimate_portfolio_slippage(
    trades: list[dict[str, Any]],
    model: SlippageModel | None = None,
) -> dict[str, float]:
    """Estimate total slippage impact on a portfolio of trades.

    Args:
        trades: List of trade dicts with entry_price, exit_price,
                ticker, contracts, and optionally otm_pct.

    Returns:
        Summary with total_slippage, avg_per_trade, pnl_impact_pct.
    """
    if not trades:
        return {"total_slippage": 0.0, "avg_per_trade": 0.0, "pnl_impact_pct": 0.0}

    total_slip = 0.0
    total_cost = 0.0

    for t in trades:
        entry_p = float(t.get("entry_price", 0))
        exit_p = float(t.get("exit_price", 0))
        ticker = str(t.get("ticker", ""))
        contracts = int(t.get("contracts", 1))
        otm_pct = float(t.get("otm_pct", 0.025))

        if entry_p <= 0:
            continue

        rt = compute_roundtrip_cost(
            entry_p, exit_p, otm_pct, ticker, contracts, model,
        )
        total_slip += rt["total_slippage"]
        total_cost += rt["entry_cost"]

    avg_slip = total_slip / len(trades) if trades else 0.0
    pnl_impact = (total_slip / total_cost * 100) if total_cost > 0 else 0.0

    return {
        "total_slippage": round(total_slip, 2),
        "avg_per_trade": round(avg_slip, 2),
        "pnl_impact_pct": round(pnl_impact, 2),
        "trade_count": float(len(trades)),
    }
