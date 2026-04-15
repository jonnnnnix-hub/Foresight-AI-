"""Real options execution layer — shared by ALL models.

NO Black-Scholes. NO slippage model. NO mock data.
Uses ACTUAL option contract prices from Massive S3 flat files.

Every model calls this to:
1. Find nearest ATM option from real cached data
2. Enter at real option price
3. Track real option prices through hold period
4. Exit at real option price
5. Apply $0.18/contract rebate (Public.com)

Data: data/flat_files_s3/{ticker}/options_1min/*.json
Format: Pre-parsed with contract, strike, expiration, dte, OHLCV
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import structlog

from flowedge.scanner.backtest.schemas import BacktestTrade, TradeOutcome

logger = structlog.get_logger()

CACHE_DIR = Path("data/flat_files_s3")
REBATE_PER_CONTRACT = 0.18  # $0.18 rebate per contract (Public.com)


def load_stock_daily(
    ticker: str,
) -> list[dict[str, Any]]:
    """Load daily OHLCV from cached minute bars."""
    min_dir = CACHE_DIR / ticker / "1min"
    if not min_dir.exists():
        return []

    raw_bars: list[dict[str, Any]] = []
    for f in sorted(min_dir.glob("*.json")):
        raw_bars.extend(json.loads(f.read_text()))

    daily: dict[str, dict[str, Any]] = {}
    for b in raw_bars:
        d = str(b.get("date", b.get("d", "")))
        if not d:
            continue
        o = float(b.get("open", b.get("o", 0)))
        h = float(b.get("high", b.get("h", 0)))
        lo = float(b.get("low", b.get("l", 0)))
        c = float(b.get("close", b.get("c", 0)))
        v = float(b.get("volume", b.get("v", 0)))

        if d not in daily:
            daily[d] = {
                "date": d, "open": o, "high": h, "low": lo,
                "close": c, "volume": v,
            }
        else:
            daily[d]["high"] = max(daily[d]["high"], h)
            if lo > 0:
                daily[d]["low"] = min(daily[d]["low"], lo)
            daily[d]["close"] = c
            daily[d]["volume"] += v

    return [daily[d] for d in sorted(daily)]


def load_options_for_ticker(
    ticker: str,
) -> dict[str, list[dict[str, Any]]]:
    """Load all cached options data for a ticker, keyed by date."""
    opt_dir = CACHE_DIR / ticker / "options_1min"
    if not opt_dir.exists():
        return {}

    result: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(opt_dir.glob("*.json")):
        parts = f.stem.split("_")
        d_str = parts[-1] if len(parts) >= 4 else ""
        if not d_str or len(d_str) != 10:
            continue
        data = json.loads(f.read_text())
        if isinstance(data, list):
            result[d_str] = data
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    result[k] = v

    return result


def find_best_option(
    option_bars: list[dict[str, Any]],
    current_price: float,
    direction: str = "bullish",
    min_dte: int = 1,
    max_dte: int = 21,
    max_strike_dist_pct: float = 0.02,
    min_volume: int = 5,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Find the best option contract and return all its minute bars.

    Returns (option_info, sorted_contract_bars).
    """
    opt_type = "C" if direction == "bullish" else "P"

    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    contract_meta: dict[str, dict[str, Any]] = {}

    for ob in option_bars:
        contract = ob.get("contract", "")
        if not contract:
            continue
        if ob.get("option_type", "") != opt_type:
            continue

        dte = int(ob.get("dte", 99))
        if dte < min_dte or dte > max_dte:
            continue

        strike = float(ob.get("strike", 0))
        dist = abs(strike - current_price) / current_price
        if dist > max_strike_dist_pct:
            continue

        vol = int(ob.get("v", ob.get("volume", 0)))
        by_contract[contract].append(ob)

        if contract not in contract_meta or vol > contract_meta[contract].get("vol", 0):
            contract_meta[contract] = {
                "contract": contract,
                "strike": strike,
                "dte": dte,
                "expiration": ob.get("expiration", ""),
                "dist": dist,
                "vol": vol,
            }

    # Filter for minimum volume across the day
    viable = {
        c: m for c, m in contract_meta.items()
        if sum(int(b.get("v", b.get("volume", 0))) for b in by_contract[c]) >= min_volume
    }

    if not viable:
        return None, []

    # Pick closest to ATM with best volume
    best_contract = min(
        viable.values(),
        key=lambda x: (x["dist"], -x["vol"]),
    )

    bars = sorted(
        by_contract[best_contract["contract"]],
        key=lambda x: str(x.get("timestamp", x.get("ts", ""))),
    )

    # Get entry price from first bar with volume
    price = 0.0
    for cb in bars:
        p = float(cb.get("c", cb.get("close", 0)))
        v = int(cb.get("v", cb.get("volume", 0)))
        if p > 0 and v >= 1:
            price = p
            break

    if price <= 0:
        return None, []

    info = {
        "contract": best_contract["contract"],
        "strike": best_contract["strike"],
        "expiration": best_contract["expiration"],
        "dte": best_contract["dte"],
        "entry_price": price,
        "daily_volume": sum(
            int(b.get("v", b.get("volume", 0))) for b in bars
        ),
    }
    return info, bars


def execute_trade(
    option_info: dict[str, Any],
    contract_bars: list[dict[str, Any]],
    entry_date: str,
    ticker: str,
    direction: str,
    strategy: str,
    conviction: float,
    underlying_entry: float,
    cash: float,
    risk_pct: float = 0.08,
    tp_pct: float = 0.50,
    max_hold_days: int = 12,
    min_hold_days: int = 3,
    trail_activation: float = 0.30,
    trail_pct: float = 0.25,
    all_options: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[BacktestTrade | None, float]:
    """Execute a trade using real option prices.

    For multi-day holds, pass all_options (date→bars) so the function
    can look up the same contract across subsequent days.

    Returns (trade_result, net_cash_change).
    Cash change includes $0.18/contract rebate.
    """
    entry_premium = option_info["entry_price"]
    contract_id = option_info["contract"]
    if entry_premium <= 0:
        return None, 0.0

    # Size position
    budget = cash * risk_pct
    contracts = max(1, int(budget / (entry_premium * 100)))
    cost = contracts * entry_premium * 100

    if cost > cash * 0.9 or cost < 10:
        return None, 0.0

    # Build multi-day contract bars from all_options
    bars_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Add entry day bars
    for cb in contract_bars:
        cb_date = str(cb.get("date", ""))
        if cb_date:
            bars_by_date[cb_date].append(cb)

    # Add subsequent days from all_options (for multi-day holds)
    if all_options:
        from datetime import date as dt, timedelta
        entry_d = dt.fromisoformat(entry_date)
        for day_offset in range(1, max_hold_days + 2):
            next_d = (entry_d + timedelta(days=day_offset)).isoformat()
            if next_d in all_options:
                for ob in all_options[next_d]:
                    if ob.get("contract", "") == contract_id:
                        bars_by_date[next_d].append(ob)

    # Walk forward through dates
    dates_after_entry = sorted(
        d for d in bars_by_date if d >= entry_date
    )

    max_premium = entry_premium
    exit_premium = entry_premium
    exit_reason = "time_exit"
    hold_bars = 0
    current_date = entry_date
    exit_underlying = underlying_entry
    days_held = 0
    exited = False

    for d in dates_after_entry:
        if exited:
            break
        if d > entry_date:
            days_held += 1

        day_bars = bars_by_date[d]
        for cb in day_bars:
            cb_price = float(cb.get("c", cb.get("close", 0)))
            if cb_price <= 0:
                continue

            hold_bars += 1
            exit_premium = cb_price
            current_date = d

            if cb_price > max_premium:
                max_premium = cb_price

            pnl_frac = (cb_price - entry_premium) / entry_premium

            # 1. Take profit
            if pnl_frac >= tp_pct:
                exit_reason = "take_profit"
                exited = True
                break

            # Min hold guard
            if days_held < min_hold_days:
                continue

            # 2. Trailing stop
            if max_premium > entry_premium * (1 + trail_activation):
                trail_level = max_premium * (1 - trail_pct)
                if cb_price <= trail_level:
                    exit_reason = "trailing_stop"
                    exited = True
                    break

        # 3. Time exit
        if days_held >= max_hold_days and not exited:
            exit_reason = "time_exit"
            exited = True

    # Compute P&L
    exit_value = exit_premium * contracts * 100
    pnl_raw = exit_value - cost

    # $0.18 rebate per contract (entry + exit = 2 transactions)
    rebate = REBATE_PER_CONTRACT * contracts * 2
    pnl_net = pnl_raw + rebate

    pnl_pct = (pnl_net / cost * 100) if cost > 0 else 0.0

    if pnl_pct >= 5:
        outcome = TradeOutcome.WIN
    elif pnl_pct < -5:
        outcome = TradeOutcome.LOSS
    else:
        outcome = TradeOutcome.BREAKEVEN

    trade = BacktestTrade(
        ticker=ticker,
        entry_date=date.fromisoformat(entry_date),
        exit_date=date.fromisoformat(current_date),
        option_type="call" if direction == "bullish" else "put",
        strike=option_info["strike"],
        entry_price=round(entry_premium, 4),
        exit_price=round(exit_premium, 4),
        underlying_entry=round(underlying_entry, 2),
        underlying_exit=round(exit_underlying, 2),
        underlying_move_pct=0,
        pnl_per_contract=round(pnl_net / max(contracts, 1), 2),
        pnl_pct=round(pnl_pct, 2),
        outcome=outcome,
        signal_score=round(conviction, 1),
        signal_type=f"{strategy}|{option_info['contract']}",
        hold_days=days_held,
        strategy=strategy,
        regime="",
        conviction=round(conviction, 2),
        exit_reason=exit_reason,
        contracts=contracts,
        cost_basis=round(cost, 2),
        exit_value=round(exit_value + rebate, 2),
    )

    return trade, pnl_net
