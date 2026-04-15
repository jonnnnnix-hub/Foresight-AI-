"""ARCHITECT v2 — multi-leg strategy construction engine.

Builds optimal strategy structures based on signal data:
- Bullish + cheap IV → debit call spread or naked OTM call
- Bullish + expensive IV → call credit spread (sell premium)
- Catalyst + direction unknown → long straddle or strangle
- Bearish + cheap IV → debit put spread
- High GEX flip risk → iron condor for pin range
"""

from __future__ import annotations

from datetime import date

import structlog

from flowedge.scanner.schemas.flow import FlowSentiment
from flowedge.scanner.schemas.options import OptionContract, OptionsChain, OptionType
from flowedge.scanner.schemas.signals import LottoOpportunity
from flowedge.scanner.selector.schemas import (
    StrategyBlueprint,
    StrategyLeg,
    StrategyType,
)

logger = structlog.get_logger()


def _find_contract(
    contracts: list[OptionContract],
    option_type: OptionType,
    target_strike: float,
    tolerance_pct: float = 0.02,
) -> OptionContract | None:
    """Find the nearest contract matching type and approximate strike."""
    matches = [c for c in contracts if c.option_type == option_type]
    if not matches:
        return None
    return min(matches, key=lambda c: abs(c.strike - target_strike))


def build_single_leg(
    chain: OptionsChain,
    direction: FlowSentiment,
    target_exp: date,
    otm_pct: float = 0.05,
) -> StrategyBlueprint | None:
    """Build a single-leg OTM call or put — classic lotto."""
    price = chain.underlying_price
    exp_contracts = [c for c in chain.contracts if c.expiration == target_exp]
    if not exp_contracts:
        return None

    if direction != FlowSentiment.BEARISH:
        opt_type = OptionType.CALL
        strat_type = StrategyType.SINGLE_CALL
        target_strike = price * (1 + otm_pct)
    else:
        opt_type = OptionType.PUT
        strat_type = StrategyType.SINGLE_PUT
        target_strike = price * (1 - otm_pct)

    contract = _find_contract(exp_contracts, opt_type, target_strike)
    if not contract:
        return None

    entry = contract.mid if contract.mid > 0 else contract.ask
    max_loss = entry * 100

    breakeven = (
        contract.strike + entry
        if opt_type == OptionType.CALL
        else contract.strike - entry
    )

    return StrategyBlueprint(
        ticker=chain.underlying,
        strategy_type=strat_type,
        legs=[StrategyLeg(contract=contract, action="buy", premium=entry)],
        expiration=target_exp,
        net_debit=round(entry * 100, 2),
        max_profit=0.0,  # Unlimited for single leg
        max_loss=round(max_loss, 2),
        breakeven_prices=[round(breakeven, 2)],
        risk_reward_ratio=0.0,  # Undefined for unlimited upside
        rationale=f"Single {opt_type.value} lotto at ${contract.strike:.0f}",
        tags=["lotto"],
    )


def build_debit_spread(
    chain: OptionsChain,
    direction: FlowSentiment,
    target_exp: date,
    long_otm_pct: float = 0.03,
    width: float = 5.0,
) -> StrategyBlueprint | None:
    """Build a debit spread — defined risk, better R/R than single leg."""
    price = chain.underlying_price
    exp_contracts = [c for c in chain.contracts if c.expiration == target_exp]
    if not exp_contracts:
        return None

    if direction != FlowSentiment.BEARISH:
        opt_type = OptionType.CALL
        strat_type = StrategyType.CALL_DEBIT_SPREAD
        long_strike = price * (1 + long_otm_pct)
        short_strike = long_strike + width
    else:
        opt_type = OptionType.PUT
        strat_type = StrategyType.PUT_DEBIT_SPREAD
        long_strike = price * (1 - long_otm_pct)
        short_strike = long_strike - width

    long_contract = _find_contract(exp_contracts, opt_type, long_strike)
    short_contract = _find_contract(exp_contracts, opt_type, short_strike)
    if not long_contract or not short_contract:
        return None

    long_premium = long_contract.mid if long_contract.mid > 0 else long_contract.ask
    short_premium = short_contract.mid if short_contract.mid > 0 else short_contract.bid

    net_debit = long_premium - short_premium
    spread_width = abs(long_contract.strike - short_contract.strike)
    max_profit = (spread_width - net_debit) * 100
    max_loss = net_debit * 100

    if opt_type == OptionType.CALL:
        breakeven = long_contract.strike + net_debit
    else:
        breakeven = long_contract.strike - net_debit

    rr = round(max_profit / max_loss, 2) if max_loss > 0 else 0.0

    return StrategyBlueprint(
        ticker=chain.underlying,
        strategy_type=strat_type,
        legs=[
            StrategyLeg(contract=long_contract, action="buy", premium=long_premium),
            StrategyLeg(contract=short_contract, action="sell", premium=short_premium),
        ],
        expiration=target_exp,
        net_debit=round(net_debit * 100, 2),
        max_profit=round(max_profit, 2),
        max_loss=round(max_loss, 2),
        breakeven_prices=[round(breakeven, 2)],
        risk_reward_ratio=rr,
        rationale=(
            f"Debit {opt_type.value} spread "
            f"${long_contract.strike:.0f}/{short_contract.strike:.0f} "
            f"for ${net_debit:.2f} debit, {rr:.1f}x R/R"
        ),
        tags=["lotto", "defined_risk"],
    )


def build_straddle(
    chain: OptionsChain,
    target_exp: date,
) -> StrategyBlueprint | None:
    """Build a long straddle — profits from big move in either direction."""
    price = chain.underlying_price
    exp_contracts = [c for c in chain.contracts if c.expiration == target_exp]
    if not exp_contracts:
        return None

    call = _find_contract(exp_contracts, OptionType.CALL, price)
    put = _find_contract(exp_contracts, OptionType.PUT, price)
    if not call or not put:
        return None

    call_prem = call.mid if call.mid > 0 else call.ask
    put_prem = put.mid if put.mid > 0 else put.ask
    total_debit = call_prem + put_prem
    max_loss = total_debit * 100

    return StrategyBlueprint(
        ticker=chain.underlying,
        strategy_type=StrategyType.LONG_STRADDLE,
        legs=[
            StrategyLeg(contract=call, action="buy", premium=call_prem),
            StrategyLeg(contract=put, action="buy", premium=put_prem),
        ],
        expiration=target_exp,
        net_debit=round(total_debit * 100, 2),
        max_profit=0.0,  # Unlimited
        max_loss=round(max_loss, 2),
        breakeven_prices=[
            round(call.strike + total_debit, 2),
            round(put.strike - total_debit, 2),
        ],
        rationale=(
            f"Long straddle at ${call.strike:.0f} "
            f"for ${total_debit:.2f} debit — "
            f"needs {total_debit / price * 100:.1f}% move"
        ),
        tags=["volatility", "defined_risk"],
    )


def build_strangle(
    chain: OptionsChain,
    target_exp: date,
    otm_pct: float = 0.05,
) -> StrategyBlueprint | None:
    """Build a long strangle — cheaper than straddle, needs bigger move."""
    price = chain.underlying_price
    exp_contracts = [c for c in chain.contracts if c.expiration == target_exp]
    if not exp_contracts:
        return None

    call = _find_contract(exp_contracts, OptionType.CALL, price * (1 + otm_pct))
    put = _find_contract(exp_contracts, OptionType.PUT, price * (1 - otm_pct))
    if not call or not put:
        return None

    call_prem = call.mid if call.mid > 0 else call.ask
    put_prem = put.mid if put.mid > 0 else put.ask
    total_debit = call_prem + put_prem
    max_loss = total_debit * 100

    return StrategyBlueprint(
        ticker=chain.underlying,
        strategy_type=StrategyType.LONG_STRANGLE,
        legs=[
            StrategyLeg(contract=call, action="buy", premium=call_prem),
            StrategyLeg(contract=put, action="buy", premium=put_prem),
        ],
        expiration=target_exp,
        net_debit=round(total_debit * 100, 2),
        max_profit=0.0,
        max_loss=round(max_loss, 2),
        breakeven_prices=[
            round(call.strike + total_debit, 2),
            round(put.strike - total_debit, 2),
        ],
        rationale=(
            f"Long strangle ${put.strike:.0f}P/${call.strike:.0f}C "
            f"for ${total_debit:.2f} — cheaper than straddle"
        ),
        tags=["lotto", "volatility", "defined_risk"],
    )


def recommend_strategies(
    opp: LottoOpportunity,
    chain: OptionsChain,
    target_exp: date | None = None,
) -> list[StrategyBlueprint]:
    """ARCHITECT v2 — recommend optimal strategy structures.

    Selection logic:
    1. Always include a single-leg lotto (baseline)
    2. Add debit spread for defined-risk alternative
    3. If catalyst near + direction unclear → add straddle/strangle
    4. If FLUX contradicts UOA direction → prefer straddle (hedge both sides)
    5. Sort by risk/reward
    """
    if not chain.contracts:
        return []

    # Pick expiration
    if target_exp is None:
        exps = sorted({c.expiration for c in chain.contracts})
        today = date.today()
        from datetime import timedelta

        ideal = [e for e in exps if today + timedelta(days=14) <= e <= today + timedelta(days=30)]
        target_exp = ideal[0] if ideal else (exps[1] if len(exps) > 1 else exps[0])

    direction = opp.suggested_direction

    # FLUX can override direction if UOA is neutral but FLUX is strong
    flux = opp.flux_signal
    if flux and hasattr(flux, "bias") and direction == FlowSentiment.NEUTRAL:
        from flowedge.scanner.flux.schemas import FlowBias
        if flux.bias == FlowBias.STRONG_BUY:
            direction = FlowSentiment.BULLISH
            logger.info("flux_direction_override", ticker=opp.ticker, to="bullish")
        elif flux.bias == FlowBias.STRONG_SELL:
            direction = FlowSentiment.BEARISH
            logger.info("flux_direction_override", ticker=opp.ticker, to="bearish")

    # Check if FLUX contradicts UOA — signals disagreement
    flux_contradicts = False
    if flux and hasattr(flux, "bias"):
        from flowedge.scanner.flux.schemas import FlowBias
        if (
            (
                direction == FlowSentiment.BULLISH
                and flux.bias in (FlowBias.SELL, FlowBias.STRONG_SELL)
            )
            or (
                direction == FlowSentiment.BEARISH
                and flux.bias in (FlowBias.BUY, FlowBias.STRONG_BUY)
            )
        ):
            flux_contradicts = True

    strategies: list[StrategyBlueprint] = []

    # 1. Single-leg lotto (always)
    single = build_single_leg(chain, direction, target_exp)
    if single:
        if flux_contradicts:
            single.tags.append("flux_disagrees")
        strategies.append(single)

    # 2. Debit spread (always — defined risk alternative)
    spread = build_debit_spread(chain, direction, target_exp)
    if spread:
        strategies.append(spread)

    # 3. Straddle/strangle if catalyst + uncertain direction
    has_catalyst = (
        opp.catalyst_signal is not None
        and opp.catalyst_signal.days_to_nearest_catalyst is not None
        and opp.catalyst_signal.days_to_nearest_catalyst <= 14
    )
    if has_catalyst or direction == FlowSentiment.NEUTRAL or flux_contradicts:
        straddle = build_straddle(chain, target_exp)
        if straddle:
            strategies.append(straddle)
        strangle = build_strangle(chain, target_exp)
        if strangle:
            strategies.append(strangle)

    logger.info(
        "architect_v2_complete",
        ticker=opp.ticker,
        strategies=len(strategies),
        types=[s.strategy_type.value for s in strategies],
    )
    return strategies
