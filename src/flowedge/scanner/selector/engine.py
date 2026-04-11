"""Smart contract selector — picks optimal contracts for lotto plays.

Given a directional thesis and signal data, selects:
- Optimal expiration (balance theta decay vs. catalyst proximity)
- Optimal strike (OTM for leverage, ATM for probability)
- Structure (single leg, spread, or risk reversal)
- Position sizing guidance
"""

from __future__ import annotations

from datetime import date, timedelta

import structlog

from flowedge.scanner.schemas.catalyst import CatalystSignal
from flowedge.scanner.schemas.flow import FlowSentiment
from flowedge.scanner.schemas.iv import IVSignal
from flowedge.scanner.schemas.options import OptionContract, OptionsChain, OptionType
from flowedge.scanner.schemas.signals import LottoOpportunity

logger = structlog.get_logger()


class ContractRecommendation:
    """A recommended contract with reasoning."""

    def __init__(
        self,
        contract: OptionContract,
        reason: str,
        structure: str = "single_leg",
        risk_reward: float = 0.0,
        max_loss: float = 0.0,
        breakeven_pct: float = 0.0,
    ) -> None:
        self.contract = contract
        self.reason = reason
        self.structure = structure
        self.risk_reward = risk_reward
        self.max_loss = max_loss
        self.breakeven_pct = breakeven_pct


def _pick_expiration(
    chain: OptionsChain,
    catalyst: CatalystSignal | None,
    direction: FlowSentiment,
) -> date | None:
    """Select optimal expiration date.

    Rules:
    - If earnings within 14d, pick expiration AFTER earnings (capture the move)
    - If no catalyst, pick 2-4 weeks out (balance theta vs. time)
    - Never pick expiration < 3 DTE (too much theta)
    """
    today = date.today()
    available_exps = sorted({c.expiration for c in chain.contracts})

    if not available_exps:
        return None

    # If catalyst is near, pick first expiry AFTER it
    if catalyst and catalyst.days_to_nearest_catalyst is not None:
        catalyst_date = today + timedelta(days=catalyst.days_to_nearest_catalyst)
        post_catalyst = [e for e in available_exps if e > catalyst_date]
        if post_catalyst:
            return post_catalyst[0]

    # Default: 2-4 weeks out, minimum 3 DTE
    min_dte = today + timedelta(days=3)
    ideal_min = today + timedelta(days=14)
    ideal_max = today + timedelta(days=30)

    in_range = [e for e in available_exps if ideal_min <= e <= ideal_max]
    if in_range:
        return in_range[0]

    # Fallback: first expiry > 3 DTE
    valid = [e for e in available_exps if e > min_dte]
    return valid[0] if valid else available_exps[-1]


def _pick_strike(
    contracts: list[OptionContract],
    underlying_price: float,
    direction: FlowSentiment,
    iv_signal: IVSignal | None,
) -> list[OptionContract]:
    """Select optimal strikes for a lotto play.

    Lotto profile: slightly OTM (delta 0.15-0.35)
    - Bullish: OTM calls, strike 3-10% above current price
    - Bearish: OTM puts, strike 3-10% below current price
    - Neutral: straddle around ATM

    When IV is cheap, go slightly further OTM (more leverage).
    When IV is expensive, stay closer to ATM (less premium risk).
    """
    if not contracts or underlying_price <= 0:
        return []

    target_type = (
        OptionType.CALL if direction != FlowSentiment.BEARISH else OptionType.PUT
    )
    typed = [c for c in contracts if c.option_type == target_type]

    if not typed:
        return []

    # Determine OTM range based on IV regime
    if iv_signal and iv_signal.is_cheap_premium:
        # Cheap IV → go further OTM for leverage
        otm_min_pct = 0.03
        otm_max_pct = 0.12
    else:
        # Normal/expensive IV → stay closer
        otm_min_pct = 0.02
        otm_max_pct = 0.08

    if target_type == OptionType.CALL:
        min_strike = underlying_price * (1 + otm_min_pct)
        max_strike = underlying_price * (1 + otm_max_pct)
    else:
        min_strike = underlying_price * (1 - otm_max_pct)
        max_strike = underlying_price * (1 - otm_min_pct)

    # Filter to OTM range
    in_range = [
        c for c in typed
        if min_strike <= c.strike <= max_strike
    ]

    if not in_range:
        # Fallback: closest OTM contracts
        if target_type == OptionType.CALL:
            otm = [c for c in typed if c.strike > underlying_price]
        else:
            otm = [c for c in typed if c.strike < underlying_price]
        otm.sort(key=lambda c: abs(c.strike - underlying_price))
        in_range = otm[:3]

    # Sort by volume (prefer liquid contracts)
    in_range.sort(key=lambda c: c.volume, reverse=True)

    # Return top 3 candidates
    return in_range[:3]


def _compute_risk_reward(
    contract: OptionContract,
    underlying_price: float,
    expected_move_pct: float,
) -> tuple[float, float, float]:
    """Compute risk/reward for a contract.

    Returns: (risk_reward_ratio, max_loss_per_contract, breakeven_pct)
    """
    entry_price = contract.mid if contract.mid > 0 else contract.ask
    if entry_price <= 0:
        return 0.0, 0.0, 0.0

    max_loss = entry_price * 100  # Per contract

    # Estimate potential profit if expected move hits
    if contract.option_type == OptionType.CALL:
        breakeven = contract.strike + entry_price
        target_price = underlying_price * (1 + expected_move_pct / 100)
        intrinsic_at_target = max(0, target_price - contract.strike)
    else:
        breakeven = contract.strike - entry_price
        target_price = underlying_price * (1 - expected_move_pct / 100)
        intrinsic_at_target = max(0, contract.strike - target_price)

    potential_profit = (intrinsic_at_target - entry_price) * 100
    risk_reward = potential_profit / max_loss if max_loss > 0 else 0.0

    breakeven_pct = abs(breakeven - underlying_price) / underlying_price * 100

    return round(risk_reward, 2), round(max_loss, 2), round(breakeven_pct, 2)


def select_contracts(
    opportunity: LottoOpportunity,
    chain: OptionsChain,
) -> list[ContractRecommendation]:
    """Select optimal contracts for a lotto opportunity.

    Combines signal data with live chain to pick the best
    strike/expiry/structure for the trade.
    """
    recommendations: list[ContractRecommendation] = []
    direction = opportunity.suggested_direction
    catalyst = opportunity.catalyst_signal
    iv_signal = opportunity.iv_signal

    # Pick optimal expiration
    target_exp = _pick_expiration(chain, catalyst, direction)
    if target_exp is None:
        logger.warning("no_valid_expiration", ticker=opportunity.ticker)
        return []

    # Filter chain to target expiration
    exp_contracts = [c for c in chain.contracts if c.expiration == target_exp]
    if not exp_contracts and target_exp is not None:
        # Try nearest expiration
        all_exps = sorted({c.expiration for c in chain.contracts})
        ref = target_exp
        if all_exps:
            target_exp = min(all_exps, key=lambda e: abs((e - ref).days))
            exp_contracts = [c for c in chain.contracts if c.expiration == target_exp]

    if not exp_contracts:
        return []

    # Pick strikes
    candidates = _pick_strike(
        exp_contracts, chain.underlying_price, direction, iv_signal
    )

    # Expected move for risk/reward calc
    expected_move = 5.0  # Default 5% if no catalyst data
    if catalyst and catalyst.expected_move:
        expected_move = catalyst.expected_move.expected_move_pct
    elif iv_signal and iv_signal.iv_rank.current_iv > 0:
        # Estimate from IV: annualized IV → expected daily move
        daily_iv = iv_signal.iv_rank.current_iv / (252 ** 0.5)
        expected_move = daily_iv * 100 * 5  # ~5 day move

    for contract in candidates:
        rr, max_loss, breakeven_pct = _compute_risk_reward(
            contract, chain.underlying_price, expected_move
        )

        reason_parts = [
            f"{contract.option_type.value.upper()} {contract.strike}",
            f"exp {target_exp}",
            f"delta={contract.delta:.2f}" if contract.delta else "",
            f"breakeven {breakeven_pct:.1f}% OTM",
        ]

        recommendations.append(
            ContractRecommendation(
                contract=contract,
                reason=" | ".join(p for p in reason_parts if p),
                structure="single_leg",
                risk_reward=rr,
                max_loss=max_loss,
                breakeven_pct=breakeven_pct,
            )
        )

    logger.info(
        "contracts_selected",
        ticker=opportunity.ticker,
        expiration=str(target_exp),
        count=len(recommendations),
    )
    return recommendations
