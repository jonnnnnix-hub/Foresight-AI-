"""Composite scalp signal scorer — Polygon flow + ORATS conviction.

Combines real-time flow metrics (speed) with ORATS IV analytics (edge)
into a single 0-10 composite score for options scalp entries.

Weights (user-specified):
  flow:       30%  — volume spike, aggression, sweeps, blocks
  momentum:   15%  — IBS, RSI3, price vs VWAP, bar direction
  liquidity:  10%  — option spread, volume, open interest
  edge:       20%  — IV rank, IV-HV spread, term structure
  regime:     15%  — VIX level, earnings proximity, market breadth
  greeks:     10%  — delta, gamma, theta profile

Distinct from NEXUS scorer (which targets lotto opportunities with
UOA/IV/Catalyst/FLUX weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from flowedge.scanner.flux.flow_state import TickerFlowState, TriggerConditions
from flowedge.scanner.flux.schemas import (
    DeltaDivergence,
    FlowBias,
    TradeDirection,
)
from flowedge.scanner.providers.orats_cache import ORATSContext

logger = structlog.get_logger()


# ── Score output ──────────────────────────────────────────────


@dataclass
class ScalpSignalScore:
    """Composite scalp signal score with per-component breakdown."""

    ticker: str
    total: float = 0.0            # 0-10 composite

    # Components (each 0-10 before weighting)
    flow_score: float = 0.0       # 30%
    momentum_score: float = 0.0   # 15%
    liquidity_score: float = 0.0  # 10%
    edge_score: float = 0.0       # 20%
    regime_score: float = 0.0     # 15%
    greeks_score: float = 0.0     # 10%

    # Direction
    direction: str = "long"       # "long" or "short"

    # Confidence tier
    confidence: str = "LOW"       # LOW / MEDIUM / HIGH / ULTRA

    # Reasoning
    reasons: list[str] = field(default_factory=list)

    # Hard rejections
    rejected: bool = False
    reject_reason: str = ""


# ── Weights ───────────────────────────────────────────────────

_W_FLOW = 0.30
_W_MOMENTUM = 0.15
_W_LIQUIDITY = 0.10
_W_EDGE = 0.20
_W_REGIME = 0.15
_W_GREEKS = 0.10

# Hard rejection thresholds
_MAX_SPREAD_PCT = 10.0       # Reject if option spread > 10%
_MIN_VOLUME_1M = 10          # Reject if < 10 trades in 1 min
_MAX_DTE = 7                 # Reject DTE > 7 for scalps
_MIN_OPEN_INTEREST = 100     # Reject illiquid contracts
_EARNINGS_BLACKOUT_DAYS = 2  # Reject if earnings within 2 days


# ── Scoring functions ─────────────────────────────────────────


def _score_flow(state: TickerFlowState) -> tuple[float, list[str]]:
    """Score real-time flow quality (0-10)."""
    score = 0.0
    reasons: list[str] = []

    # Aggression ratio
    ar = state.aggression_ratio
    if ar > 0.70:
        score += 3.0
        reasons.append(f"strong buy aggression ({ar:.0%})")
    elif ar > 0.58:
        score += 2.0
        reasons.append(f"net buy pressure ({ar:.0%})")
    elif ar < 0.30:
        score += 3.0
        reasons.append(f"strong sell aggression ({ar:.0%})")
    elif ar < 0.42:
        score += 2.0
        reasons.append(f"net sell pressure ({ar:.0%})")
    else:
        score += 0.5

    # Sweep detection
    if state.sweep_count_5m > 0:
        score += 2.0
        reasons.append(f"{state.sweep_count_5m} sweep(s) detected")

    # Block prints
    if state.block_prints:
        score += 1.5
        reasons.append(f"{len(state.block_prints)} block print(s)")

    # Persistent bias (3+ consecutive same-direction readings)
    if state.is_bias_persistent:
        score += 1.5
        reasons.append("persistent directional bias")

    # Divergence (hidden accumulation/distribution)
    if state.divergence in (DeltaDivergence.BULLISH, DeltaDivergence.BEARISH):
        score += 2.0
        reasons.append(f"delta divergence: {state.divergence.value}")

    return min(score, 10.0), reasons


def _score_momentum(
    state: TickerFlowState,
    ibs: float | None = None,
    rsi3: float | None = None,
    price_vs_vwap: float | None = None,
) -> tuple[float, list[str]]:
    """Score momentum alignment (0-10)."""
    score = 0.0
    reasons: list[str] = []

    # Volume ratio
    if state.volume_ratio_1m > 3.0:
        score += 3.0
        reasons.append(f"volume spike {state.volume_ratio_1m:.1f}x")
    elif state.volume_ratio_1m > 2.0:
        score += 2.0
    elif state.volume_ratio_1m > 1.5:
        score += 1.0

    # IBS (if provided by scanner)
    if ibs is not None:
        if ibs < 0.15:
            score += 2.0
            reasons.append(f"IBS oversold ({ibs:.2f})")
        elif ibs > 0.85:
            score += 2.0
            reasons.append(f"IBS overbought ({ibs:.2f})")

    # RSI3 (if provided)
    if rsi3 is not None:
        if rsi3 < 15:
            score += 2.0
            reasons.append(f"RSI3 extreme ({rsi3:.1f})")
        elif rsi3 < 25:
            score += 1.0

    # Rapid-fire trades
    if state.rapid_fire_count > 0:
        score += 1.5
        reasons.append("rapid-fire trades detected")

    # Price vs VWAP alignment with flow direction
    if price_vs_vwap is not None:
        if price_vs_vwap < -0.1 and state.bias in (FlowBias.BUY, FlowBias.STRONG_BUY):
            score += 1.5
            reasons.append("below VWAP with buy flow")

    return min(score, 10.0), reasons


def _score_liquidity(
    option_spread_pct: float = 0.0,
    option_volume: int = 0,
    option_oi: int = 0,
) -> tuple[float, list[str]]:
    """Score option liquidity (0-10). Can hard-reject."""
    score = 0.0
    reasons: list[str] = []

    # Spread quality
    if option_spread_pct <= 0:
        score += 1.0  # No data, neutral
    elif option_spread_pct < 3.0:
        score += 4.0
        reasons.append(f"tight spread ({option_spread_pct:.1f}%)")
    elif option_spread_pct < 5.0:
        score += 3.0
    elif option_spread_pct < 8.0:
        score += 1.5
    else:
        score += 0.5
        reasons.append(f"wide spread ({option_spread_pct:.1f}%)")

    # Volume
    if option_volume > 500:
        score += 3.0
        reasons.append(f"high option volume ({option_volume})")
    elif option_volume > 100:
        score += 2.0
    elif option_volume > 10:
        score += 1.0

    # Open interest
    if option_oi > 1000:
        score += 3.0
    elif option_oi > 500:
        score += 2.0
    elif option_oi > 100:
        score += 1.0

    return min(score, 10.0), reasons


def _score_edge(ctx: ORATSContext | None) -> tuple[float, list[str]]:
    """Score ORATS IV edge (0-10)."""
    if ctx is None:
        return 5.0, ["no ORATS data"]

    score = 0.0
    reasons: list[str] = []

    # IV Rank: cheap options = good for buyers (rank < 30)
    if ctx.iv_rank > 0:
        if ctx.iv_rank < 30:
            score += 3.0
            reasons.append(f"IV rank low ({ctx.iv_rank:.0f}) — cheap options")
        elif ctx.iv_rank < 50:
            score += 2.0
        elif ctx.iv_rank < 70:
            score += 1.0
        else:
            score += 0.5
            reasons.append(f"IV rank elevated ({ctx.iv_rank:.0f})")

    # IV-HV spread: negative = options underpriced vs realized
    if ctx.iv_hv_spread != 0:
        if ctx.iv_hv_spread < -0.05:
            score += 3.0
            reasons.append("IV below HV — options underpriced")
        elif ctx.iv_hv_spread < 0:
            score += 2.0
        elif ctx.iv_hv_spread < 0.05:
            score += 1.0
        else:
            score += 0.5

    # Term structure: normal contango is better for short-dated
    if ctx.contango > 0.02:
        score += 2.0
        reasons.append("normal contango — no backwardation risk")
    elif ctx.contango > 0:
        score += 1.5
    elif ctx.contango < -0.02:
        score += 0.5
        reasons.append("backwardation warning")
    else:
        score += 1.0

    # ATM IV reasonableness
    if 0.15 < ctx.atm_iv < 0.60:
        score += 2.0
    elif ctx.atm_iv > 0:
        score += 1.0

    return min(score, 10.0), reasons


def _score_regime(ctx: ORATSContext | None) -> tuple[float, list[str]]:
    """Score market regime for scalp suitability (0-10)."""
    score = 5.0  # Neutral default
    reasons: list[str] = []

    if ctx is None:
        return score, ["no regime data"]

    # IV rank in sweet spot (30-70 = normal regime)
    if ctx.iv_rank > 0:
        if 30 <= ctx.iv_rank <= 70:
            score += 2.0
            reasons.append("IV rank in normal range")
        elif ctx.iv_rank > 80:
            score -= 1.0
            reasons.append("elevated IV — reversal risk")

    # Earnings blackout
    if ctx.in_earnings_blackout:
        score -= 3.0
        reasons.append("EARNINGS BLACKOUT — vol crush risk")
    elif ctx.days_to_earnings is not None and ctx.days_to_earnings <= 5:
        score -= 1.0
        reasons.append(f"earnings in {ctx.days_to_earnings} days")

    # Normal term structure
    if ctx.contango > 0:
        score += 1.0

    return max(0.0, min(score, 10.0)), reasons


def _score_greeks(
    delta: float = 0.0,
    gamma: float = 0.0,
    theta: float = 0.0,
    premium: float = 0.0,
    dte: int = 0,
) -> tuple[float, list[str]]:
    """Score greeks profile for scalp suitability (0-10)."""
    score = 5.0  # Neutral default
    reasons: list[str] = []

    # Delta: near ATM (0.40-0.60) = responsive to underlying moves
    abs_delta = abs(delta)
    if 0.40 <= abs_delta <= 0.60:
        score += 2.0
        reasons.append(f"near-ATM delta ({delta:.2f})")
    elif 0.25 <= abs_delta <= 0.75:
        score += 1.0
    else:
        score -= 0.5

    # Gamma: higher = more price sensitivity (good for scalps)
    if gamma > 0.05:
        score += 2.0
        reasons.append("high gamma")
    elif gamma > 0.02:
        score += 1.0

    # Theta: fast decay on 0DTE is a risk
    if premium > 0 and theta != 0:
        theta_pct = abs(theta) / premium
        if theta_pct > 0.10 and dte <= 1:
            score -= 2.0
            reasons.append("heavy theta decay on 0DTE")
        elif theta_pct > 0.05:
            score -= 1.0

    return max(0.0, min(score, 10.0)), reasons


# ── Main scoring function ─────────────────────────────────────


def score_scalp_signal(
    ticker: str,
    flow_state: TickerFlowState,
    orats_ctx: ORATSContext | None = None,
    *,
    ibs: float | None = None,
    rsi3: float | None = None,
    price_vs_vwap: float | None = None,
    option_spread_pct: float = 0.0,
    option_volume: int = 0,
    option_oi: int = 0,
    option_delta: float = 0.0,
    option_gamma: float = 0.0,
    option_theta: float = 0.0,
    option_premium: float = 0.0,
    option_dte: int = 0,
) -> ScalpSignalScore:
    """Compute composite scalp signal score.

    Returns ScalpSignalScore with total (0-10), per-component breakdown,
    direction, confidence tier, and reasoning.
    """
    result = ScalpSignalScore(ticker=ticker)

    # Hard rejections
    if option_spread_pct > _MAX_SPREAD_PCT and option_spread_pct > 0:
        result.rejected = True
        result.reject_reason = f"spread too wide ({option_spread_pct:.1f}%)"
        return result

    if orats_ctx and orats_ctx.in_earnings_blackout:
        result.rejected = True
        result.reject_reason = "earnings blackout"
        return result

    # Score each component
    flow, flow_reasons = _score_flow(flow_state)
    momentum, mom_reasons = _score_momentum(
        flow_state, ibs, rsi3, price_vs_vwap,
    )
    liquidity, liq_reasons = _score_liquidity(
        option_spread_pct, option_volume, option_oi,
    )
    edge, edge_reasons = _score_edge(orats_ctx)
    regime, regime_reasons = _score_regime(orats_ctx)
    greeks, greek_reasons = _score_greeks(
        option_delta, option_gamma, option_theta, option_premium, option_dte,
    )

    result.flow_score = flow
    result.momentum_score = momentum
    result.liquidity_score = liquidity
    result.edge_score = edge
    result.regime_score = regime
    result.greeks_score = greeks

    # Weighted composite
    result.total = round(
        flow * _W_FLOW
        + momentum * _W_MOMENTUM
        + liquidity * _W_LIQUIDITY
        + edge * _W_EDGE
        + regime * _W_REGIME
        + greeks * _W_GREEKS,
        2,
    )

    # Direction from flow bias
    if flow_state.bias in (FlowBias.BUY, FlowBias.STRONG_BUY):
        result.direction = "long"
    elif flow_state.bias in (FlowBias.SELL, FlowBias.STRONG_SELL):
        result.direction = "short"
    else:
        result.direction = "long"  # Default long for scalps

    # Confidence tier
    if result.total >= 9.0:
        result.confidence = "ULTRA"
    elif result.total >= 8.0:
        result.confidence = "HIGH"
    elif result.total >= 6.5:
        result.confidence = "MEDIUM"
    else:
        result.confidence = "LOW"

    # Collect all reasons
    result.reasons = flow_reasons + mom_reasons + liq_reasons + edge_reasons + regime_reasons + greek_reasons

    return result
