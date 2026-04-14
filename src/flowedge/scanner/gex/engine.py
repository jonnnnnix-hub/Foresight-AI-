"""VORTEX — maps dealer gamma forces that pin, repel, or accelerate price.

Uses:
- Orats live strikes for per-strike OI and greeks
- Unusual Whales greeks endpoint for charm/vanna
- UW max pain data
- UW net premium ticks for intraday flow
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.gex.schemas import GEXProfile, GEXRegime, StrikeLevel
from flowedge.scanner.providers.registry import ProviderRegistry

logger = structlog.get_logger()


def _compute_net_gamma(
    strikes_data: list[dict[str, Any]],
    underlying_price: float,
) -> list[StrikeLevel]:
    """Compute net gamma at each strike from Orats strike data.

    Dealer gamma:
    - Dealers are typically short calls → long gamma on calls
    - Dealers are typically long puts → short gamma on puts
    - Net GEX = call_gamma * call_OI * 100 - put_gamma * put_OI * 100
    """
    levels: list[StrikeLevel] = []

    for row in strikes_data:
        strike = float(row.get("strike", 0))
        if strike <= 0:
            continue

        call_gamma = abs(float(row.get("callGamma", row.get("smoothSmvGamma", 0))))
        put_gamma = abs(float(row.get("putGamma", row.get("smoothSmvGamma", 0))))
        call_oi = int(row.get("callOpenInt", 0))
        put_oi = int(row.get("putOpenInt", 0))

        # Dealer GEX: long gamma from sold calls, short gamma from sold puts
        call_gex = call_gamma * call_oi * 100 * underlying_price
        put_gex = put_gamma * put_oi * 100 * underlying_price
        net = call_gex - put_gex

        levels.append(
            StrikeLevel(
                strike=strike,
                call_gamma=round(call_gex, 2),
                put_gamma=round(put_gex, 2),
                net_gamma=round(net, 2),
                call_oi=call_oi,
                put_oi=put_oi,
            )
        )

    return sorted(levels, key=lambda lv: abs(lv.net_gamma), reverse=True)


def _find_gex_flip(
    levels: list[StrikeLevel], underlying_price: float
) -> float | None:
    """Find the price where GEX flips from positive to negative.

    This is the critical level: above = pinned, below = volatile.
    """
    sorted_by_strike = sorted(levels, key=lambda lv: lv.strike)

    for i in range(len(sorted_by_strike) - 1):
        curr = sorted_by_strike[i]
        nxt = sorted_by_strike[i + 1]
        if curr.net_gamma > 0 and nxt.net_gamma < 0:
            return round((curr.strike + nxt.strike) / 2, 2)
        if curr.net_gamma < 0 and nxt.net_gamma > 0:
            return round((curr.strike + nxt.strike) / 2, 2)

    return None


def _classify_regime(
    total_gex: float, underlying_price: float
) -> GEXRegime:
    """Classify gamma regime based on net exposure."""
    # Normalize by price to make comparable across tickers
    normalized = total_gex / max(underlying_price, 1)
    if normalized > 1_000_000:
        return GEXRegime.POSITIVE
    if normalized < -1_000_000:
        return GEXRegime.NEGATIVE
    return GEXRegime.NEUTRAL


def _score_gex_for_lottos(
    regime: GEXRegime,
    underlying_price: float,
    gex_flip: float | None,
    max_pain: float | None,
) -> tuple[float, bool, str]:
    """Score how favorable GEX is for lotto plays.

    Negative GEX = dealers amplify moves = lottos benefit.
    Positive GEX = dealers dampen moves = lottos die.
    """
    score = 0.0
    favorable = False
    parts: list[str] = []

    if regime == GEXRegime.NEGATIVE:
        score += 4.0
        favorable = True
        parts.append("Negative GEX — dealers amplify moves")
    elif regime == GEXRegime.NEUTRAL:
        score += 2.0
        parts.append("Neutral GEX — mixed dealer positioning")
    else:
        score += 0.5
        parts.append("Positive GEX — dealers dampen moves")

    # Near GEX flip = potential regime change
    if gex_flip and underlying_price > 0:
        dist_pct = abs(underlying_price - gex_flip) / underlying_price * 100
        if dist_pct < 2.0:
            score += 3.0
            parts.append(f"Near GEX flip at ${gex_flip:.0f} ({dist_pct:.1f}% away)")
        elif dist_pct < 5.0:
            score += 1.5
            parts.append(f"GEX flip at ${gex_flip:.0f} ({dist_pct:.1f}% away)")

    # Max pain distance = potential magnet or breakout
    if max_pain and underlying_price > 0:
        mp_dist = abs(underlying_price - max_pain) / underlying_price * 100
        if mp_dist > 5.0:
            score += 2.0
            parts.append(
                f"Far from max pain ${max_pain:.0f} ({mp_dist:.1f}%) — breakout zone"
            )
        else:
            parts.append(f"Near max pain ${max_pain:.0f} — pin risk")

    return min(score, 10.0), favorable, "; ".join(parts)


async def compute_gex_profile(
    ticker: str,
    registry: ProviderRegistry,
    settings: Settings | None = None,
) -> GEXProfile:
    """Compute full GEX profile for a ticker.

    Combines Orats strike data with UW max pain and greeks.
    """
    settings = settings or get_settings()

    # Get strike data from Orats (has OI + greeks per strike)
    orats = registry.get_iv_provider()
    try:
        strikes_data = await orats.get_live_strikes(ticker)  # type: ignore[attr-defined]
    except (AttributeError, Exception) as e:
        logger.warning("orats_strikes_failed", ticker=ticker, error=str(e))
        strikes_data = []

    # Get underlying price from Orats cores
    underlying_price = 0.0
    try:
        cores = await orats.get_cores(ticker)  # type: ignore[attr-defined]
        underlying_price = float(cores.get("stockPrice", cores.get("pxAtmIv", 0)))
    except (AttributeError, Exception):
        pass

    # Get max pain from UW
    max_pain_val: float | None = None
    try:
        uw = registry.get_flow_provider()
        mp_data = await uw.get_max_pain(ticker)  # type: ignore[attr-defined]
        if mp_data:
            # Nearest expiry max pain
            max_pain_val = float(mp_data[0].get("max_pain", 0))
    except (AttributeError, Exception):
        pass

    # Compute gamma levels
    levels = _compute_net_gamma(strikes_data, underlying_price)

    # Total GEX
    total_gex = sum(lv.net_gamma for lv in levels)

    # GEX flip point
    gex_flip = _find_gex_flip(levels, underlying_price)

    # Classify regime
    regime = _classify_regime(total_gex, underlying_price)

    # Support/resistance from gamma concentration
    top_levels = levels[:20]  # Top 20 by absolute gamma
    support = sorted(
        [lv.strike for lv in top_levels if lv.put_gamma > lv.call_gamma],
        reverse=True,
    )[:5]
    resistance = sorted(
        [lv.strike for lv in top_levels if lv.call_gamma > lv.put_gamma],
    )[:5]

    # Score for lottos
    score, favorable, rationale = _score_gex_for_lottos(
        regime, underlying_price, gex_flip, max_pain_val
    )

    # Mark max pain and flip levels
    for level in levels:
        if max_pain_val and abs(level.strike - max_pain_val) < 1.0:
            level.is_max_pain = True
        if gex_flip and abs(level.strike - gex_flip) < 1.0:
            level.is_gex_flip = True

    # Pin risk zone: within 2% of max pain
    pin_zone = None
    if max_pain_val and underlying_price > 0:
        spread = underlying_price * 0.02
        pin_zone = (
            round(max_pain_val - spread, 2),
            round(max_pain_val + spread, 2),
        )

    # Breakout levels: first strike with high negative gamma beyond current price
    breakout_above = None
    breakout_below = None
    for lv in sorted(levels, key=lambda x: x.strike):
        if lv.strike > underlying_price and lv.net_gamma < 0 and breakout_above is None:
            breakout_above = lv.strike
        if lv.strike < underlying_price and lv.net_gamma < 0:
            breakout_below = lv.strike

    logger.info(
        "gex_computed",
        ticker=ticker,
        regime=regime.value,
        total_gex=round(total_gex),
        flip=gex_flip,
        max_pain=max_pain_val,
        score=score,
    )

    return GEXProfile(
        ticker=ticker,
        underlying_price=underlying_price,
        regime=regime,
        total_gex=round(total_gex, 2),
        gex_flip_price=gex_flip,
        max_pain=max_pain_val,
        key_levels=levels[:30],
        support_levels=support,
        resistance_levels=resistance,
        pin_risk_zone=pin_zone,
        breakout_above=breakout_above,
        breakout_below=breakout_below,
        lotto_favorable=favorable,
        strength=score,
        rationale=rationale,
        computed_at=datetime.now(),
    )
