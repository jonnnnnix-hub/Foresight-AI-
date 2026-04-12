"""NEXUS — fuses all signal dimensions into a single conviction score.

Now supports adaptive weights from the learning system. If adaptive
weights exist on disk, they override the default config weights.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.schemas.catalyst import CatalystSignal
from flowedge.scanner.schemas.flow import FlowSentiment, UOASignal
from flowedge.scanner.schemas.iv import IVSignal
from flowedge.scanner.schemas.options import OptionContract, OptionType
from flowedge.scanner.schemas.signals import ContractPick, LottoOpportunity, ScannerResult

logger = structlog.get_logger()


def _determine_direction(
    uoa: UOASignal | None,
    catalyst: CatalystSignal | None,
) -> FlowSentiment:
    """Determine suggested trade direction from available signals."""
    if uoa and uoa.direction != FlowSentiment.NEUTRAL:
        return uoa.direction
    if catalyst and catalyst.net_insider_sentiment == "bullish":
        return FlowSentiment.BULLISH
    if catalyst and catalyst.net_insider_sentiment == "bearish":
        return FlowSentiment.BEARISH
    return FlowSentiment.NEUTRAL


def _generate_entry_criteria(
    uoa: UOASignal | None,
    iv: IVSignal | None,
    catalyst: CatalystSignal | None,
) -> list[str]:
    """Generate actionable entry criteria from signals."""
    criteria: list[str] = []

    if uoa and uoa.strength >= 6.0:
        criteria.append(
            f"UOA confirmed: {uoa.signal_type} with "
            f"${uoa.total_premium:,.0f} premium"
        )
    if iv and iv.is_cheap_premium:
        criteria.append(
            f"IV Rank at {iv.iv_rank.iv_rank:.0f}% — "
            f"premium historically cheap"
        )
    if catalyst and catalyst.days_to_nearest_catalyst is not None:
        criteria.append(
            f"Catalyst in {catalyst.days_to_nearest_catalyst} days"
        )
    if catalyst and catalyst.net_insider_sentiment == "bullish":
        criteria.append(
            f"Insider buying cluster: "
            f"{catalyst.insider_buy_count} buys"
        )

    return criteria


def _generate_risk_flags(
    uoa: UOASignal | None,
    iv: IVSignal | None,
    catalyst: CatalystSignal | None,
) -> list[str]:
    """Flag risks that could undermine the lotto play."""
    flags: list[str] = []

    if iv and iv.iv_rank.iv_rank > 70:
        flags.append("High IV rank — premium is expensive, IV crush risk")
    if iv and not iv.is_contango:
        flags.append("Term structure in backwardation — elevated near-term vol")
    if (
        catalyst
        and catalyst.days_to_nearest_catalyst is not None
        and catalyst.days_to_nearest_catalyst <= 2
    ):
        flags.append("Earnings imminent — IV crush after announcement")
    if catalyst and catalyst.net_insider_sentiment == "bearish":
        flags.append(
            f"Net insider selling: "
            f"{catalyst.insider_sell_count} sells"
        )
    if uoa and uoa.direction == FlowSentiment.NEUTRAL:
        flags.append("Mixed flow direction — no clear directional bias")

    return flags


def _suggest_contracts(
    direction: FlowSentiment,
    uoa: UOASignal | None,
) -> list[OptionContract]:
    """Suggest specific contracts from flow data if available."""
    if not uoa or not uoa.alerts:
        return []

    # Pick the contracts with highest premium from flow alerts
    target_type = OptionType.CALL if direction == FlowSentiment.BULLISH else OptionType.PUT
    matching = [
        a for a in uoa.alerts if a.option_type == target_type
    ]
    if not matching:
        matching = uoa.alerts

    # Return top alerts as suggested contracts
    top = sorted(matching, key=lambda a: a.premium, reverse=True)[:3]
    return [
        OptionContract(
            symbol=a.option_symbol or f"{a.ticker}_{a.strike}_{a.expiration}",
            underlying=a.ticker,
            option_type=a.option_type,
            strike=a.strike,
            expiration=a.expiration,
            volume=a.volume,
            open_interest=a.open_interest,
            source="flow_alert",
        )
        for a in top
    ]


def _build_contract_picks(
    direction: FlowSentiment,
    uoa: UOASignal | None,
    iv: IVSignal | None,
    catalyst: CatalystSignal | None,
) -> list[ContractPick]:
    """Build specific contract recommendations from flow data."""
    if not uoa or not uoa.alerts:
        return []

    target_type = OptionType.CALL if direction != FlowSentiment.BEARISH else OptionType.PUT
    matching = [a for a in uoa.alerts if a.option_type == target_type]
    if not matching:
        matching = uoa.alerts

    # Pick highest-premium contracts with real volume
    significant = [a for a in matching if a.premium >= 10_000 and a.volume >= 50]
    if not significant:
        significant = sorted(matching, key=lambda a: a.premium, reverse=True)[:5]

    picks: list[ContractPick] = []
    for a in sorted(significant, key=lambda x: x.premium, reverse=True)[:3]:
        entry_cost = a.premium / max(a.volume, 1) / 100  # per-share estimate
        max_loss = round(entry_cost * 100, 2)

        reason_parts: list[str] = []
        if a.premium >= 100_000:
            reason_parts.append("block-size premium")
        if a.volume_oi_ratio >= 3:
            reason_parts.append(f"vol/OI {a.volume_oi_ratio:.1f}x")
        if iv and iv.is_cheap_premium:
            reason_parts.append("cheap IV regime")
        if catalyst and catalyst.days_to_nearest_catalyst is not None:
            reason_parts.append(f"catalyst in {catalyst.days_to_nearest_catalyst}d")

        picks.append(
            ContractPick(
                symbol=a.option_symbol,
                option_type=a.option_type.value,
                strike=a.strike,
                expiration=str(a.expiration),
                volume=a.volume,
                open_interest=a.open_interest,
                max_loss_per_contract=max_loss,
                reason="; ".join(reason_parts) if reason_parts else "high flow activity",
            )
        )

    return picks


def score_lottos(
    uoa_signals: list[UOASignal],
    iv_signals: list[IVSignal],
    catalyst_signals: list[CatalystSignal],
    settings: Settings | None = None,
    market_tide: list[dict[str, object]] | None = None,
) -> ScannerResult:
    """Combine all signal types into ranked lotto opportunities.

    Mirrors the scoring/engine.py pattern: weighted dimensions,
    composite score, sorted descending.

    Optional market_tide from UW provides market-wide regime context:
    risk-on boosts bullish scores, risk-off boosts bearish scores.
    """
    settings = settings or get_settings()

    # Market regime from UW Market Tide (if available)
    market_regime_boost = 0.0
    market_regime_label = ""
    if market_tide and len(market_tide) >= 5:
        # Average recent call/put volume ratio across market
        recent = market_tide[-5:]
        total_call = sum(
            float(v) if isinstance(v := t.get("call_volume", 0), (int, float, str)) else 0.0
            for t in recent
        )
        total_put = sum(
            float(v) if isinstance(v := t.get("put_volume", 0), (int, float, str)) else 0.0
            for t in recent
        )
        if total_put > 0:
            mkt_ratio = total_call / total_put
            if mkt_ratio > 1.5:
                market_regime_boost = 0.5
                market_regime_label = f"Market risk-on (C/P {mkt_ratio:.1f}x)"
            elif mkt_ratio < 0.7:
                market_regime_boost = -0.3
                market_regime_label = f"Market risk-off (C/P {mkt_ratio:.1f}x)"

    # Index signals by ticker
    uoa_by_ticker = {s.ticker: s for s in uoa_signals}
    iv_by_ticker = {s.ticker: s for s in iv_signals}
    catalyst_by_ticker = {s.ticker: s for s in catalyst_signals}

    # Union of all tickers
    all_tickers = set(uoa_by_ticker) | set(iv_by_ticker) | set(catalyst_by_ticker)

    opportunities: list[LottoOpportunity] = []
    now = datetime.now()

    for ticker in all_tickers:
        uoa = uoa_by_ticker.get(ticker)
        iv = iv_by_ticker.get(ticker)
        catalyst = catalyst_by_ticker.get(ticker)

        uoa_score = uoa.strength if uoa else 0.0
        iv_score = iv.strength if iv else 0.0
        catalyst_score = catalyst.strength if catalyst else 0.0

        # Use adaptive weights if available, else config defaults
        adaptive = _get_adaptive_weights()
        if adaptive:
            from flowedge.scanner.learning.adaptive import compute_adaptive_score
            base = (
                uoa_score * adaptive.uoa_weight
                + iv_score * adaptive.iv_weight
                + catalyst_score * adaptive.catalyst_weight
                + market_regime_boost
            )
            base = max(0.0, min(10.0, base))
            base_100 = min(100, round(base * 10))
            composite, score_100, adj_notes = compute_adaptive_score(
                base, adaptive,
                uoa_score, iv_score, catalyst_score,
                ticker=ticker, nexus_score_100=base_100,
            )
        else:
            composite = (
                uoa_score * settings.lotto_score_uoa_weight
                + iv_score * settings.lotto_score_iv_weight
                + catalyst_score * settings.lotto_score_catalyst_weight
                + market_regime_boost
            )
            composite = max(0.0, min(10.0, composite))
            score_100 = min(100, round(composite * 10))
            adj_notes = []

        direction = _determine_direction(uoa, catalyst)
        entry_criteria = _generate_entry_criteria(uoa, iv, catalyst)
        risk_flags = _generate_risk_flags(uoa, iv, catalyst)
        suggested = _suggest_contracts(direction, uoa)
        picks = _build_contract_picks(direction, uoa, iv, catalyst)

        rationale_parts: list[str] = []
        if market_regime_label:
            rationale_parts.append(market_regime_label)
        if uoa:
            rationale_parts.append(f"UOA: {uoa.rationale}")
        if iv:
            rationale_parts.append(f"IV: {iv.rationale}")
        if catalyst:
            rationale_parts.append(f"Catalyst: {catalyst.rationale}")
        if adj_notes:
            rationale_parts.extend(adj_notes)

        opportunities.append(
            LottoOpportunity(
                ticker=ticker,
                composite_score=round(composite, 2),
                score_100=score_100,
                uoa_score=uoa_score,
                iv_score=iv_score,
                catalyst_score=catalyst_score,
                uoa_signal=uoa,
                iv_signal=iv,
                catalyst_signal=catalyst,
                suggested_direction=direction,
                suggested_contracts=suggested,
                contract_picks=picks,
                entry_criteria=entry_criteria,
                risk_flags=risk_flags,
                rationale=" | ".join(rationale_parts),
                scanned_at=now,
            )
        )

    opportunities.sort(key=lambda o: o.composite_score, reverse=True)

    scan_id = str(uuid.uuid4())[:12]
    logger.info(
        "lotto_scoring_complete",
        scan_id=scan_id,
        total_tickers=len(all_tickers),
        opportunities=len(opportunities),
    )

    return ScannerResult(
        scan_id=scan_id,
        scanned_at=now,
        tickers_scanned=len(all_tickers),
        opportunities=opportunities,
    )


def _get_adaptive_weights() -> AdaptiveWeights | None:  # type: ignore[name-defined]
    """Try to load adaptive weights. Returns None if unavailable."""
    try:
        from pathlib import Path

        from flowedge.scanner.learning.adaptive import load_weights
        if Path("./data/learning/adaptive_weights.json").exists():
            return load_weights()
    except Exception:
        pass
    return None
