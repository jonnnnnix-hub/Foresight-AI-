"""NEXUS — fuses all signal dimensions into a single conviction score."""

from __future__ import annotations

import uuid
from datetime import datetime

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.schemas.catalyst import CatalystSignal
from flowedge.scanner.schemas.flow import FlowSentiment, UOASignal
from flowedge.scanner.schemas.iv import IVSignal
from flowedge.scanner.schemas.options import OptionContract, OptionType
from flowedge.scanner.schemas.signals import LottoOpportunity, ScannerResult

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


def score_lottos(
    uoa_signals: list[UOASignal],
    iv_signals: list[IVSignal],
    catalyst_signals: list[CatalystSignal],
    settings: Settings | None = None,
) -> ScannerResult:
    """Combine all signal types into ranked lotto opportunities.

    Mirrors the scoring/engine.py pattern: weighted dimensions,
    composite score, sorted descending.
    """
    settings = settings or get_settings()

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

        composite = (
            uoa_score * settings.lotto_score_uoa_weight
            + iv_score * settings.lotto_score_iv_weight
            + catalyst_score * settings.lotto_score_catalyst_weight
        )

        direction = _determine_direction(uoa, catalyst)
        entry_criteria = _generate_entry_criteria(uoa, iv, catalyst)
        risk_flags = _generate_risk_flags(uoa, iv, catalyst)
        suggested = _suggest_contracts(direction, uoa)

        rationale_parts: list[str] = []
        if uoa:
            rationale_parts.append(f"UOA: {uoa.rationale}")
        if iv:
            rationale_parts.append(f"IV: {iv.rationale}")
        if catalyst:
            rationale_parts.append(f"Catalyst: {catalyst.rationale}")

        opportunities.append(
            LottoOpportunity(
                ticker=ticker,
                composite_score=round(composite, 2),
                uoa_score=uoa_score,
                iv_score=iv_score,
                catalyst_score=catalyst_score,
                uoa_signal=uoa,
                iv_signal=iv,
                catalyst_signal=catalyst,
                suggested_direction=direction,
                suggested_contracts=suggested,
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
