"""AI interpreter — uses Claude to generate trade theses for lotto plays.

Takes scanner signals (UOA, IV, Catalyst, GEX) and asks Claude to:
1. Explain what the smart money is doing
2. Build a narrative for why a big move could happen
3. Assess conviction level
4. Recommend entry, exit, and sizing
"""

from __future__ import annotations

import structlog

from flowedge.agents.llm import call_agent_llm
from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.gex.schemas import GEXProfile
from flowedge.scanner.interpreter.schemas import ConvictionLevel, TradeThesis
from flowedge.scanner.schemas.signals import LottoOpportunity

logger = structlog.get_logger()

INTERPRETER_PROMPT = """You are an expert options trader and market microstructure analyst.

You analyze scanner data to generate concise, actionable trade theses for
short-dated options plays (lotto plays). You explain WHY a setup is
interesting based on the data, not just repeat the numbers.

Rules:
- Be direct and specific. No filler language.
- Distinguish between what the data shows vs your inference.
- If the setup is weak, say so. Do not manufacture conviction.
- Always address position sizing (this is a lotto, not a core position).
- Reference specific data points from the signals provided.
- Think about what the SMART MONEY is doing based on flow data.
- Consider dealer gamma positioning and its effect on price movement.
"""


def _build_opportunity_context(
    opp: LottoOpportunity,
    gex: GEXProfile | None = None,
) -> str:
    """Format all signal data into a context string for Claude."""
    parts: list[str] = [
        f"TICKER: {opp.ticker}",
        f"Composite Score: {opp.composite_score:.1f}/10",
        f"Direction: {opp.suggested_direction.value}",
        "",
    ]

    if opp.uoa_signal:
        uoa = opp.uoa_signal
        parts.append("=== OPTIONS FLOW (UOA) ===")
        parts.append(f"Signal type: {uoa.signal_type}")
        parts.append(f"Flow direction: {uoa.direction.value}")
        parts.append(f"Total premium: ${uoa.total_premium:,.0f}")
        parts.append(f"Call volume: {uoa.call_volume:,} | Put volume: {uoa.put_volume:,}")
        parts.append(f"Call/Put ratio: {uoa.call_put_ratio:.2f}")
        parts.append(f"Alert count: {len(uoa.alerts)}")
        if uoa.dark_pool_trades:
            dp_total = sum(t.notional for t in uoa.dark_pool_trades)
            parts.append(
                f"Dark pool: {len(uoa.dark_pool_trades)} prints, "
                f"${dp_total:,.0f} notional"
            )
        parts.append("")

    if opp.iv_signal:
        iv = opp.iv_signal
        parts.append("=== IMPLIED VOLATILITY ===")
        parts.append(f"IV Rank: {iv.iv_rank.iv_rank:.1f}%")
        parts.append(f"IV Percentile: {iv.iv_rank.iv_percentile:.1f}%")
        parts.append(f"Current IV: {iv.iv_rank.current_iv:.2f}")
        if iv.iv_rank.hv_20:
            parts.append(f"HV20: {iv.iv_rank.hv_20:.4f}")
        if iv.iv_rank.iv_hv_spread is not None:
            parts.append(f"IV-HV Spread: {iv.iv_rank.iv_hv_spread:+.4f}")
        parts.append(f"Regime: {iv.regime.value}")
        parts.append(f"Cheap premium: {'YES' if iv.is_cheap_premium else 'No'}")
        parts.append(f"Term structure: {'Contango' if iv.is_contango else 'BACKWARDATION'}")
        parts.append("")

    if opp.catalyst_signal:
        cat = opp.catalyst_signal
        parts.append("=== CATALYSTS ===")
        if cat.days_to_nearest_catalyst is not None:
            parts.append(f"Days to catalyst: {cat.days_to_nearest_catalyst}")
        if cat.earnings:
            for e in cat.earnings[:3]:
                parts.append(
                    f"Earnings: {e.report_date} ({e.time_of_day}) "
                    f"EPS est: {e.eps_estimate}"
                )
        if cat.expected_move:
            parts.append(f"Expected move: {cat.expected_move.expected_move_pct:.1f}%")
        parts.append(f"Insider sentiment: {cat.net_insider_sentiment}")
        parts.append(
            f"Insider trades: {cat.insider_buy_count} buys / {cat.insider_sell_count} sells"
        )
        if cat.insider_net_value != 0:
            parts.append(f"Insider net value: ${cat.insider_net_value:+,.0f}")
        parts.append("")

    if gex:
        parts.append("=== GAMMA EXPOSURE (GEX) ===")
        parts.append(f"GEX Regime: {gex.regime.value}")
        parts.append(f"Total GEX: {gex.total_gex:,.0f}")
        if gex.gex_flip_price:
            parts.append(f"GEX Flip: ${gex.gex_flip_price:.2f}")
        if gex.max_pain:
            parts.append(f"Max Pain: ${gex.max_pain:.2f}")
        parts.append(f"Lotto favorable: {'YES' if gex.lotto_favorable else 'No'}")
        if gex.support_levels:
            parts.append(f"Support: {', '.join(f'${s:.0f}' for s in gex.support_levels[:3])}")
        if gex.resistance_levels:
            parts.append(
                f"Resistance: {', '.join(f'${r:.0f}' for r in gex.resistance_levels[:3])}"
            )
        parts.append("")

    if opp.entry_criteria:
        parts.append("=== ENTRY CRITERIA ===")
        for c in opp.entry_criteria:
            parts.append(f"  + {c}")
        parts.append("")

    if opp.risk_flags:
        parts.append("=== RISK FLAGS ===")
        for f in opp.risk_flags:
            parts.append(f"  ! {f}")

    return "\n".join(parts)


async def interpret_opportunity(
    opp: LottoOpportunity,
    gex: GEXProfile | None = None,
    settings: Settings | None = None,
) -> TradeThesis:
    """Use Claude to generate a trade thesis for a lotto opportunity."""
    settings = settings or get_settings()

    if not settings.anthropic_api_key:
        logger.warning("no_anthropic_key_for_interpreter")
        return _fallback_thesis(opp, gex)

    context = _build_opportunity_context(opp, gex)

    try:
        result = await call_agent_llm(
            system_prompt=INTERPRETER_PROMPT,
            user_content=(
                f"Generate a trade thesis for this lotto opportunity:\n\n{context}\n\n"
                "Be specific about conviction level, entry timing, and position sizing."
            ),
            output_type=TradeThesis,
        )
        result.ticker = opp.ticker
        return result
    except Exception as e:
        logger.warning("interpreter_llm_failed", ticker=opp.ticker, error=str(e))
        return _fallback_thesis(opp, gex)


def _fallback_thesis(
    opp: LottoOpportunity,
    gex: GEXProfile | None = None,
) -> TradeThesis:
    """Generate a rule-based thesis when Claude is unavailable."""
    # Determine conviction from scores
    if opp.composite_score >= 7.0:
        conviction = ConvictionLevel.HIGH
    elif opp.composite_score >= 5.0:
        conviction = ConvictionLevel.MEDIUM
    elif opp.composite_score >= 3.0:
        conviction = ConvictionLevel.LOW
    else:
        conviction = ConvictionLevel.AVOID

    risks: list[str] = list(opp.risk_flags)

    # Smart money read
    smart_money = "No unusual flow detected."
    if opp.uoa_signal and opp.uoa_signal.strength >= 5.0:
        if opp.uoa_signal.call_put_ratio >= 2.0:
            smart_money = (
                f"Heavy call buying ({opp.uoa_signal.call_put_ratio:.1f}x C/P ratio) "
                f"with ${opp.uoa_signal.total_premium:,.0f} total premium. "
                "Institutions appear to be positioning bullish."
            )
        elif opp.uoa_signal.call_put_ratio <= 0.5:
            smart_money = (
                "Put-heavy flow suggests institutional hedging or bearish positioning."
            )
        else:
            smart_money = (
                f"Mixed flow with ${opp.uoa_signal.total_premium:,.0f} total premium. "
                "No clear directional bias from institutions."
            )

    # IV context
    iv_context = "No IV data available."
    if opp.iv_signal:
        iv = opp.iv_signal
        if iv.is_cheap_premium:
            iv_context = (
                f"IV Rank at {iv.iv_rank.iv_rank:.0f}% — historically cheap premium. "
                "Options are underpriced relative to past ranges, "
                "favorable for long premium lottery plays."
            )
        else:
            iv_context = (
                f"IV Rank at {iv.iv_rank.iv_rank:.0f}% — "
                f"{'expensive' if iv.iv_rank.iv_rank > 70 else 'normal'} premium. "
            )

    # Catalyst narrative
    catalyst = "No near-term catalyst identified."
    if opp.catalyst_signal and opp.catalyst_signal.days_to_nearest_catalyst is not None:
        d = opp.catalyst_signal.days_to_nearest_catalyst
        catalyst = f"Earnings in {d} days — a binary event that could drive an outsized move."
        if opp.catalyst_signal.expected_move:
            em_pct = opp.catalyst_signal.expected_move.expected_move_pct
            catalyst += f" Market implies a {em_pct:.1f}% move."

    # GEX context
    gex_note = "No GEX data available."
    if gex:
        if gex.lotto_favorable:
            gex_note = (
                "Negative GEX regime — dealers will amplify price moves. "
                "Favorable for lottery plays."
            )
        else:
            gex_note = (
                f"{gex.regime.value.title()} GEX — "
                "dealers likely to dampen moves near max pain."
            )

    # Thesis summary
    summary_parts = []
    if opp.iv_signal and opp.iv_signal.is_cheap_premium:
        summary_parts.append("cheap premium")
    if opp.uoa_signal and opp.uoa_signal.strength >= 5:
        summary_parts.append("unusual flow")
    if opp.catalyst_signal and opp.catalyst_signal.days_to_nearest_catalyst:
        summary_parts.append("catalyst ahead")
    if gex and gex.lotto_favorable:
        summary_parts.append("negative GEX")

    thesis = (
        f"{opp.ticker} lotto play — {conviction.value} conviction based on: "
        f"{', '.join(summary_parts) if summary_parts else 'weak signals'}."
    )

    return TradeThesis(
        ticker=opp.ticker,
        conviction=conviction,
        thesis_summary=thesis,
        smart_money_read=smart_money,
        catalyst_narrative=catalyst,
        iv_context=iv_context,
        gex_context=gex_note,
        ideal_entry=(
            "Enter on confirmation of direction. "
            "Wait for volume to confirm the flow signal."
        ),
        target_exit="Take profit at 100%+ gain or exit before catalyst IV crush.",
        stop_logic="Max loss: entire premium. Size accordingly.",
        key_risks=risks,
        why_this_could_work=(
            "Cheap premium + catalyst + flow alignment = asymmetric payoff."
            if conviction in (ConvictionLevel.HIGH, ConvictionLevel.MEDIUM)
            else "Weak signal confluence — not enough edge."
        ),
        why_this_could_fail=(
            "Theta decay eats the premium before the move. "
            "IV crush post-catalyst wipes out extrinsic value."
        ),
        position_sizing_note=(
            "Lotto position: 0.5-1% of portfolio max. "
            "This is a defined-risk trade — you can only lose the premium."
        ),
    )


async def interpret_batch(
    opportunities: list[LottoOpportunity],
    gex_profiles: dict[str, GEXProfile] | None = None,
    max_interpret: int = 5,
    settings: Settings | None = None,
) -> list[TradeThesis]:
    """Generate trade theses for top opportunities."""
    settings = settings or get_settings()
    gex_profiles = gex_profiles or {}
    theses: list[TradeThesis] = []

    for opp in opportunities[:max_interpret]:
        gex = gex_profiles.get(opp.ticker)
        thesis = await interpret_opportunity(opp, gex, settings)
        theses.append(thesis)
        logger.info(
            "thesis_generated",
            ticker=opp.ticker,
            conviction=thesis.conviction.value,
        )

    return theses
