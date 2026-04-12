"""COMMITTEE — specialist agents that dissect every losing trade.

Five specialists analyze each loss from different angles:
1. Flow Analyst — Was the UOA signal genuine or noise?
2. Volatility Analyst — Did IV regime help or hurt?
3. Catalyst Analyst — Did the catalyst thesis play out?
4. Timing Analyst — Was the entry/exit timing wrong?
5. Risk Analyst — Were risk flags ignored?

After individual analysis, the committee reaches consensus on
root cause and generates concrete model improvement suggestions.
"""

from __future__ import annotations

import structlog

from flowedge.agents.llm import call_agent_llm
from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.learning.schemas import (
    FailureCategory,
    SpecialistVerdict,
    TradePostMortem,
)
from flowedge.scanner.performance.schemas import SimulatedTrade

logger = structlog.get_logger()

# ──────────────────────────────────────────────────────────────
# Specialist system prompts
# ────────────────────��─────────────────────────────────────────

FLOW_ANALYST_PROMPT = """You are an expert options flow analyst specializing in
detecting genuine institutional activity versus noise.

You analyze losing trades to determine whether the unusual options activity (UOA)
signal that triggered the trade was:
- Genuine smart money positioning that was correct but the trade management failed
- Hedging activity that was misread as directional
- Market-maker inventory rebalancing, not directional flow
- Retail-driven volume spike with no edge
- Legitimate flow that was front-run or already priced in

Be specific and cite data. Rate your confidence 0-1."""

VOLATILITY_ANALYST_PROMPT = """You are an expert volatility trader who specializes
in IV dynamics, term structure, and volatility surface analysis.

You analyze losing trades to determine whether the volatility regime
contributed to the loss:
- Was IV rank too high at entry (premium too expensive)?
- Did IV crush destroy the position after a catalyst?
- Was the term structure misleading (backwardation/contango misread)?
- Was HV/IV spread providing a false signal?
- Was the option's vega exposure appropriate for the thesis?

Be specific about what the vol surface was telling us vs what we assumed."""

CATALYST_ANALYST_PROMPT = """You are an expert catalyst analyst who specializes in
earnings, corporate events, and macro catalysts.

You analyze losing trades to determine whether the catalyst thesis was valid:
- Did the catalyst occur as expected?
- Was the expected move estimate accurate?
- Did insiders know something the model missed?
- Was the catalyst already priced in?
- Did a competing catalyst (macro, sector rotation) overwhelm the thesis?

Focus on whether the thesis was fundamentally wrong vs execution failure."""

TIMING_ANALYST_PROMPT = """You are an expert trade timing specialist focused on
entry/exit optimization for short-dated options.

You analyze losing trades to determine whether timing was the root cause:
- Was the entry too early (before momentum confirmation)?
- Was the entry too late (after the move already happened)?
- Should the stop loss have been tighter or wider?
- Was the hold period appropriate for the thesis?
- Did the exit miss a recovery that would have turned profit?

Focus on what timing adjustments would have changed the outcome."""

RISK_ANALYST_PROMPT = """You are a risk management specialist for options trading.

You analyze losing trades to determine whether risk controls failed:
- Were existing risk flags ignored or underweighted?
- Was position sizing appropriate?
- Was the score inflated relative to true signal quality?
- Were correlated positions creating hidden concentration risk?
- What filter or rule would have prevented this loss?

Be concrete: what specific filter or threshold change would prevent this type of loss?"""


class SpecialistAnalyzer:
    """One specialist that analyzes a trade from a specific angle."""

    def __init__(self, name: str, system_prompt: str) -> None:
        self.name = name
        self.prompt = system_prompt

    async def analyze(
        self,
        trade: SimulatedTrade,
        context: str,
    ) -> SpecialistVerdict:
        """Analyze a losing trade from this specialist's perspective."""
        try:
            result = await call_agent_llm(
                system_prompt=self.prompt,
                user_content=(
                    f"Analyze this LOSING trade and diagnose the root cause:\n\n"
                    f"{context}\n\n"
                    f"Classify the root cause as ONE of:\n"
                    f"- flow_misleading: UOA was noise\n"
                    f"- iv_crush: Vol collapse destroyed premium\n"
                    f"- bad_timing: Right idea, wrong entry\n"
                    f"- catalyst_miss: Catalyst didn't deliver\n"
                    f"- wrong_direction: Momentum read was backwards\n"
                    f"- theta_decay: Time ate the position\n"
                    f"- market_regime: Broad market moved against\n"
                    f"- overscored: Model gave false conviction\n"
                    f"- low_liquidity: Thin market, bad fills\n"
                    f"- unknown: Cannot determine\n\n"
                    f"Return your diagnosis, root cause classification, "
                    f"confidence (0-1), specific evidence, and one concrete "
                    f"recommendation for model improvement."
                ),
                output_type=SpecialistVerdict,
            )
            result.specialist = self.name
            return result
        except Exception as e:
            logger.warning(
                "specialist_analysis_failed",
                specialist=self.name,
                error=str(e),
            )
            return SpecialistVerdict(
                specialist=self.name,
                diagnosis=f"Analysis failed: {e}",
                root_cause=FailureCategory.UNKNOWN,
                confidence=0.0,
            )


# ──────────────────────────────────────────────────────────────
# Build the specialist committee
# ─��─────────────────────���──────────────────────────────────────

def _build_committee() -> list[SpecialistAnalyzer]:
    """Create the five-specialist analysis committee."""
    return [
        SpecialistAnalyzer("flow_analyst", FLOW_ANALYST_PROMPT),
        SpecialistAnalyzer("volatility_analyst", VOLATILITY_ANALYST_PROMPT),
        SpecialistAnalyzer("catalyst_analyst", CATALYST_ANALYST_PROMPT),
        SpecialistAnalyzer("timing_analyst", TIMING_ANALYST_PROMPT),
        SpecialistAnalyzer("risk_analyst", RISK_ANALYST_PROMPT),
    ]


def _format_trade_context(trade: SimulatedTrade) -> str:
    """Build rich context string for specialist analysis."""
    parts = [
        f"TRADE ID: {trade.trade_id}",
        f"Ticker: {trade.ticker}",
        f"Direction: {trade.direction}",
        f"Option type: {trade.option_type}",
        f"Strike: ${trade.strike:.2f}",
        f"Entry date: {trade.entry_date}",
        f"Exit date: {trade.exit_date or 'still open'}",
        f"Entry underlying: ${trade.entry_underlying:.2f}",
        f"Exit underlying: ${trade.exit_underlying or 0:.2f}",
        f"Entry premium: ${trade.entry_premium:.4f}",
        f"Exit premium: ${trade.exit_premium or 0:.4f}",
        f"Contracts: {trade.contracts}",
        f"Cost basis: ${trade.cost_basis:.2f}",
        f"P&L: ${trade.pnl_dollars:.2f} ({trade.pnl_pct:+.1f}%)",
        f"Hold days: {trade.hold_days}",
        f"NEXUS score at entry: {trade.nexus_score}/100",
        f"Exit reason: {trade.exit_reason}",
    ]

    # Add price movement context
    if trade.entry_underlying > 0 and trade.exit_underlying:
        move_pct = (
            (trade.exit_underlying - trade.entry_underlying)
            / trade.entry_underlying * 100
        )
        parts.append(f"Underlying move: {move_pct:+.2f}%")

        # Was direction right?
        if trade.direction == "bullish":
            parts.append(
                f"Direction correct: "
                f"{'YES' if move_pct > 0 else 'NO — stock moved against thesis'}"
            )
        else:
            parts.append(
                f"Direction correct: "
                f"{'YES' if move_pct < 0 else 'NO — stock moved against thesis'}"
            )

    return "\n".join(parts)


# ──────────────────────────��───────────────────────────────────
# Consensus synthesis
# ─────────────────────���────────────────────────────────────────

CONSENSUS_PROMPT = """You are the chief strategist synthesizing specialist analyses
of a losing trade. Five specialists have each given their diagnosis.

Your job:
1. Weigh each specialist's confidence and evidence quality
2. Determine the CONSENSUS root cause
3. Write a clear thesis explaining what happened at entry vs what actually happened
4. Extract the single most important lesson
5. Decide if this trade SHOULD have been filtered out entirely
6. If so, describe the specific filter that would have caught it

Be precise. No hedging. State what went wrong and what must change."""


async def _synthesize_consensus(
    trade: SimulatedTrade,
    verdicts: list[SpecialistVerdict],
    context: str,
) -> TradePostMortem:
    """Synthesize specialist verdicts into a consensus post-mortem."""
    verdict_text = "\n\n".join(
        f"=== {v.specialist.upper()} ===\n"
        f"Diagnosis: {v.diagnosis}\n"
        f"Root cause: {v.root_cause.value}\n"
        f"Confidence: {v.confidence:.0%}\n"
        f"Evidence: {'; '.join(v.evidence)}\n"
        f"Recommendation: {v.recommendation}"
        for v in verdicts
    )

    try:
        result = await call_agent_llm(
            system_prompt=CONSENSUS_PROMPT,
            user_content=(
                f"TRADE CONTEXT:\n{context}\n\n"
                f"SPECIALIST ANALYSES:\n{verdict_text}\n\n"
                f"Synthesize into a final post-mortem with consensus root cause, "
                f"thesis at entry, what actually happened, key lesson, and "
                f"whether this trade should have been filtered."
            ),
            output_type=TradePostMortem,
        )
        # Preserve trade metadata
        result.trade_id = trade.trade_id
        result.ticker = trade.ticker
        result.entry_date = str(trade.entry_date)
        result.exit_date = str(trade.exit_date or "")
        result.pnl_pct = trade.pnl_pct
        result.nexus_score = trade.nexus_score
        result.direction = trade.direction
        result.option_type = trade.option_type
        result.strike = trade.strike
        result.verdicts = verdicts
        return result
    except Exception as e:
        logger.warning("consensus_synthesis_failed", error=str(e))
        return _rule_based_post_mortem(trade, verdicts)


def _rule_based_post_mortem(
    trade: SimulatedTrade,
    verdicts: list[SpecialistVerdict],
) -> TradePostMortem:
    """Fallback post-mortem when Claude is unavailable."""
    # Vote on root cause
    cause_votes: dict[FailureCategory, float] = {}
    for v in verdicts:
        cause_votes[v.root_cause] = (
            cause_votes.get(v.root_cause, 0) + v.confidence
        )

    consensus = (
        max(cause_votes, key=lambda k: cause_votes[k])
        if cause_votes else FailureCategory.UNKNOWN
    )
    consensus_conf = cause_votes.get(consensus, 0) / max(len(verdicts), 1)

    return TradePostMortem(
        trade_id=trade.trade_id,
        ticker=trade.ticker,
        entry_date=str(trade.entry_date),
        exit_date=str(trade.exit_date or ""),
        pnl_pct=trade.pnl_pct,
        nexus_score=trade.nexus_score,
        direction=trade.direction,
        option_type=trade.option_type,
        strike=trade.strike,
        verdicts=verdicts,
        consensus_cause=consensus,
        consensus_confidence=round(consensus_conf, 2),
        thesis_at_entry=(
            f"Score {trade.nexus_score}/100 {trade.direction} "
            f"{trade.option_type} on {trade.ticker}"
        ),
        what_actually_happened=(
            f"Lost {trade.pnl_pct:.1f}% over {trade.hold_days} days. "
            f"Exit: {trade.exit_reason}"
        ),
        key_lesson=f"Primary failure: {consensus.value}",
        should_have_been_filtered=trade.nexus_score < 40,
        suggested_filter=(
            "Filter trades with score < 40"
            if trade.nexus_score < 40
            else ""
        ),
    )


# ──────────────────────────────────���───────────────────────────
# Public API
# ────���─────────────────────────────────────────────────────────

async def analyze_loss(
    trade: SimulatedTrade,
    settings: Settings | None = None,
) -> TradePostMortem:
    """Run the full specialist committee on a single losing trade."""
    settings = settings or get_settings()
    context = _format_trade_context(trade)

    if not settings.anthropic_api_key:
        return _rule_based_post_mortem(trade, [])

    committee = _build_committee()
    verdicts: list[SpecialistVerdict] = []

    for specialist in committee:
        verdict = await specialist.analyze(trade, context)
        verdicts.append(verdict)
        logger.info(
            "specialist_verdict",
            specialist=specialist.name,
            ticker=trade.ticker,
            cause=verdict.root_cause.value,
            confidence=verdict.confidence,
        )

    post_mortem = await _synthesize_consensus(trade, verdicts, context)

    logger.info(
        "post_mortem_complete",
        trade_id=trade.trade_id,
        ticker=trade.ticker,
        consensus=post_mortem.consensus_cause.value,
        should_filter=post_mortem.should_have_been_filtered,
    )
    return post_mortem


async def analyze_losses_batch(
    trades: list[SimulatedTrade],
    max_analyze: int = 20,
    settings: Settings | None = None,
) -> list[TradePostMortem]:
    """Analyze a batch of losing trades through the committee."""
    settings = settings or get_settings()
    losses = [t for t in trades if t.pnl_dollars < 0]

    # Prioritize worst losses and highest-scored failures
    losses.sort(key=lambda t: (t.nexus_score, -abs(t.pnl_dollars)), reverse=True)

    post_mortems: list[TradePostMortem] = []
    for trade in losses[:max_analyze]:
        pm = await analyze_loss(trade, settings)
        post_mortems.append(pm)

    logger.info(
        "batch_analysis_complete",
        losses_analyzed=len(post_mortems),
        total_losses=len(losses),
    )
    return post_mortems
