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
- CRITICAL: Was this a PREMATURE STOP — the trade direction was correct
  but the stop loss triggered before the winning move? If the underlying
  eventually moved in the thesis direction after exit, this was a premature stop.

If the direction was correct but the trade was stopped out, classify as
'premature_stop' not 'bad_timing'. This distinction is critical for stop loss tuning.

Focus on what timing adjustments would have changed the outcome."""

RISK_ANALYST_PROMPT = """You are a risk management specialist for options trading.

You analyze losing trades to determine whether risk controls failed:
- Were existing risk flags ignored or underweighted?
- Was position sizing appropriate?
- Was the score inflated relative to true signal quality?
- Were correlated positions creating hidden concentration risk?
- What filter or rule would have prevented this loss?

Be concrete: what specific filter or threshold change would prevent this type of loss?"""

STOP_LOSS_ANALYST_PROMPT = """You are a stop-loss optimization specialist for options.

Your SOLE focus is analyzing whether the stop-loss parameters were correct:

1. HARD STOP: Was the -35% hard stop too tight or too loose?
   - If the underlying moved in the right direction after the stop hit,
     the stop was too tight.
   - Calculate what stop level would have held through the drawdown.

2. TRAILING STOP: Was the 35% trailing stop from max premium appropriate?
   - For volatile names, trailing stops often trigger on normal retracements.
   - Was the max premium a spike that made the trail too aggressive?

3. TIME STOP: Was the 9-day max hold too short?
   - If the move happened on day 10-12, the time exit was premature.
   - Some strategies need more time (vol squeeze: 12 days, trend: 10 days).

4. SHOULD-HAVE-BEEN-WIN ANALYSIS:
   - If the direction was correct (underlying moved favorably after exit),
     calculate the hypothetical P&L if stops were:
     - 10% wider (e.g., -45% instead of -35%)
     - 20% wider
     - Different trailing distance
   - Report the "missed profit" from premature stopping.

For EVERY trade, classify: was this a legitimate stop (correct exit) or
a PREMATURE STOP (should have held longer)?

If premature, give the SPECIFIC stop parameters that would have produced a win."""


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
                    f"- premature_stop: Direction was right but stopped out"
                    f" before the winning move materialized\n"
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
    """Create the six-specialist analysis committee.

    v2: Added stop_loss_analyst for premature stop detection.
    """
    return [
        SpecialistAnalyzer("flow_analyst", FLOW_ANALYST_PROMPT),
        SpecialistAnalyzer("volatility_analyst", VOLATILITY_ANALYST_PROMPT),
        SpecialistAnalyzer("catalyst_analyst", CATALYST_ANALYST_PROMPT),
        SpecialistAnalyzer("timing_analyst", TIMING_ANALYST_PROMPT),
        SpecialistAnalyzer("risk_analyst", RISK_ANALYST_PROMPT),
        SpecialistAnalyzer("stop_loss_analyst", STOP_LOSS_ANALYST_PROMPT),
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
            dir_correct = move_pct > 0
            parts.append(
                f"Direction correct: "
                f"{'YES' if dir_correct else 'NO — stock moved against thesis'}"
            )
        else:
            dir_correct = move_pct < 0
            parts.append(
                f"Direction correct: "
                f"{'YES' if dir_correct else 'NO — stock moved against thesis'}"
            )

        # Stop-loss analysis context
        exit_reason = trade.exit_reason or ""
        if "hard_stop" in exit_reason or "trailing_stop" in exit_reason:
            parts.append("\n--- STOP LOSS ANALYSIS ---")
            parts.append(f"Exit triggered by: {exit_reason}")
            parts.append(f"Premium P&L at exit: {trade.pnl_pct:+.1f}%")
            if dir_correct:
                parts.append(
                    "⚠ DIRECTION WAS CORRECT but stopped out — "
                    "likely premature stop"
                )
            parts.append(
                f"Entry premium: ${trade.entry_premium:.4f}"
            )
            parts.append(
                f"Exit premium: ${trade.exit_premium or 0:.4f}"
            )
            if trade.hold_days < 5:
                parts.append(
                    f"Held only {trade.hold_days} days — "
                    f"stop may have triggered on normal volatility"
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
    """Fallback post-mortem when Claude is unavailable.

    v2: Uses weighted specialist voting based on specialist accuracy.
    Detects premature stops from trade data without LLM.
    """
    # Load specialist weights for weighted voting
    from flowedge.scanner.learning.adaptive import load_weights

    weights = load_weights()
    spec_weights: dict[str, float] = {}
    for sa in weights.specialist_accuracy:
        spec_weights[sa.specialist_name] = sa.weight_in_consensus

    # Weighted vote on root cause
    cause_votes: dict[FailureCategory, float] = {}
    for v in verdicts:
        w = spec_weights.get(v.specialist, 0.15)
        cause_votes[v.root_cause] = (
            cause_votes.get(v.root_cause, 0) + v.confidence * w
        )

    # Auto-detect premature stop from trade data
    exit_reason = trade.exit_reason or ""
    is_stop_exit = "hard_stop" in exit_reason or "trailing_stop" in exit_reason
    direction_correct = False
    if trade.entry_underlying > 0 and trade.exit_underlying:
        move_pct = (
            (trade.exit_underlying - trade.entry_underlying)
            / trade.entry_underlying * 100
        )
        direction_correct = (
            (trade.direction == "bullish" and move_pct > 0.5)
            or (trade.direction == "bearish" and move_pct < -0.5)
        )

    if is_stop_exit and direction_correct:
        # Direction was correct but stop killed the trade
        cause_votes[FailureCategory.PREMATURE_STOP] = (
            cause_votes.get(FailureCategory.PREMATURE_STOP, 0) + 0.8
        )

    consensus = (
        max(cause_votes, key=lambda k: cause_votes[k])
        if cause_votes else FailureCategory.UNKNOWN
    )
    total_weight = sum(cause_votes.values())
    consensus_conf = (
        cause_votes.get(consensus, 0) / total_weight if total_weight > 0 else 0.0
    )

    # Generate appropriate lesson based on cause
    if consensus == FailureCategory.PREMATURE_STOP:
        key_lesson = (
            f"Direction was correct but {exit_reason} triggered prematurely. "
            f"Consider wider stops for {trade.ticker} or this strategy."
        )
        suggested_filter = (
            f"Widen stop-loss for trades matching this pattern. "
            f"Hold days: {trade.hold_days}, Exit reason: {exit_reason}"
        )
    else:
        key_lesson = f"Primary failure: {consensus.value}"
        suggested_filter = (
            "Filter trades with score < 40"
            if trade.nexus_score < 40
            else ""
        )

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
        key_lesson=key_lesson,
        should_have_been_filtered=trade.nexus_score < 40,
        suggested_filter=suggested_filter,
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
