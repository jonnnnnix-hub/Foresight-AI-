"""PATTERN EXTRACTOR — finds recurring failure modes across post-mortems.

Aggregates the committee's individual trade analyses to identify
systematic patterns that the scoring model gets wrong, then
generates concrete weight/threshold adjustments.
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime

import structlog

from flowedge.agents.llm import call_agent_llm
from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.learning.schemas import (
    AdaptiveWeights,
    FailureCategory,
    LearningInsight,
    ModelRefinement,
    ScoringRule,
    TradePostMortem,
    WeightAdjustment,
)
from flowedge.scanner.performance.schemas import (
    PerformanceReport,
    SimulatedTrade,
    TradeResult,
)

logger = structlog.get_logger()


# ──────────────────────────────────────────────────────────────
# Pattern extraction from post-mortems
# ──────────────────────────────────────────────────────────────

def extract_failure_distribution(
    post_mortems: list[TradePostMortem],
) -> dict[str, int]:
    """Count how often each failure category appears."""
    counts = Counter(pm.consensus_cause.value for pm in post_mortems)
    return dict(counts.most_common())


def extract_insights(
    post_mortems: list[TradePostMortem],
    trades: list[SimulatedTrade],
) -> list[LearningInsight]:
    """Extract actionable insights from post-mortem patterns."""
    insights: list[LearningInsight] = []
    closed = [t for t in trades if t.result != TradeResult.OPEN]
    if not closed:
        return insights

    # ── Pattern 1: Low-score trades that keep losing ──
    low_score_losses = [
        pm for pm in post_mortems if pm.nexus_score < 40
    ]
    low_score_total = [t for t in closed if t.nexus_score < 40]
    if low_score_losses and low_score_total:
        low_wr = sum(
            1 for t in low_score_total if t.result == TradeResult.WIN
        ) / len(low_score_total)
        if low_wr < 0.25:
            avg_loss = sum(t.pnl_pct for t in low_score_total) / len(low_score_total)
            insights.append(LearningInsight(
                insight_id=f"INS-{uuid.uuid4().hex[:8]}",
                category=FailureCategory.OVERSCORED,
                pattern=(
                    f"Trades with score < 40 have {low_wr:.0%} win rate "
                    f"and avg {avg_loss:+.1f}% P&L. These should be filtered."
                ),
                frequency=len(low_score_losses),
                avg_loss_when_present=round(avg_loss, 2),
                suggested_action="Raise minimum entry score to 40+",
                evidence_trade_ids=[pm.trade_id for pm in low_score_losses],
                confidence=0.85,
            ))

    # ── Pattern 2: Directional errors ──
    wrong_dir = [
        pm for pm in post_mortems
        if pm.consensus_cause == FailureCategory.WRONG_DIRECTION
    ]
    if len(wrong_dir) >= 3:
        insights.append(LearningInsight(
            insight_id=f"INS-{uuid.uuid4().hex[:8]}",
            category=FailureCategory.WRONG_DIRECTION,
            pattern=(
                f"{len(wrong_dir)} trades failed due to wrong direction. "
                f"Momentum signal is not reliable enough alone."
            ),
            frequency=len(wrong_dir),
            avg_loss_when_present=round(
                sum(pm.pnl_pct for pm in wrong_dir) / len(wrong_dir), 2
            ),
            suggested_action=(
                "Require UOA flow confirmation before entering "
                "momentum-only trades. Add volume-weighted direction check."
            ),
            evidence_trade_ids=[pm.trade_id for pm in wrong_dir],
            confidence=0.7,
        ))

    # ── Pattern 3: Theta decay kills ──
    theta_kills = [
        pm for pm in post_mortems
        if pm.consensus_cause == FailureCategory.THETA_DECAY
    ]
    if len(theta_kills) >= 2:
        avg_hold = sum(
            t.hold_days for t in closed
            if t.trade_id in {pm.trade_id for pm in theta_kills}
        ) / max(len(theta_kills), 1)
        insights.append(LearningInsight(
            insight_id=f"INS-{uuid.uuid4().hex[:8]}",
            category=FailureCategory.THETA_DECAY,
            pattern=(
                f"{len(theta_kills)} trades died from theta decay. "
                f"Average hold: {avg_hold:.1f} days."
            ),
            frequency=len(theta_kills),
            suggested_action=(
                "Tighten max hold period. Add theta-adjusted "
                "stop loss that accelerates as DTE decreases."
            ),
            evidence_trade_ids=[pm.trade_id for pm in theta_kills],
            confidence=0.75,
        ))

    # ── Pattern 4: IV crush post-catalyst ──
    iv_crush = [
        pm for pm in post_mortems
        if pm.consensus_cause == FailureCategory.IV_CRUSH
    ]
    if len(iv_crush) >= 2:
        insights.append(LearningInsight(
            insight_id=f"INS-{uuid.uuid4().hex[:8]}",
            category=FailureCategory.IV_CRUSH,
            pattern=(
                f"{len(iv_crush)} trades hit by IV crush. "
                f"Entering before known catalysts with elevated IV "
                f"is systematically unprofitable."
            ),
            frequency=len(iv_crush),
            suggested_action=(
                "Penalize trades where IV rank > 60 AND catalyst < 3 days away. "
                "The expected move needs to exceed IV-adjusted breakeven."
            ),
            evidence_trade_ids=[pm.trade_id for pm in iv_crush],
            confidence=0.8,
        ))

    # ── Pattern 5: High-score wins pattern (learn from winners too) ──
    high_score_wins = [
        t for t in closed
        if t.nexus_score >= 60 and t.result == TradeResult.WIN
    ]
    high_score_total = [t for t in closed if t.nexus_score >= 60]
    if high_score_total:
        hs_wr = len(high_score_wins) / len(high_score_total)
        if hs_wr > 0.4:
            avg_win = (
                sum(t.pnl_pct for t in high_score_wins) / len(high_score_wins)
                if high_score_wins else 0
            )
            insights.append(LearningInsight(
                insight_id=f"INS-{uuid.uuid4().hex[:8]}",
                category=FailureCategory.OVERSCORED,  # Inverse
                pattern=(
                    f"High-score trades (60+) have {hs_wr:.0%} win rate "
                    f"with avg {avg_win:+.1f}% on winners. "
                    f"The model's high-conviction signals have real edge."
                ),
                frequency=len(high_score_total),
                avg_loss_when_present=round(avg_win, 2),
                suggested_action="Increase position sizing for score 60+ trades.",
                evidence_trade_ids=[t.trade_id for t in high_score_wins[:10]],
                confidence=0.65,
            ))

    # ── Pattern 6: Ticker-specific patterns ──
    from collections import defaultdict
    ticker_losses: dict[str, list[TradePostMortem]] = defaultdict(list)
    for pm in post_mortems:
        ticker_losses[pm.ticker].append(pm)

    for ticker, pms in ticker_losses.items():
        ticker_trades = [t for t in closed if t.ticker == ticker]
        if len(ticker_trades) >= 3:
            ticker_wr = sum(
                1 for t in ticker_trades if t.result == TradeResult.WIN
            ) / len(ticker_trades)
            if ticker_wr < 0.15:
                insights.append(LearningInsight(
                    insight_id=f"INS-{uuid.uuid4().hex[:8]}",
                    category=FailureCategory.MARKET_REGIME,
                    pattern=(
                        f"{ticker} has {ticker_wr:.0%} win rate across "
                        f"{len(ticker_trades)} trades. "
                        f"Model consistently fails on this ticker."
                    ),
                    frequency=len(pms),
                    avg_loss_when_present=round(
                        sum(t.pnl_pct for t in ticker_trades) / len(ticker_trades), 2
                    ),
                    suggested_action=(
                        f"Add ticker-level penalty for {ticker} or "
                        f"exclude from lotto plays until regime changes."
                    ),
                    evidence_trade_ids=[pm.trade_id for pm in pms],
                    confidence=0.6,
                ))

    return insights


# ──────────────────────────────────────────────────────────────
# Weight & rule generation from insights
# ──────────────────────────────────────────────────────────────

def compute_weight_adjustments(
    insights: list[LearningInsight],
    current_weights: AdaptiveWeights,
    report: PerformanceReport,
) -> list[WeightAdjustment]:
    """Compute concrete weight changes from insights."""
    adjustments: list[WeightAdjustment] = []
    closed = [t for t in report.trades if t.result != TradeResult.OPEN]
    if not closed:
        return adjustments

    # Analyze which signal dimension has best predictive power
    winners = [t for t in closed if t.result == TradeResult.WIN]
    losers = [t for t in closed if t.result != TradeResult.WIN]

    avg_score_w = sum(t.nexus_score for t in winners) / len(winners) if winners else 50
    avg_score_l = sum(t.nexus_score for t in losers) / len(losers) if losers else 50

    # If score separation is poor, the weights need adjustment
    score_gap = avg_score_w - avg_score_l
    if score_gap < 3:
        # Scoring model isn't separating winners from losers well
        # Check which failure categories dominate
        failure_cats = [
            i.category for i in insights if i.confidence > 0.5
        ]

        if (
            FailureCategory.WRONG_DIRECTION in failure_cats
            and current_weights.uoa_weight > 0.25
        ):
                adjustments.append(WeightAdjustment(
                    parameter="uoa_weight",
                    current_value=current_weights.uoa_weight,
                    suggested_value=round(current_weights.uoa_weight - 0.05, 2),
                    reason="UOA signals have poor directional accuracy",
                    expected_impact="Reduce false conviction from noisy flow",
                    confidence=0.6,
                ))

        if FailureCategory.IV_CRUSH in failure_cats:
            # IV component needs more weight to prevent expensive entries
            adjustments.append(WeightAdjustment(
                parameter="iv_weight",
                current_value=current_weights.iv_weight,
                suggested_value=round(current_weights.iv_weight + 0.05, 2),
                reason="IV crush is a primary loss driver",
                expected_impact="Better filter expensive premium entries",
                confidence=0.65,
            ))

        if FailureCategory.THETA_DECAY in failure_cats:
            # Need cheaper entry premiums
            adjustments.append(WeightAdjustment(
                parameter="iv_rank_sweet_spot_high",
                current_value=current_weights.iv_rank_sweet_spot_high,
                suggested_value=round(
                    current_weights.iv_rank_sweet_spot_high - 5, 1
                ),
                reason="Theta decay losses suggest entering with too-expensive premium",
                expected_impact="Force cheaper entry points",
                confidence=0.55,
            ))

    # Min entry score adjustment
    low_score_trades = [t for t in closed if t.nexus_score < 40]
    if low_score_trades:
        low_wr = sum(
            1 for t in low_score_trades if t.result == TradeResult.WIN
        ) / len(low_score_trades)
        if low_wr < 0.2 and current_weights.min_entry_score < 40:
            adjustments.append(WeightAdjustment(
                parameter="min_entry_score",
                current_value=current_weights.min_entry_score,
                suggested_value=40.0,
                reason=(
                    f"Trades below score 40 have {low_wr:.0%} win rate — "
                    f"no edge, pure noise"
                ),
                expected_impact="Eliminate lowest-conviction trades",
                confidence=0.85,
            ))

    return adjustments


def generate_scoring_rules(
    insights: list[LearningInsight],
    post_mortems: list[TradePostMortem],
) -> list[ScoringRule]:
    """Generate new scoring rules from learned patterns."""
    rules: list[ScoringRule] = []

    # From committee suggestions
    filter_suggestions = [
        pm.suggested_filter
        for pm in post_mortems
        if pm.should_have_been_filtered and pm.suggested_filter
    ]

    seen_filters: set[str] = set()
    for suggestion in filter_suggestions:
        normalized = suggestion.strip().lower()
        if normalized not in seen_filters:
            seen_filters.add(normalized)
            rules.append(ScoringRule(
                rule_id=f"RULE-{uuid.uuid4().hex[:8]}",
                name=f"Learned filter: {suggestion[:60]}",
                condition=suggestion,
                score_adjustment=-2.0,
                applies_to="composite",
                reason="Committee identified this as a filterable loss pattern",
            ))

    # From pattern insights
    for insight in insights:
        if insight.confidence >= 0.7 and insight.suggested_action:
            rules.append(ScoringRule(
                rule_id=f"RULE-{uuid.uuid4().hex[:8]}",
                name=f"Pattern: {insight.category.value}",
                condition=insight.pattern[:100],
                score_adjustment=-1.5 if insight.avg_loss_when_present < 0 else 0.5,
                applies_to="composite",
                reason=insight.suggested_action,
                win_rate_improvement=0.0,
            ))

    return rules


# ──────────────────────────────────────────────────────────────
# Master refinement generator (uses Claude for synthesis)
# ──────────────────────────────────────────────────────────────

REFINEMENT_PROMPT = """You are the chief model architect for an options trading scanner.

You have deep expertise in:
- Quantitative model validation and refinement
- Options pricing, greeks, and volatility dynamics
- Signal processing and noise filtering
- Overfitting prevention in trading systems
- Walk-forward optimization

You are reviewing the post-mortem analyses from a specialist committee,
the pattern insights extracted, and the overall performance data.

Your job is to synthesize this into a CONCRETE model refinement plan:
1. Which weight adjustments should be made (be conservative — max 5% change per cycle)
2. Which new rules should be added to the scoring pipeline
3. Which filters should be added to prevent systematic loss patterns
4. What is the expected impact on win rate and profit factor
5. What should NOT be changed (avoid overfitting to recent data)

CRITICAL: Be conservative. Small adjustments. The goal is incremental improvement,
not a complete overhaul. Trading models that change too much too fast blow up."""


async def generate_refinement(
    post_mortems: list[TradePostMortem],
    report: PerformanceReport,
    current_weights: AdaptiveWeights,
    settings: Settings | None = None,
) -> ModelRefinement:
    """Generate a complete model refinement from learning data."""
    settings = settings or get_settings()
    closed = [t for t in report.trades if t.result != TradeResult.OPEN]
    losses = [t for t in closed if t.result != TradeResult.WIN]
    wins = [t for t in closed if t.result == TradeResult.WIN]

    # Extract patterns
    failure_dist = extract_failure_distribution(post_mortems)
    insights = extract_insights(post_mortems, report.trades)
    weight_adj = compute_weight_adjustments(insights, current_weights, report)
    new_rules = generate_scoring_rules(insights, post_mortems)

    cycle_id = f"CYCLE-{uuid.uuid4().hex[:8]}"

    # Try Claude synthesis for the overall rationale
    rationale = _rule_based_rationale(
        failure_dist, insights, report, current_weights
    )

    if settings.anthropic_api_key:
        try:
            context = (
                f"PERFORMANCE SUMMARY:\n"
                f"  Period: {report.start_date} to {report.end_date}\n"
                f"  Trades: {report.total_trades} "
                f"({len(wins)}W / {len(losses)}L)\n"
                f"  Win rate: {report.win_rate:.1%}\n"
                f"  Profit factor: {report.profit_factor:.2f}\n"
                f"  Max drawdown: {report.max_drawdown_pct:.1f}%\n\n"
                f"FAILURE DISTRIBUTION:\n"
                + "\n".join(
                    f"  {cat}: {count} trades"
                    for cat, count in failure_dist.items()
                )
                + "\n\nLEARNED INSIGHTS:\n"
                + "\n".join(
                    f"  [{i.confidence:.0%}] {i.pattern}\n"
                    f"    Action: {i.suggested_action}"
                    for i in insights
                )
                + f"\n\nCURRENT WEIGHTS:\n"
                f"  UOA: {current_weights.uoa_weight}\n"
                f"  IV: {current_weights.iv_weight}\n"
                f"  Catalyst: {current_weights.catalyst_weight}\n"
                f"  Min entry score: {current_weights.min_entry_score}\n"
            )

            from flowedge.scanner.learning.schemas import ModelRefinement as LearningRefinement
            result = await call_agent_llm(
                system_prompt=REFINEMENT_PROMPT,
                user_content=(
                    f"{context}\n\n"
                    f"Generate a conservative refinement plan. "
                    f"Focus on the top 3 failure modes. "
                    f"Max 5% weight change per dimension per cycle."
                ),
                output_type=LearningRefinement,
            )
            rationale = result.rationale or rationale
        except Exception as e:
            logger.warning("refinement_synthesis_failed", error=str(e))

    refinement = ModelRefinement(
        cycle_id=cycle_id,
        generated_at=datetime.now(),
        trades_analyzed=len(closed),
        losses_analyzed=len(post_mortems),
        wins_analyzed=len(wins),
        post_mortems=post_mortems,
        failure_distribution=failure_dist,
        insights=insights,
        weight_adjustments=weight_adj,
        new_rules=new_rules,
        filters_to_add=[
            pm.suggested_filter
            for pm in post_mortems
            if pm.should_have_been_filtered and pm.suggested_filter
        ],
        rationale=rationale,
    )

    logger.info(
        "refinement_generated",
        cycle=cycle_id,
        insights=len(insights),
        weight_changes=len(weight_adj),
        new_rules=len(new_rules),
    )
    return refinement


def _rule_based_rationale(
    failure_dist: dict[str, int],
    insights: list[LearningInsight],
    report: PerformanceReport,
    weights: AdaptiveWeights,
) -> str:
    """Generate a rule-based refinement rationale."""
    parts: list[str] = []

    if report.win_rate < 0.3:
        parts.append(
            f"Win rate ({report.win_rate:.0%}) is below 30% — "
            f"model is generating too many false positives."
        )

    if failure_dist:
        top_cause = max(failure_dist, key=lambda k: failure_dist[k])
        parts.append(
            f"Primary failure mode: {top_cause} "
            f"({failure_dist[top_cause]} trades)."
        )

    high_conf = [i for i in insights if i.confidence >= 0.7]
    if high_conf:
        parts.append(
            f"{len(high_conf)} high-confidence insights identified. "
            f"Top: {high_conf[0].pattern[:80]}"
        )

    if report.profit_factor < 1.0:
        parts.append(
            f"Profit factor {report.profit_factor:.2f} < 1.0 — "
            f"losses exceed gains. Need tighter entry criteria."
        )

    return " ".join(parts) if parts else "Insufficient data for analysis."
