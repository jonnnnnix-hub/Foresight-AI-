"""FEEDBACK LOOP — the master orchestrator for continuous model refinement.

Connects: Performance Data → Loss Analysis Committee → Pattern Extraction
→ Weight Adjustment → Adaptive Scorer → Next Cycle

This is the "teacher" that uses real P&L results to make the
scoring models incrementally better over time.

Pipeline:
1. Load latest performance report (real trade outcomes)
2. Identify worst losses and highest-scored failures
3. Run specialist committee on each loss (ANALYST)
4. Extract patterns across all post-mortems (PATTERNS)
5. Generate weight adjustments and new rules (PATTERNS)
6. Apply refinements to adaptive weights (ADAPTIVE)
7. Save updated weights for next scoring cycle
8. Log everything for auditability

Safety:
- Max 5% weight change per cycle
- Weights are bounded and must sum to 1.0
- Min entry score can only increase
- At least 10 trades required before learning
- Overfitting guards: insights need >= 3 trade evidence
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import structlog

from flowedge.config.settings import Settings, get_settings
from flowedge.scanner.learning.adaptive import (
    apply_refinement,
    load_weights,
    save_refinement,
    save_weights,
)
from flowedge.scanner.learning.analyst import analyze_losses_batch
from flowedge.scanner.learning.patterns import generate_refinement
from flowedge.scanner.learning.schemas import ModelRefinement
from flowedge.scanner.performance.schemas import (
    PerformanceReport,
    TradeResult,
)
from flowedge.scanner.performance.simulator import load_report

logger = structlog.get_logger()

FEEDBACK_LOG = Path("./data/learning/feedback_log.json")
MIN_TRADES_FOR_LEARNING = 10
MAX_LOSSES_TO_ANALYZE = 15


async def run_learning_cycle(
    report: PerformanceReport | None = None,
    settings: Settings | None = None,
    max_losses: int = MAX_LOSSES_TO_ANALYZE,
    dry_run: bool = False,
) -> ModelRefinement | None:
    """Execute one full learning cycle.

    This is the main entry point — call after each performance update.

    Args:
        report: Performance data to learn from (loads from disk if None).
        settings: App settings.
        max_losses: Max number of losses to analyze per cycle.
        dry_run: If True, generate refinements but don't apply them.

    Returns:
        The ModelRefinement if generated, None if insufficient data.
    """
    settings = settings or get_settings()
    report = report or load_report()

    if report is None:
        logger.warning("no_performance_data_for_learning")
        return None

    closed = [t for t in report.trades if t.result != TradeResult.OPEN]
    losses = [t for t in closed if t.result != TradeResult.WIN]

    if len(closed) < MIN_TRADES_FOR_LEARNING:
        logger.info(
            "insufficient_trades_for_learning",
            trades=len(closed),
            required=MIN_TRADES_FOR_LEARNING,
        )
        return None

    logger.info(
        "learning_cycle_starting",
        total_trades=len(closed),
        losses=len(losses),
        win_rate=report.win_rate,
    )

    # ── Step 1: Load current adaptive weights ──
    current_weights = load_weights()
    logger.info(
        "current_weights",
        version=current_weights.version,
        uoa=current_weights.uoa_weight,
        iv=current_weights.iv_weight,
        catalyst=current_weights.catalyst_weight,
        min_score=current_weights.min_entry_score,
    )

    # ── Step 2: Prioritize which losses to analyze ──
    # Focus on: high-score failures (model was confident but wrong)
    # and large dollar losses
    prioritized = sorted(
        losses,
        key=lambda t: (t.nexus_score * 0.6 + abs(t.pnl_dollars) * 0.4),
        reverse=True,
    )

    # ── Step 3: Run specialist committee on each loss ──
    logger.info(
        "running_specialist_committee",
        losses_to_analyze=min(max_losses, len(prioritized)),
    )
    post_mortems = await analyze_losses_batch(
        prioritized, max_analyze=max_losses, settings=settings,
    )

    # ── Step 4: Generate refinement from patterns ──
    refinement = await generate_refinement(
        post_mortems, report, current_weights, settings,
    )

    logger.info(
        "refinement_generated",
        cycle=refinement.cycle_id,
        insights=len(refinement.insights),
        weight_changes=len(refinement.weight_adjustments),
        new_rules=len(refinement.new_rules),
        filters=len(refinement.filters_to_add),
    )

    if dry_run:
        logger.info("dry_run_mode — refinements NOT applied")
        _log_feedback(refinement, applied=False)
        return refinement

    # ── Step 4.5: Walk-forward validation (overfitting guard) ──
    from flowedge.scanner.learning.walk_forward import (
        detect_regime,
        regime_adjusted_weights,
        run_walk_forward,
    )

    wf_result = run_walk_forward(report.trades)
    if wf_result.is_overfit:
        logger.warning(
            "overfitting_detected",
            wfe=wf_result.walkforward_efficiency,
            train_wr=wf_result.avg_train_wr,
            test_wr=wf_result.avg_test_wr,
        )
        # Scale down weight adjustments by WFE ratio to reduce overfitting
        for adj in refinement.weight_adjustments:
            delta = adj.suggested_value - adj.current_value
            adj.suggested_value = round(
                adj.current_value + delta * max(wf_result.walkforward_efficiency, 0.3),
                4,
            )

    # Regime-adaptive adjustments (inspired by freqAI-LSTM)
    regime = detect_regime(report.trades)
    regime_adj = regime_adjusted_weights(current_weights, regime)
    if regime_adj:
        refinement.weight_adjustments.extend(regime_adj)
        logger.info(
            "regime_adjustments",
            regime=regime.label,
            confidence=regime.confidence,
            adjustments=len(regime_adj),
        )

    # ── Step 5: Apply refinements to weights ──
    updated_weights = apply_refinement(current_weights, refinement)

    # ── Step 6: Save everything ──
    save_weights(updated_weights)
    save_refinement(refinement)
    _log_feedback(refinement, applied=True)

    logger.info(
        "learning_cycle_complete",
        cycle=refinement.cycle_id,
        old_version=current_weights.version,
        new_version=updated_weights.version,
        uoa_weight=f"{current_weights.uoa_weight} → {updated_weights.uoa_weight}",
        iv_weight=f"{current_weights.iv_weight} → {updated_weights.iv_weight}",
        catalyst_weight=(
            f"{current_weights.catalyst_weight} → {updated_weights.catalyst_weight}"
        ),
        min_score=f"{current_weights.min_entry_score} → {updated_weights.min_entry_score}",
        penalty_rules=len(updated_weights.penalty_rules),
        bonus_rules=len(updated_weights.bonus_rules),
    )

    return refinement


def _log_feedback(refinement: ModelRefinement, applied: bool) -> None:
    """Log feedback cycle to disk for auditability."""
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    import contextlib
    log: list[dict[str, object]] = []
    if FEEDBACK_LOG.exists():
        with contextlib.suppress(Exception):
            log = json.loads(FEEDBACK_LOG.read_text())

    entry = {
        "cycle_id": refinement.cycle_id,
        "timestamp": datetime.now().isoformat(),
        "applied": applied,
        "trades_analyzed": refinement.trades_analyzed,
        "losses_analyzed": refinement.losses_analyzed,
        "failure_distribution": refinement.failure_distribution,
        "insights": [
            {
                "pattern": i.pattern[:100],
                "confidence": i.confidence,
                "action": i.suggested_action[:100],
            }
            for i in refinement.insights
        ],
        "weight_changes": [
            {
                "param": wa.parameter,
                "from": wa.current_value,
                "to": wa.suggested_value,
            }
            for wa in refinement.weight_adjustments
        ],
        "new_rules": len(refinement.new_rules),
        "rationale": refinement.rationale[:300],
    }
    log.append(entry)
    FEEDBACK_LOG.write_text(json.dumps(log[-100:], indent=2, default=str))


async def run_full_pipeline(
    settings: Settings | None = None,
) -> dict[str, object]:
    """Run simulation + learning cycle as a single pipeline.

    This is what the nightly cron should call:
    1. Run performance simulation (updates P&L data)
    2. Run learning cycle (analyzes losses, refines model)
    3. Return summary of both

    Returns a summary dict suitable for logging/alerting.
    """
    from flowedge.scanner.performance.simulator import run_historical_simulation

    settings = settings or get_settings()

    # Phase 1: Update performance data
    logger.info("pipeline_phase_1: running simulation")
    report = await run_historical_simulation(settings=settings)

    # Phase 2: Run learning cycle
    logger.info("pipeline_phase_2: running learning cycle")
    refinement = await run_learning_cycle(report=report, settings=settings)

    summary: dict[str, object] = {
        "simulation": {
            "start": str(report.start_date),
            "end": str(report.end_date),
            "trades": report.total_trades,
            "win_rate": report.win_rate,
            "profit_factor": report.profit_factor,
            "total_return_pct": report.total_return_pct,
            "ending_value": report.ending_value,
        },
        "learning": None,
    }

    if refinement:
        summary["learning"] = {
            "cycle_id": refinement.cycle_id,
            "losses_analyzed": refinement.losses_analyzed,
            "insights": len(refinement.insights),
            "weight_changes": len(refinement.weight_adjustments),
            "new_rules": len(refinement.new_rules),
            "rationale": refinement.rationale[:200],
        }

    return summary
