"""ADAPTIVE SCORER — applies learned refinements to live scoring.

Loads the latest AdaptiveWeights from disk and modifies the composite
scoring pipeline in real-time. The weights evolve after each learning
cycle but change conservatively (max 5% per dimension per cycle).

The adaptive scorer wraps the base NEXUS scorer, applying:
1. Adjusted dimension weights (UOA, IV, Catalyst)
2. Penalty rules from loss analysis
3. Bonus rules from win pattern analysis
4. Minimum entry score filter
5. Ticker-specific adjustments
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from flowedge.scanner.learning.schemas import (
    AdaptiveWeights,
    ModelRefinement,
    WeightAdjustment,
)

logger = structlog.get_logger()

WEIGHTS_FILE = Path("./data/learning/adaptive_weights.json")
HISTORY_FILE = Path("./data/learning/refinement_history.json")


def load_weights() -> AdaptiveWeights:
    """Load the current adaptive weights from disk."""
    if WEIGHTS_FILE.exists():
        try:
            data = json.loads(WEIGHTS_FILE.read_text())
            return AdaptiveWeights.model_validate(data)
        except Exception as e:
            logger.warning("adaptive_weights_load_failed", error=str(e))
    return AdaptiveWeights()  # Defaults


def save_weights(weights: AdaptiveWeights) -> None:
    """Save adaptive weights to disk."""
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_FILE.write_text(
        json.dumps(weights.model_dump(mode="json"), indent=2, default=str)
    )
    logger.info("adaptive_weights_saved", version=weights.version)


def save_refinement(refinement: ModelRefinement) -> None:
    """Append a refinement to the history log."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    if HISTORY_FILE.exists():
        import contextlib
        with contextlib.suppress(Exception):
            history = json.loads(HISTORY_FILE.read_text())

    # Store summary, not full post-mortems (too large)
    summary = {
        "cycle_id": refinement.cycle_id,
        "generated_at": str(refinement.generated_at),
        "trades_analyzed": refinement.trades_analyzed,
        "losses_analyzed": refinement.losses_analyzed,
        "failure_distribution": refinement.failure_distribution,
        "insights_count": len(refinement.insights),
        "weight_adjustments": [
            wa.model_dump(mode="json") for wa in refinement.weight_adjustments
        ],
        "new_rules_count": len(refinement.new_rules),
        "rationale": refinement.rationale,
    }
    history.append(summary)
    # Keep last 50 cycles
    HISTORY_FILE.write_text(
        json.dumps(history[-50:], indent=2, default=str)
    )


def apply_refinement(
    current: AdaptiveWeights,
    refinement: ModelRefinement,
    max_weight_change: float = 0.05,
) -> AdaptiveWeights:
    """Apply a ModelRefinement to current weights with safety clamps.

    Rules:
    - Max 5% change per dimension per cycle (prevents overfitting)
    - Weights must sum to ~1.0
    - Min entry score can only increase (never lower the bar)
    - New rules are additive (old rules preserved)
    """
    updated = current.model_copy(deep=True)

    for adj in refinement.weight_adjustments:
        _apply_weight_adjustment(updated, adj, max_weight_change)

    # Add new penalty/bonus rules
    for rule in refinement.new_rules:
        if rule.score_adjustment < 0:
            # Dedup by condition
            existing = {r.condition for r in updated.penalty_rules}
            if rule.condition not in existing:
                updated.penalty_rules.append(rule)
        elif rule.score_adjustment > 0:
            existing = {r.condition for r in updated.bonus_rules}
            if rule.condition not in existing:
                updated.bonus_rules.append(rule)

    # Normalize weights to sum to 1.0
    total = updated.uoa_weight + updated.iv_weight + updated.catalyst_weight
    if total > 0:
        updated.uoa_weight = round(updated.uoa_weight / total, 3)
        updated.iv_weight = round(updated.iv_weight / total, 3)
        updated.catalyst_weight = round(updated.catalyst_weight / total, 3)

    # Update metadata
    updated.version += 1
    updated.cycles_applied += 1
    updated.learning_history.append(refinement.cycle_id)

    return updated


def _apply_weight_adjustment(
    weights: AdaptiveWeights,
    adj: WeightAdjustment,
    max_change: float,
) -> None:
    """Apply a single weight adjustment with safety clamps."""
    param = adj.parameter
    if not hasattr(weights, param):
        logger.warning("unknown_weight_parameter", parameter=param)
        return

    current_val = getattr(weights, param)
    desired = adj.suggested_value
    delta = desired - current_val

    # Clamp change magnitude
    clamped_delta = max(-max_change, min(max_change, delta))
    new_val = round(current_val + clamped_delta, 4)

    # Specific bounds per parameter
    bounds: dict[str, tuple[float, float]] = {
        "uoa_weight": (0.15, 0.50),
        "iv_weight": (0.15, 0.50),
        "catalyst_weight": (0.15, 0.50),
        "min_entry_score": (0.0, 60.0),
        "high_conviction_threshold": (50.0, 90.0),
        "uoa_min_premium": (1000.0, 100_000.0),
        "iv_rank_sweet_spot_low": (5.0, 25.0),
        "iv_rank_sweet_spot_high": (20.0, 50.0),
        "catalyst_max_days": (5, 30),
        "catalyst_min_days": (0, 5),
    }

    if param in bounds:
        lo, hi = bounds[param]
        new_val = max(lo, min(hi, new_val))

    # Min entry score can only go up (never lower the bar)
    if param == "min_entry_score" and new_val < current_val:
        return

    setattr(weights, param, new_val)
    logger.info(
        "weight_adjusted",
        parameter=param,
        old=current_val,
        new=new_val,
        delta=round(clamped_delta, 4),
    )


def compute_adaptive_score(
    base_score: float,
    weights: AdaptiveWeights,
    uoa_score: float,
    iv_score: float,
    catalyst_score: float,
    ticker: str = "",
    nexus_score_100: int = 0,
) -> tuple[float, int, list[str]]:
    """Compute adaptive-adjusted composite score.

    Returns (composite_score, score_100, applied_adjustments).
    """
    adjustments: list[str] = []

    # Reweight with adaptive weights
    composite = (
        uoa_score * weights.uoa_weight
        + iv_score * weights.iv_weight
        + catalyst_score * weights.catalyst_weight
    )

    # Apply penalty rules
    for rule in weights.penalty_rules:
        # Rule conditions are human-readable — we check simple patterns
        adj = rule.score_adjustment
        applied = False

        cond = rule.condition.lower()
        if (
            ("score < 40" in cond and nexus_score_100 < 40)
            or ("iv rank > 60" in cond and iv_score < 3)
            or (ticker and ticker.lower() in cond)
        ):
            composite += adj
            applied = True

        if applied:
            adjustments.append(f"Penalty: {rule.name} ({adj:+.1f})")

    # Apply bonus rules
    for rule in weights.bonus_rules:
        adj = rule.score_adjustment
        applied = False

        if "score 60+" in rule.condition.lower() and nexus_score_100 >= 60:
            composite += adj
            applied = True

        if applied:
            adjustments.append(f"Bonus: {rule.name} ({adj:+.1f})")

    # Clamp
    composite = max(0.0, min(10.0, composite))
    score_100 = min(100, round(composite * 10))

    # Below min entry threshold?
    if score_100 < weights.min_entry_score:
        adjustments.append(
            f"Below min entry ({weights.min_entry_score}) — filtered"
        )

    return round(composite, 2), score_100, adjustments
