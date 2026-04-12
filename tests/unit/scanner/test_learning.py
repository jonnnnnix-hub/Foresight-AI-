"""Tests for the learning and adaptive scoring system."""

from __future__ import annotations

from datetime import date

from flowedge.scanner.learning.adaptive import (
    apply_refinement,
    compute_adaptive_score,
)
from flowedge.scanner.learning.patterns import (
    extract_failure_distribution,
    extract_insights,
)
from flowedge.scanner.learning.schemas import (
    AdaptiveWeights,
    FailureCategory,
    LearningInsight,
    ModelRefinement,
    ScoringRule,
    SpecialistVerdict,
    TradePostMortem,
    WeightAdjustment,
)
from flowedge.scanner.performance.schemas import (
    SimulatedTrade,
    TradeResult,
)

# ── Schema tests ──

def test_adaptive_weights_defaults():
    w = AdaptiveWeights()
    assert w.uoa_weight == 0.35
    assert w.iv_weight == 0.30
    assert w.catalyst_weight == 0.35
    assert abs(w.uoa_weight + w.iv_weight + w.catalyst_weight - 1.0) < 0.01


def test_failure_category_values():
    assert FailureCategory.FLOW_MISLEADING == "flow_misleading"
    assert FailureCategory.IV_CRUSH == "iv_crush"
    assert len(FailureCategory) == 10


def test_trade_post_mortem_serialization():
    pm = TradePostMortem(
        trade_id="TEST-001",
        ticker="TSLA",
        entry_date="2026-01-15",
        pnl_pct=-45.0,
        consensus_cause=FailureCategory.WRONG_DIRECTION,
    )
    data = pm.model_dump(mode="json")
    assert data["consensus_cause"] == "wrong_direction"
    assert data["pnl_pct"] == -45.0


def test_scoring_rule_schema():
    rule = ScoringRule(
        rule_id="R-001",
        name="Low score filter",
        condition="score < 40",
        score_adjustment=-2.0,
    )
    assert rule.score_adjustment == -2.0


# ── Adaptive scoring tests ──

def test_compute_adaptive_score_basic():
    w = AdaptiveWeights()
    composite, score_100, notes = compute_adaptive_score(
        base_score=5.0, weights=w,
        uoa_score=6.0, iv_score=4.0, catalyst_score=5.0,
    )
    assert 0 <= composite <= 10
    assert 0 <= score_100 <= 100
    assert isinstance(notes, list)


def test_compute_adaptive_score_with_penalty():
    w = AdaptiveWeights()
    w.penalty_rules = [
        ScoringRule(
            rule_id="R-001",
            name="Low score penalty",
            condition="score < 40",
            score_adjustment=-2.0,
        )
    ]
    _, score_hi, _ = compute_adaptive_score(
        base_score=6.0, weights=w,
        uoa_score=6.0, iv_score=6.0, catalyst_score=6.0,
        nexus_score_100=60,
    )
    _, score_lo, notes_lo = compute_adaptive_score(
        base_score=3.0, weights=w,
        uoa_score=3.0, iv_score=3.0, catalyst_score=3.0,
        nexus_score_100=30,
    )
    # Low score should get penalized
    assert any("Penalty" in n for n in notes_lo)


def test_compute_adaptive_score_with_bonus():
    w = AdaptiveWeights()
    w.bonus_rules = [
        ScoringRule(
            rule_id="R-002",
            name="High conviction bonus",
            condition="score 60+",
            score_adjustment=0.5,
        )
    ]
    _, _, notes = compute_adaptive_score(
        base_score=7.0, weights=w,
        uoa_score=7.0, iv_score=7.0, catalyst_score=7.0,
        nexus_score_100=70,
    )
    assert any("Bonus" in n for n in notes)


def test_compute_adaptive_score_clamped():
    w = AdaptiveWeights()
    # Even with extreme inputs, should stay in bounds
    composite, score_100, _ = compute_adaptive_score(
        base_score=15.0, weights=w,
        uoa_score=10.0, iv_score=10.0, catalyst_score=10.0,
    )
    assert composite <= 10.0
    assert score_100 <= 100


# ── Refinement application tests ──

def test_apply_refinement_weight_change():
    current = AdaptiveWeights()
    refinement = ModelRefinement(
        cycle_id="TEST-CYCLE",
        weight_adjustments=[
            WeightAdjustment(
                parameter="uoa_weight",
                current_value=0.35,
                suggested_value=0.30,
                reason="UOA not predictive enough",
            )
        ],
    )
    updated = apply_refinement(current, refinement)
    # Weight should decrease (clamped to max 5% change)
    assert updated.uoa_weight < current.uoa_weight
    # Weights should sum to ~1.0
    total = updated.uoa_weight + updated.iv_weight + updated.catalyst_weight
    assert abs(total - 1.0) < 0.01
    assert updated.version == current.version + 1


def test_apply_refinement_max_change_clamped():
    current = AdaptiveWeights()
    refinement = ModelRefinement(
        cycle_id="TEST",
        weight_adjustments=[
            WeightAdjustment(
                parameter="uoa_weight",
                current_value=0.35,
                suggested_value=0.10,  # 25% drop — should be clamped
                reason="test",
            )
        ],
    )
    updated = apply_refinement(current, refinement, max_weight_change=0.05)
    # Should only drop by max 5%
    assert updated.uoa_weight >= current.uoa_weight - 0.05 - 0.01


def test_apply_refinement_min_score_only_increases():
    current = AdaptiveWeights(min_entry_score=40.0)
    refinement = ModelRefinement(
        cycle_id="TEST",
        weight_adjustments=[
            WeightAdjustment(
                parameter="min_entry_score",
                current_value=40.0,
                suggested_value=30.0,  # Trying to lower — should be blocked
                reason="test",
            )
        ],
    )
    updated = apply_refinement(current, refinement)
    assert updated.min_entry_score >= 40.0  # Never lowered


def test_apply_refinement_adds_rules():
    current = AdaptiveWeights()
    assert len(current.penalty_rules) == 0

    refinement = ModelRefinement(
        cycle_id="TEST",
        new_rules=[
            ScoringRule(
                rule_id="R-001",
                name="Test penalty",
                condition="score < 40",
                score_adjustment=-2.0,
            )
        ],
    )
    updated = apply_refinement(current, refinement)
    assert len(updated.penalty_rules) == 1


def test_apply_refinement_deduplicates_rules():
    current = AdaptiveWeights()
    current.penalty_rules = [
        ScoringRule(
            rule_id="R-001",
            name="Existing",
            condition="score < 40",
            score_adjustment=-2.0,
        )
    ]
    refinement = ModelRefinement(
        cycle_id="TEST",
        new_rules=[
            ScoringRule(
                rule_id="R-002",
                name="Duplicate",
                condition="score < 40",  # Same condition
                score_adjustment=-1.5,
            )
        ],
    )
    updated = apply_refinement(current, refinement)
    assert len(updated.penalty_rules) == 1  # Not duplicated


# ── Pattern extraction tests ──

def test_extract_failure_distribution():
    pms = [
        TradePostMortem(
            trade_id="T1", ticker="TSLA", entry_date="2026-01-01",
            consensus_cause=FailureCategory.WRONG_DIRECTION,
        ),
        TradePostMortem(
            trade_id="T2", ticker="NVDA", entry_date="2026-01-02",
            consensus_cause=FailureCategory.WRONG_DIRECTION,
        ),
        TradePostMortem(
            trade_id="T3", ticker="AAPL", entry_date="2026-01-03",
            consensus_cause=FailureCategory.IV_CRUSH,
        ),
    ]
    dist = extract_failure_distribution(pms)
    assert dist["wrong_direction"] == 2
    assert dist["iv_crush"] == 1


def test_extract_insights_low_score():
    pms = [
        TradePostMortem(
            trade_id=f"T{i}", ticker="TSLA", entry_date="2026-01-01",
            nexus_score=30, pnl_pct=-50.0,
            consensus_cause=FailureCategory.OVERSCORED,
        )
        for i in range(5)
    ]
    trades = [
        SimulatedTrade(
            trade_id=f"T{i}", ticker="TSLA",
            entry_date=date(2026, 1, 1),
            nexus_score=30, pnl_pct=-50.0, pnl_dollars=-50.0,
            result=TradeResult.LOSS,
        )
        for i in range(5)
    ]
    insights = extract_insights(pms, trades)
    # Should detect low-score pattern
    low_score_insights = [
        i for i in insights
        if "score < 40" in i.pattern.lower() or "score < 40" in i.suggested_action.lower()
    ]
    assert len(low_score_insights) >= 1


def test_extract_insights_empty():
    insights = extract_insights([], [])
    assert insights == []


# ── Specialist verdict tests ──

def test_specialist_verdict_schema():
    v = SpecialistVerdict(
        specialist="flow_analyst",
        diagnosis="UOA was retail noise",
        root_cause=FailureCategory.FLOW_MISLEADING,
        confidence=0.8,
        evidence=["Vol/OI ratio only 1.2x", "No block prints"],
        recommendation="Require vol/OI > 3x for UOA signal",
    )
    data = v.model_dump(mode="json")
    assert data["specialist"] == "flow_analyst"
    assert data["confidence"] == 0.8
    assert len(data["evidence"]) == 2


# ── Learning insight tests ──

def test_learning_insight_schema():
    insight = LearningInsight(
        insight_id="INS-001",
        category=FailureCategory.THETA_DECAY,
        pattern="3 trades died from theta decay",
        frequency=3,
        avg_loss_when_present=-65.0,
        suggested_action="Tighten max hold period",
        confidence=0.75,
    )
    assert insight.confidence == 0.75
    assert insight.frequency == 3


# ── Model refinement tests ──

def test_model_refinement_full():
    r = ModelRefinement(
        cycle_id="CYCLE-001",
        trades_analyzed=50,
        losses_analyzed=30,
        wins_analyzed=20,
        failure_distribution={"wrong_direction": 10, "theta_decay": 8},
        insights=[
            LearningInsight(
                category=FailureCategory.WRONG_DIRECTION,
                pattern="test",
                confidence=0.7,
            )
        ],
        weight_adjustments=[
            WeightAdjustment(
                parameter="uoa_weight",
                current_value=0.35,
                suggested_value=0.30,
                reason="test",
            )
        ],
        rationale="Test refinement",
    )
    data = r.model_dump(mode="json")
    assert data["losses_analyzed"] == 30
    assert len(data["insights"]) == 1
