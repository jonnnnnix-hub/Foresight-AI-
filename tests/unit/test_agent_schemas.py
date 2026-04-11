"""Tests for specialist agent output schemas."""

from flowedge.schemas.agents import (
    CartographerOutput,
    ExecutionAnalystOutput,
    JudgeOutput,
    MLAnalystOutput,
    ProductArchitectOutput,
    ResearchAnalystOutput,
    RiskAnalystOutput,
    SkepticOutput,
)


def test_cartographer_output_defaults() -> None:
    out = CartographerOutput(repo_summary="Test repo")
    assert out.repo_summary == "Test repo"
    assert out.key_directories == []
    assert out.entry_points == []


def test_research_analyst_output_score_bounds() -> None:
    out = ResearchAnalystOutput(score=8.5, score_rationale="Strong backtesting")
    assert 0 <= out.score <= 10
    assert out.backtesting_type == ""
    assert not out.parameter_sweep_support


def test_execution_analyst_output() -> None:
    out = ExecutionAnalystOutput(
        score=6.0,
        score_rationale="Partial execution",
        slippage_modeling=True,
        broker_integrations=["IBKR"],
    )
    assert out.slippage_modeling is True
    assert "IBKR" in out.broker_integrations


def test_ml_analyst_output() -> None:
    out = MLAnalystOutput(
        score=3.0,
        score_rationale="Notebook only",
        feature_pipeline_exists=False,
        leakage_concerns=["No walk-forward split"],
    )
    assert not out.feature_pipeline_exists
    assert len(out.leakage_concerns) == 1


def test_product_architect_output() -> None:
    out = ProductArchitectOutput(score=7.0, score_rationale="Modular design")
    assert out.modularity_assessment == ""
    assert out.reusable_components == []


def test_risk_analyst_output() -> None:
    out = RiskAnalystOutput(
        score=5.0,
        score_rationale="Mixed reliability",
        operational_risks=["Single maintainer"],
    )
    assert len(out.operational_risks) == 1


def test_skeptic_output() -> None:
    out = SkepticOutput(
        overclaim_flags=["Claims real-time but no WebSocket code"],
        hidden_assumptions=["Assumes zero-fee trading"],
    )
    assert len(out.overclaim_flags) == 1
    assert len(out.hidden_assumptions) == 1


def test_judge_output() -> None:
    out = JudgeOutput(
        repo_ranking=["repoA", "repoB"],
        top_repo_by_category={"research": "repoA", "execution": "repoB"},
        borrow_avoid_replace={
            "repoA": {"borrow": ["backtester"], "avoid": ["UI"], "replace": []},
        },
        merged_architecture={"data_ingestion": "repoA", "execution": "repoB"},
    )
    assert out.repo_ranking[0] == "repoA"
    assert out.top_repo_by_category["execution"] == "repoB"
    assert "backtester" in out.borrow_avoid_replace["repoA"]["borrow"]


def test_all_schemas_roundtrip_json() -> None:
    """All agent schemas must survive JSON serialization roundtrip."""
    schemas = [
        CartographerOutput(repo_summary="test"),
        ResearchAnalystOutput(score=5.0, score_rationale="test"),
        ExecutionAnalystOutput(score=5.0, score_rationale="test"),
        MLAnalystOutput(score=5.0, score_rationale="test"),
        ProductArchitectOutput(score=5.0, score_rationale="test"),
        RiskAnalystOutput(score=5.0, score_rationale="test"),
        SkepticOutput(),
        JudgeOutput(),
    ]
    for schema in schemas:
        json_str = schema.model_dump_json()
        rebuilt = type(schema).model_validate_json(json_str)
        assert rebuilt == schema, f"Roundtrip failed for {type(schema).__name__}"
