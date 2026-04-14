"""Tests for graph state and pipeline structure."""

from flowedge.graph.state import GraphState


def test_graph_state_defaults() -> None:
    state = GraphState()
    assert state.run_id == ""
    assert state.repo_urls == []
    assert state.evidence_packs == []
    assert state.specialist_outputs == []
    assert state.scorecards == []
    assert state.debate_record is None
    assert state.report is None
    assert state.errors == []
    assert state.status == "pending"


def test_graph_state_with_urls() -> None:
    state = GraphState(repo_urls=["https://github.com/test/repo"])
    assert len(state.repo_urls) == 1


def test_graph_state_json_roundtrip() -> None:
    state = GraphState(
        run_id="test-123",
        repo_urls=["https://github.com/a/b"],
        status="validated",
    )
    json_str = state.model_dump_json()
    rebuilt = GraphState.model_validate_json(json_str)
    assert rebuilt.run_id == "test-123"
    assert rebuilt.status == "validated"


def test_pipeline_graph_compiles() -> None:
    """The LangGraph pipeline must compile without errors."""
    from flowedge.graph.pipeline import compile_analysis_graph

    compiled = compile_analysis_graph()
    assert compiled is not None
