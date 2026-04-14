"""Main LangGraph pipeline — linear analysis flow."""

from __future__ import annotations

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from flowedge.graph.nodes import (
    fetch_and_extract,
    run_debate_round,
    run_specialists,
    score_repos,
    synthesize_report,
    validate_request,
)
from flowedge.graph.state import GraphState


def build_analysis_graph() -> StateGraph[GraphState]:
    """Build the main analysis pipeline graph.

    Flow: validate -> extract -> specialists -> debate -> score -> synthesize
    """
    graph: StateGraph[GraphState] = StateGraph(GraphState)

    graph.add_node("validate_request", validate_request)
    graph.add_node("fetch_and_extract", fetch_and_extract)
    graph.add_node("run_specialists", run_specialists)
    graph.add_node("run_debate", run_debate_round)
    graph.add_node("score_repos", score_repos)
    graph.add_node("synthesize_report", synthesize_report)

    graph.set_entry_point("validate_request")
    graph.add_edge("validate_request", "fetch_and_extract")
    graph.add_edge("fetch_and_extract", "run_specialists")
    graph.add_edge("run_specialists", "run_debate")
    graph.add_edge("run_debate", "score_repos")
    graph.add_edge("score_repos", "synthesize_report")

    return graph


def compile_analysis_graph() -> CompiledStateGraph:  # type: ignore[type-arg]
    """Compile the analysis graph for execution."""
    graph = build_analysis_graph()
    return graph.compile()
