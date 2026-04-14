"""Individual graph node functions for the analysis pipeline."""

from __future__ import annotations

import uuid

import structlog

from flowedge.agents import ALL_ANALYSTS, AgentOutput
from flowedge.agents.judge import run_judge
from flowedge.debate.engine import run_debate
from flowedge.extraction.file_scanner import scan_repo
from flowedge.graph.state import GraphState
from flowedge.ingestion.clone import clone_repo, validate_repo_url
from flowedge.scoring.engine import compute_scorecards
from flowedge.synthesis.engine import build_synthesis_report

logger = structlog.get_logger()


async def validate_request(state: GraphState) -> GraphState:
    """Validate all repo URLs in the request."""
    validated: list[str] = []
    for url in state.repo_urls:
        try:
            validated.append(validate_repo_url(url))
        except ValueError as e:
            state.errors.append(f"Invalid URL {url}: {e}")
    state.repo_urls = validated
    state.run_id = state.run_id or str(uuid.uuid4())[:12]
    state.status = "validated" if validated else "failed"
    logger.info("request_validated", run_id=state.run_id, repo_count=len(validated))
    return state


async def fetch_and_extract(state: GraphState) -> GraphState:
    """Clone repos and extract evidence packs."""
    for url in state.repo_urls:
        try:
            repo_path = clone_repo(url)
            pack = scan_repo(repo_path)
            pack.repo_url = url
            state.evidence_packs.append(pack)
            logger.info("evidence_extracted", repo=pack.repo_name, items=len(pack.items))
        except Exception as e:
            state.errors.append(f"Failed to process {url}: {e}")
            logger.error("extraction_failed", url=url, error=str(e))
    state.status = "extracted" if state.evidence_packs else "failed"
    return state


async def run_specialists(state: GraphState) -> GraphState:
    """Run all specialist agents on each evidence pack."""
    all_outputs: list[dict[str, object]] = []
    for pack in state.evidence_packs:
        repo_outputs: dict[str, object] = {"repo_name": pack.repo_name}
        agent_results: list[AgentOutput] = []
        for agent_cls in ALL_ANALYSTS:
            agent = agent_cls()
            try:
                output = await agent.analyze(pack)
                agent_results.append(output)
                logger.info(
                    "agent_complete",
                    agent=agent.name,
                    repo=pack.repo_name,
                    score_count=len(output.scores),
                )
            except Exception as e:
                state.errors.append(f"{agent.name} failed on {pack.repo_name}: {e}")
                logger.error("agent_failed", agent=agent.name, repo=pack.repo_name, error=str(e))
        repo_outputs["outputs"] = [o.model_dump() for o in agent_results]
        all_outputs.append(repo_outputs)
    state.specialist_outputs = all_outputs
    state.status = "analyzed"
    return state


async def run_debate_round(state: GraphState) -> GraphState:
    """Run adversarial debate across specialist findings."""
    # Reconstruct AgentOutput objects from serialized dicts
    repo_agent_outputs: dict[str, list[AgentOutput]] = {}
    for repo_data in state.specialist_outputs:
        repo_name = str(repo_data.get("repo_name", "unknown"))
        raw_outputs = repo_data.get("outputs", [])
        if isinstance(raw_outputs, list):
            repo_agent_outputs[repo_name] = [
                AgentOutput.model_validate(o) for o in raw_outputs
            ]

    debate_record = await run_debate(repo_agent_outputs)
    state.debate_record = debate_record
    state.status = "debated"
    logger.info("debate_complete", rounds=len(debate_record.rounds))
    return state


async def score_repos(state: GraphState) -> GraphState:
    """Compute weighted scorecards for each repo."""
    repo_agent_outputs: dict[str, list[AgentOutput]] = {}
    for repo_data in state.specialist_outputs:
        repo_name = str(repo_data.get("repo_name", "unknown"))
        raw_outputs = repo_data.get("outputs", [])
        if isinstance(raw_outputs, list):
            repo_agent_outputs[repo_name] = [
                AgentOutput.model_validate(o) for o in raw_outputs
            ]

    scorecards = compute_scorecards(repo_agent_outputs)
    state.scorecards = scorecards
    state.status = "scored"
    logger.info("scoring_complete", repos=len(scorecards))
    return state


async def synthesize_report(state: GraphState) -> GraphState:
    """Generate final synthesis report."""
    repo_agent_outputs: dict[str, list[AgentOutput]] = {}
    for repo_data in state.specialist_outputs:
        repo_name = str(repo_data.get("repo_name", "unknown"))
        raw_outputs = repo_data.get("outputs", [])
        if isinstance(raw_outputs, list):
            repo_agent_outputs[repo_name] = [
                AgentOutput.model_validate(o) for o in raw_outputs
            ]

    judge_output = await run_judge(repo_agent_outputs)
    report = build_synthesis_report(
        run_id=state.run_id,
        scorecards=state.scorecards,
        debate_record=state.debate_record,
        judge_output=judge_output,
    )
    state.report = report
    state.status = "complete"
    logger.info("synthesis_complete", run_id=state.run_id)
    return state
