"""API route definitions."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from flowedge.graph.state import GraphState
from flowedge.schemas.report import SynthesisReport

router = APIRouter()

# In-memory store for MVP — replace with DB persistence later
_runs: dict[str, GraphState] = {}


class AnalyzeRequest(BaseModel):
    """Request to analyze repos."""

    repo_urls: list[str] = Field(min_length=1, description="GitHub repo URLs")


class AnalyzeResponse(BaseModel):
    """Response with run ID."""

    run_id: str
    status: str = "accepted"
    repo_count: int


class RunStatusResponse(BaseModel):
    """Status of an analysis run."""

    run_id: str
    status: str
    errors: list[str] = Field(default_factory=list)
    report: SynthesisReport | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


async def _run_analysis(run_id: str, repo_urls: list[str]) -> None:
    """Run the analysis pipeline in the background."""
    from flowedge.graph.pipeline import compile_analysis_graph

    state = GraphState(run_id=run_id, repo_urls=repo_urls)
    _runs[run_id] = state

    graph = compile_analysis_graph()
    result = await graph.ainvoke(state)
    if isinstance(result, dict):
        _runs[run_id] = GraphState.model_validate(result)
    else:
        _runs[run_id] = GraphState.model_validate(result)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest, background_tasks: BackgroundTasks
) -> AnalyzeResponse:
    """Submit repos for analysis."""
    run_id = str(uuid.uuid4())[:12]
    _runs[run_id] = GraphState(run_id=run_id, repo_urls=request.repo_urls, status="queued")
    background_tasks.add_task(_run_analysis, run_id, request.repo_urls)
    return AnalyzeResponse(
        run_id=run_id,
        status="accepted",
        repo_count=len(request.repo_urls),
    )


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str) -> RunStatusResponse:
    """Get the status and results of an analysis run."""
    state = _runs.get(run_id)
    if state is None:
        return RunStatusResponse(run_id=run_id, status="not_found")
    return RunStatusResponse(
        run_id=run_id,
        status=state.status,
        errors=state.errors,
        report=state.report,
    )
