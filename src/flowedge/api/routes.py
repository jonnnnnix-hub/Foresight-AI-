"""API route definitions."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class AnalyzeRequest(BaseModel):
    """Request to analyze repos."""

    repo_urls: list[str] = Field(min_length=1, description="GitHub repo URLs")


class AnalyzeResponse(BaseModel):
    """Response with run ID."""

    run_id: str
    status: str = "accepted"
    repo_count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Submit repos for analysis."""
    import uuid

    run_id = str(uuid.uuid4())[:8]
    # TODO: kick off LangGraph run
    return AnalyzeResponse(
        run_id=run_id,
        status="accepted",
        repo_count=len(request.repo_urls),
    )
