"""Evidence schemas extracted from repository analysis."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class EvidenceSource(StrEnum):
    """Where a piece of evidence was found."""

    README = "readme"
    DOCS = "docs"
    SOURCE = "source"
    TESTS = "tests"
    EXAMPLES = "examples"
    DEPS = "deps"
    COMMITS = "commits"
    RELEASES = "releases"
    CI = "ci"


class EvidenceItem(BaseModel):
    """A single piece of evidence extracted from a repo."""

    source: EvidenceSource
    file_path: str = Field(description="Path within the repo")
    content_snippet: str = Field(max_length=2000, description="Relevant excerpt")
    claim: str = Field(description="What this evidence supports or refutes")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")


class EvidencePack(BaseModel):
    """Complete evidence collection for one repository."""

    repo_url: str
    repo_name: str
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    items: list[EvidenceItem] = Field(default_factory=list)
    file_tree_summary: str = Field(default="", description="Top-level directory listing")
    language_breakdown: dict[str, float] = Field(
        default_factory=dict, description="Language percentages"
    )
    total_files: int = 0
    total_lines: int = 0
