"""Repo Cartographer — maps repository structure and architecture."""

from flowedge.agents.base import AgentOutput, BaseAnalyst
from flowedge.agents.llm import call_agent_llm
from flowedge.agents.prompts import CARTOGRAPHER_PROMPT
from flowedge.schemas.agents import CartographerOutput
from flowedge.schemas.evidence import EvidencePack


def _build_evidence_summary(evidence: EvidencePack) -> str:
    """Format evidence pack into LLM-consumable text."""
    parts = [
        f"Repository: {evidence.repo_name}",
        f"URL: {evidence.repo_url}",
        f"Total files: {evidence.total_files}",
        f"Total lines: {evidence.total_lines}",
        f"Languages: {evidence.language_breakdown}",
        f"\nFile tree:\n{evidence.file_tree_summary}",
    ]
    for item in evidence.items[:80]:  # Cap to avoid token overflow
        parts.append(f"\n[{item.source}] {item.file_path}:\n{item.content_snippet[:500]}")
    return "\n".join(parts)


class RepoCartographer(BaseAnalyst):
    """Maps repo structure, modules, entry points, and architecture seams."""

    name = "repo_cartographer"

    async def analyze(self, evidence: EvidencePack) -> AgentOutput:
        """Analyze repo structure and produce a cartographic map."""
        summary = _build_evidence_summary(evidence)
        result = await call_agent_llm(
            system_prompt=CARTOGRAPHER_PROMPT,
            user_content=f"Analyze this repository:\n\n{summary}",
            output_type=CartographerOutput,
        )
        return AgentOutput(
            agent_name=self.name,
            summary=result.repo_summary,
            findings=result.architecture_observations,
            scores=[],  # Cartographer does not score directly
            evidence_refs=result.evidence_refs,
            raw=result.model_dump(),
        )
