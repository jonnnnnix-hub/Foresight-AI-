"""Judge — synthesizes specialist outputs into final recommendations."""

from __future__ import annotations

import json

from flowedge.agents.base import AgentOutput
from flowedge.agents.llm import call_agent_llm
from flowedge.agents.prompts import JUDGE_PROMPT
from flowedge.schemas.agents import JudgeOutput


async def run_judge(
    specialist_outputs: dict[str, list[AgentOutput]],
) -> JudgeOutput:
    """Synthesize all specialist outputs into a final judgment.

    Args:
        specialist_outputs: repo_name -> list of AgentOutput from all specialists
    """
    # Format specialist outputs for the judge
    parts: list[str] = []
    for repo_name, outputs in specialist_outputs.items():
        parts.append(f"\n## {repo_name}")
        for output in outputs:
            parts.append(f"\n### {output.agent_name}")
            parts.append(f"Summary: {output.summary}")
            parts.append(f"Findings: {json.dumps(output.findings[:10])}")
            for score in output.scores:
                parts.append(
                    f"Score [{score.dimension}]: {score.score}/10 — {score.rationale}"
                )

    combined = "\n".join(parts)
    return await call_agent_llm(
        system_prompt=JUDGE_PROMPT,
        user_content=f"Synthesize these specialist findings:\n{combined}",
        output_type=JudgeOutput,
    )
