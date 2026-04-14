"""System prompts for each specialist agent, loaded from .claude/agents/*.md."""

from pathlib import Path


def _load_prompt(filename: str) -> str:
    """Load an agent prompt from the .claude/agents directory."""
    # Try relative to project root
    for base in [Path.cwd(), Path(__file__).parent.parent.parent.parent]:
        path = base / ".claude" / "agents" / filename
        if path.exists():
            content = path.read_text()
            # Strip YAML frontmatter
            if content.startswith("---"):
                end = content.find("---", 3)
                if end > 0:
                    content = content[end + 3 :].strip()
            return content
    raise FileNotFoundError(f"Agent prompt not found: {filename}")


CARTOGRAPHER_PROMPT = _load_prompt("repo-cartographer.md")
RESEARCH_ANALYST_PROMPT = _load_prompt("research-analyst.md")
EXECUTION_ANALYST_PROMPT = _load_prompt("execution-analyst.md")
ML_ANALYST_PROMPT = _load_prompt("ml-analyst.md")
PRODUCT_ARCHITECT_PROMPT = _load_prompt("product-architect.md")
RISK_ANALYST_PROMPT = _load_prompt("risk-analyst.md")
SKEPTIC_PROMPT = _load_prompt("skeptic.md")
JUDGE_PROMPT = _load_prompt("judge.md")
