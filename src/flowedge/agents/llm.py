"""LLM provider abstraction for specialist agents."""

from __future__ import annotations

import json
from typing import TypeVar

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from flowedge.config.settings import get_settings

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


def get_model(model_name: str = "claude-sonnet-4-20250514") -> ChatAnthropic:
    """Return a configured Anthropic chat model."""
    settings = get_settings()
    return ChatAnthropic(
        model=model_name,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
        temperature=0.0,
    )


async def call_agent_llm(
    system_prompt: str,
    user_content: str,
    output_type: type[T],
    model_name: str = "claude-sonnet-4-20250514",
) -> T:
    """Call LLM with a system prompt and parse structured output.

    Uses with_structured_output for reliable schema-conformant responses.
    Falls back to raw JSON extraction if structured output fails.
    """
    model = get_model(model_name)

    try:
        structured_model = model.with_structured_output(output_type)
        result = await structured_model.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
        )
        if isinstance(result, output_type):
            return result
        # If the model returned a dict, parse it
        return output_type.model_validate(result)
    except Exception as e:
        logger.warning("structured_output_failed", error=str(e), fallback="raw_json")

    # Fallback: raw call + JSON extraction
    raw_result = await model.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"{user_content}\n\nRespond with valid JSON matching this schema:\n"
                f"{json.dumps(output_type.model_json_schema(), indent=2)}"
            ),
        ]
    )
    content = raw_result.content
    if isinstance(content, str):
        # Try to extract JSON from the response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return output_type.model_validate_json(content[start:end])
    raise ValueError(f"Failed to parse {output_type.__name__} from LLM response")
