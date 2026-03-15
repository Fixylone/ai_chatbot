"""Tool ideation service.

Generates fictional software tool descriptions using a reasoning-capable model
and assigns document types per tool in a reproducible way.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

from mirascope import llm  # type: ignore[import-untyped]

from chatbot.core.config import GenerationConfig
from chatbot.core.models import IdeationResult, ToolDescription
from chatbot.utils.prompt_loader import render_prompt

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_IDEATION_SYSTEM_PROMPT = _PROMPTS_DIR / "ideation_system.yaml"
_IDEATION_USER_PROMPT = _PROMPTS_DIR / "ideation_user.yaml"


def _format_document_types_bulleted(document_types: list[str]) -> str:
    """Format document types as markdown bullet list for prompt readability."""
    return "\n".join(f"- {doc_type}" for doc_type in document_types)


def _assign_document_types(
    tools: list[ToolDescription],
    document_types: list[str],
    docs_per_tool: int,
    seed: int,
) -> list[ToolDescription]:
    """Assign document types to each tool using deterministic random sampling."""
    rng = random.Random(seed)

    for tool in tools:
        assigned = rng.sample(document_types, k=docs_per_tool)
        tool.assigned_doc_types = assigned

    return tools


async def _build_ideation_prompt(config: GenerationConfig) -> str:
    """Combine system and user templates into a single prompt string."""
    system_prompt = await render_prompt(_IDEATION_SYSTEM_PROMPT)
    user_prompt = await render_prompt(
        _IDEATION_USER_PROMPT,
        variables={
            "num_tools": config.num_tools,
            "document_types_bulleted": _format_document_types_bulleted(
                config.document_types
            ),
        },
    )
    return f"{system_prompt}\n\n{user_prompt}"


async def generate_tool_ideation(config: GenerationConfig) -> IdeationResult:
    """Generate fictional tools and assign document types.

    Args:
        config: Generation configuration.

    Returns:
        Structured ideation result with assigned document types.
    """
    prompt = await _build_ideation_prompt(config)

    @llm.call(
        config.ideation_model,
        format=llm.format(IdeationResult, mode="strict"),
        temperature=config.ideation_temperature,
        top_p=config.top_p,
        seed=config.seed,
    )
    def _ideation_call(prompt_text: str) -> str:
        return prompt_text

    response = await asyncio.to_thread(_ideation_call, prompt)
    ideation_result = response.parse()

    tools_with_docs = _assign_document_types(
        tools=ideation_result.tools,
        document_types=config.document_types,
        docs_per_tool=config.docs_per_tool,
        seed=config.seed,
    )

    return IdeationResult(tools=tools_with_docs)
