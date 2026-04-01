"""Tool ideation service — generates fictional software tools via LLM."""

from __future__ import annotations

import random
from pathlib import Path
from typing import cast

from mirascope import llm  # type: ignore[import-untyped]

from chatbot.core.config import GenerationConfig
from chatbot.core.models import IdeationResponse, IdeationResult, ToolDescription
from chatbot.utils.llm_runtime import call_llm_with_retries
from chatbot.utils.prompt_loader import render_prompt

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _format_doc_types_bulleted(doc_types: list[str]) -> str:
    return "\n".join(f"- {dt}" for dt in doc_types)


def _assign_document_types(
    tools: list[ToolDescription],
    document_types: list[str],
    docs_per_tool: int,
) -> None:
    rng = random.Random()
    for tool in tools:
        tool.assigned_doc_types = rng.sample(document_types, k=docs_per_tool)


async def generate_tool_ideation(config: GenerationConfig) -> IdeationResult:
    """Generate fictional tools and assign document types to each."""
    system_prompt = await render_prompt(_PROMPTS_DIR / "ideation_system.yaml")
    user_prompt = await render_prompt(
        _PROMPTS_DIR / "ideation_user.yaml",
        variables={
            "num_tools": config.num_tools,
            "document_types_bulleted": _format_doc_types_bulleted(
                config.document_types
            ),
        },
    )
    prompt = f"{system_prompt}\n\n{user_prompt}"

    @llm.call(
        config.ideation_model,
        format=llm.format(IdeationResponse, mode="strict"),
        temperature=config.ideation_temperature,
        top_p=config.top_p,
    )
    def _ideation_call(prompt_text: str) -> str:
        return prompt_text

    response = await call_llm_with_retries(
        _ideation_call,
        prompt,
        config,
        stage="ideation",
        model_id=config.ideation_model,
    )
    parsed = cast(IdeationResponse, response.parse())  # type: ignore[attr-defined]

    tools = [
        ToolDescription(
            name=t.name,
            purpose=t.purpose,
            category=t.category,
            typical_user_base=t.typical_user_base,
        )
        for t in parsed.tools
    ]
    _assign_document_types(tools, config.document_types, config.docs_per_tool)

    return IdeationResult(tools=tools)
