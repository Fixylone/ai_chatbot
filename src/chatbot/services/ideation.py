"""Tool ideation service.

Generates fictional software tool descriptions using a reasoning-capable model
and assigns document types per tool.
"""

from __future__ import annotations

import random
from pathlib import Path

from mirascope import llm  # type: ignore[import-untyped]

from chatbot.core.config import GenerationConfig
from chatbot.core.models import IdeationLLMResult, IdeationResult, ToolDescription
from chatbot.utils.llm_params import (
    build_llm_call_kwargs,
    build_prompt_cache_key,
    is_prompt_cache_param_error,
)
from chatbot.utils.llm_runtime import call_llm_with_retries
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
) -> list[ToolDescription]:
    """Assign document types to each tool using random sampling."""
    rng = random.Random()

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

    base_call_kwargs = build_llm_call_kwargs(
        temperature=config.ideation_temperature,
        top_p=config.top_p,
    )

    @llm.call(
        config.ideation_model,
        format=llm.format(IdeationLLMResult, mode="strict"),
        **base_call_kwargs,
    )
    def _ideation_call(prompt_text: str) -> str:
        return prompt_text

    active_call = _ideation_call
    if config.prompt_caching_enabled:
        cache_call_kwargs = build_llm_call_kwargs(
            temperature=config.ideation_temperature,
            top_p=config.top_p,
            prompt_cache_key=build_prompt_cache_key(
                config.prompt_cache_key_namespace,
                ["ideation", config.ideation_model],
            ),
            prompt_cache_retention=config.prompt_cache_retention,
        )

        @llm.call(
            config.ideation_model,
            format=llm.format(IdeationLLMResult, mode="strict"),
            **cache_call_kwargs,
        )
        def _ideation_call_cached(prompt_text: str) -> str:
            return prompt_text

        active_call = _ideation_call_cached

    try:
        response = await call_llm_with_retries(
            active_call,
            prompt,
            config,
            stage="ideation",
            model_id=config.ideation_model,
        )
    except Exception as exc:
        if not config.prompt_caching_enabled or not is_prompt_cache_param_error(exc):
            raise
        if config.runtime_feedback:
            print(
                "[cache] stage=ideation provider rejected cache params; "
                "retrying without cache options",
                flush=True,
            )
        response = await call_llm_with_retries(
            _ideation_call,
            prompt,
            config,
            stage="ideation",
            model_id=config.ideation_model,
        )

    ideation_llm_result = response.parse()
    internal_tools = [
        ToolDescription(
            name=tool.name,
            purpose=tool.purpose,
            category=tool.category,
            typical_user_base=tool.typical_user_base,
        )
        for tool in ideation_llm_result.tools
    ]

    tools_with_docs = _assign_document_types(
        tools=internal_tools,
        document_types=config.document_types,
        docs_per_tool=config.docs_per_tool,
    )

    return IdeationResult(tools=tools_with_docs)
