"""Table-of-contents generation service."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mirascope import llm  # type: ignore[import-untyped]

from chatbot.core.config import GenerationConfig
from chatbot.core.models import TableOfContents, TOCEntry, ToolDescription
from chatbot.utils.prompt_loader import render_prompt

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_TOC_SYSTEM_PROMPT = _PROMPTS_DIR / "toc_system.yaml"
_TOC_USER_PROMPT = _PROMPTS_DIR / "toc_user.yaml"
_MAX_TOC_ATTEMPTS = 3


def _max_toc_depth(sections: list[TOCEntry]) -> int:
    """Return max TOC depth for nested section tree."""

    def _depth(node: TOCEntry) -> int:
        children = node.children
        if not children:
            return 1
        return 1 + max(_depth(child) for child in children)

    if not sections:
        return 0

    return max(_depth(section) for section in sections)


async def _build_toc_prompt(
    tool: ToolDescription,
    document_type: str,
) -> str:
    """Build the final TOC prompt from system and user templates."""
    system_prompt = await render_prompt(_TOC_SYSTEM_PROMPT)
    user_prompt = await render_prompt(
        _TOC_USER_PROMPT,
        variables={
            "tool_name": tool.name,
            "tool_purpose": tool.purpose,
            "tool_category": tool.category,
            "tool_user_base": tool.typical_user_base,
            "document_type": document_type,
        },
    )
    return f"{system_prompt}\n\n{user_prompt}"


async def generate_table_of_contents(
    config: GenerationConfig,
    tool: ToolDescription,
    document_type: str,
) -> TableOfContents:
    """Generate a structured TOC for one tool + document type.

    Args:
        config: Generation settings.
        tool: Tool context produced by ideation.
        document_type: Target legal document type.

    Returns:
        Structured table of contents.
    """
    base_prompt = await _build_toc_prompt(tool, document_type)

    @llm.call(
        config.toc_model,
        format=llm.format(TableOfContents, mode="strict"),
        temperature=config.toc_temperature,
        top_p=config.top_p,
        seed=config.seed,
    )
    def _toc_call(prompt_text: str) -> str:
        return prompt_text

    toc: TableOfContents | None = None

    for attempt in range(1, _MAX_TOC_ATTEMPTS + 1):
        call_prompt = base_prompt
        if attempt > 1:
            call_prompt = (
                f"{base_prompt}\n\n"
                "Retry instruction: Ensure the outline includes meaningful "
                "nesting with at least some third-level structure where natural."
            )

        response = await asyncio.to_thread(_toc_call, call_prompt)
        candidate = response.parse()

        if _max_toc_depth(candidate.sections) >= 3:
            toc = candidate
            break

    if toc is None:
        msg = (
            f"Model failed to generate a sufficiently nested TOC for "
            f"'{document_type}' after {_MAX_TOC_ATTEMPTS} attempts."
        )
        raise ValueError(msg)

    # Keep envelope metadata aligned with current generation context.
    toc.document_type = document_type
    toc.tool_name = tool.name
    return toc
