"""Table-of-contents generation service."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from mirascope import llm  # type: ignore[import-untyped]

from chatbot.core.config import GenerationConfig
from chatbot.core.models import (
    TOCEntry,
    TOCEntryResponse,
    TableOfContents,
    TOCResponse,
    ToolDescription,
)
from chatbot.utils.llm_runtime import call_llm_with_retries
from chatbot.utils.prompt_loader import render_prompt

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_MAX_TOC_ATTEMPTS = 3


def _max_depth(sections: list[TOCEntry]) -> int:
    if not sections:
        return 0

    def _depth(node: TOCEntry) -> int:
        if not node.children:
            return 1
        return 1 + max(_depth(c) for c in node.children)

    return max(_depth(s) for s in sections)


def _map_entry(node: TOCEntryResponse) -> TOCEntry:
    return TOCEntry(
        id=node.id,
        title=node.title,
        children=[_map_entry(c) for c in node.children],
    )


async def generate_table_of_contents(
    config: GenerationConfig,
    tool: ToolDescription,
    document_type: str,
    *,
    temperature: float | None = None,
) -> TableOfContents:
    """Generate a structured TOC for one tool + document type."""
    system_prompt = await render_prompt(_PROMPTS_DIR / "toc_system.yaml")
    user_prompt = await render_prompt(
        _PROMPTS_DIR / "toc_user.yaml",
        variables={
            "tool_name": tool.name,
            "tool_purpose": tool.purpose,
            "tool_category": tool.category,
            "tool_user_base": tool.typical_user_base,
            "document_type": document_type,
        },
    )
    base_prompt = f"{system_prompt}\n\n{user_prompt}"

    temp = temperature if temperature is not None else config.toc_temperature

    @llm.call(
        config.toc_model,
        format=llm.format(TOCResponse, mode="strict"),
        temperature=temp,
        top_p=config.top_p,
    )
    def _toc_call(prompt_text: str) -> str:
        return prompt_text

    toc_sections: list[TOCEntry] | None = None

    for attempt in range(1, _MAX_TOC_ATTEMPTS + 1):
        prompt = base_prompt
        if attempt > 1:
            prompt += (
                "\n\nRetry instruction: Ensure the outline includes meaningful "
                "nesting with at least some third-level structure where natural."
            )

        response = await call_llm_with_retries(
            _toc_call,
            prompt,
            config,
            stage=f"toc:{document_type}:attempt-{attempt}",
            model_id=config.toc_model,
        )
        candidate = cast(TOCResponse, response.parse())  # type: ignore[attr-defined]
        mapped = [_map_entry(e) for e in candidate.sections]

        if _max_depth(mapped) >= 3:
            toc_sections = mapped
            break

    if toc_sections is None:
        raise ValueError(
            f"TOC for '{document_type}' lacked sufficient depth "
            f"after {_MAX_TOC_ATTEMPTS} attempts."
        )

    return TableOfContents(
        document_type=document_type,
        tool_name=tool.name,
        sections=toc_sections,
    )
