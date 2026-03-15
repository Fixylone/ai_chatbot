"""Section-by-section HTML generation service."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mirascope import llm  # type: ignore[import-untyped]

from chatbot.core.config import GenerationConfig
from chatbot.core.models import (
    SectionOutput,
    TableOfContents,
    TOCEntry,
    ToolDescription,
)
from chatbot.utils.html_utils import is_valid_html_fragment
from chatbot.utils.prompt_loader import render_prompt

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_SECTION_SYSTEM_PROMPT = _PROMPTS_DIR / "section_system.yaml"
_SECTION_USER_PROMPT = _PROMPTS_DIR / "section_user.yaml"
_MAX_SECTION_ATTEMPTS = 3


def _flatten_toc_preorder(entries: list[TOCEntry]) -> list[TOCEntry]:
    """Flatten TOC entries in preorder traversal order."""
    ordered: list[TOCEntry] = []

    for entry in entries:
        ordered.append(entry)
        if entry.children:
            ordered.extend(_flatten_toc_preorder(entry.children))

    return ordered


async def _build_section_prompt(
    tool: ToolDescription,
    document_type: str,
    toc: TableOfContents,
    target_section: TOCEntry,
    previous_sections_html: str,
) -> str:
    """Build prompt for generating one section."""
    system_prompt = await render_prompt(_SECTION_SYSTEM_PROMPT)
    user_prompt = await render_prompt(
        _SECTION_USER_PROMPT,
        variables={
            "tool_name": tool.name,
            "tool_purpose": tool.purpose,
            "tool_category": tool.category,
            "tool_user_base": tool.typical_user_base,
            "document_type": document_type,
            "section_id": target_section.id,
            "section_title": target_section.title,
            "toc_json": toc.model_dump_json(indent=2),
            "previous_sections_html": previous_sections_html,
        },
    )
    return f"{system_prompt}\n\n{user_prompt}"


async def generate_document_sections(
    config: GenerationConfig,
    tool: ToolDescription,
    toc: TableOfContents,
) -> list[SectionOutput]:
    """Generate all document sections sequentially from a TOC.

    Each section call receives all previously generated sections to preserve
    coherence and logical flow.

    Args:
        config: Generation settings.
        tool: Tool context.
        toc: TOC for the target document.

    Returns:
        Ordered section outputs matching TOC traversal.
    """
    ordered_sections = _flatten_toc_preorder(toc.sections)
    generated_sections: list[SectionOutput] = []

    @llm.call(
        config.section_model,
        format=llm.format(SectionOutput, mode="strict"),
        temperature=config.section_temperature,
        top_p=config.top_p,
        seed=config.seed,
    )
    def _section_call(prompt_text: str) -> str:
        return prompt_text

    for section in ordered_sections:
        previous_sections_html = "\n\n".join(
            item.html_content for item in generated_sections
        )

        prompt = await _build_section_prompt(
            tool=tool,
            document_type=toc.document_type,
            toc=toc,
            target_section=section,
            previous_sections_html=previous_sections_html,
        )

        section_output: SectionOutput | None = None

        for attempt in range(1, _MAX_SECTION_ATTEMPTS + 1):
            call_prompt = prompt
            if attempt > 1:
                call_prompt = (
                    f"{prompt}\n\n"
                    "Retry instruction: The previous output was not valid HTML. "
                    "Return only valid HTML fragment content in html_content."
                )

            response = await asyncio.to_thread(_section_call, call_prompt)
            candidate = response.parse()

            if is_valid_html_fragment(candidate.html_content):
                section_output = candidate
                break

        if section_output is None:
            msg = (
                f"Model failed to generate valid HTML fragment for section "
                f"'{section.id}' after {_MAX_SECTION_ATTEMPTS} attempts."
            )
            raise ValueError(msg)

        # Ensure section id is always consistent with TOC node id.
        section_output.section_id = section.id
        generated_sections.append(section_output)

    return generated_sections
