"""Table-of-contents generation service."""

from __future__ import annotations

from pathlib import Path

from mirascope import llm  # type: ignore[import-untyped]

from chatbot.core.config import GenerationConfig
from chatbot.core.models import (
    TOCEntry,
    TOCEntryLLM,
    TableOfContents,
    TableOfContentsLLM,
    ToolDescription,
)
from chatbot.utils.llm_params import (
    build_llm_call_kwargs,
    build_prompt_cache_key,
    is_prompt_cache_param_error,
)
from chatbot.utils.llm_runtime import call_llm_with_retries
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


def _map_toc_entry(node: TOCEntryLLM) -> TOCEntry:
    """Convert an LLM TOC node into the internal recursive model."""
    return TOCEntry(
        id=node.id,
        title=node.title,
        children=[_map_toc_entry(child) for child in node.children],
    )


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

    base_call_kwargs = build_llm_call_kwargs(
        temperature=config.toc_temperature,
        top_p=config.top_p,
    )

    @llm.call(
        config.toc_model,
        format=llm.format(TableOfContentsLLM, mode="strict"),
        **base_call_kwargs,
    )
    def _toc_call(prompt_text: str) -> str:
        return prompt_text

    active_call = _toc_call
    if config.prompt_caching_enabled:
        cache_call_kwargs = build_llm_call_kwargs(
            temperature=config.toc_temperature,
            top_p=config.top_p,
            prompt_cache_key=build_prompt_cache_key(
                config.prompt_cache_key_namespace,
                ["toc", config.toc_model, tool.name, document_type],
            ),
            prompt_cache_retention=config.prompt_cache_retention,
        )

        @llm.call(
            config.toc_model,
            format=llm.format(TableOfContentsLLM, mode="strict"),
            **cache_call_kwargs,
        )
        def _toc_call_cached(prompt_text: str) -> str:
            return prompt_text

        active_call = _toc_call_cached

    toc_sections: list[TOCEntry] | None = None

    for attempt in range(1, _MAX_TOC_ATTEMPTS + 1):
        call_prompt = base_prompt
        if attempt > 1:
            call_prompt = (
                f"{base_prompt}\n\n"
                "Retry instruction: Ensure the outline includes meaningful "
                "nesting with at least some third-level structure where natural."
            )

        try:
            response = await call_llm_with_retries(
                active_call,
                call_prompt,
                config,
                stage=f"toc:{document_type}:attempt-{attempt}",
                model_id=config.toc_model,
            )
        except Exception as exc:
            if (
                not config.prompt_caching_enabled
                or active_call is _toc_call
                or not is_prompt_cache_param_error(exc)
            ):
                raise
            if config.runtime_feedback:
                print(
                    "[cache] stage=toc provider rejected cache params; "
                    "retrying without cache options",
                    flush=True,
                )
            active_call = _toc_call
            response = await call_llm_with_retries(
                active_call,
                call_prompt,
                config,
                stage=f"toc:{document_type}:attempt-{attempt}",
                model_id=config.toc_model,
            )

        candidate = response.parse()
        mapped_sections = [_map_toc_entry(entry) for entry in candidate.sections]

        if _max_toc_depth(mapped_sections) >= 3:
            toc_sections = mapped_sections
            break

    if toc_sections is None:
        msg = (
            f"Model failed to generate a sufficiently nested TOC for "
            f"'{document_type}' after {_MAX_TOC_ATTEMPTS} attempts."
        )
        raise ValueError(msg)

    return TableOfContents(
        document_type=document_type,
        tool_name=tool.name,
        sections=toc_sections,
    )
