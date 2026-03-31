"""Section-by-section HTML generation service."""

from __future__ import annotations

from pathlib import Path

from mirascope import llm  # type: ignore[import-untyped]

from chatbot.core.config import GenerationConfig
from chatbot.core.models import (
    SectionLLMOutput,
    SectionOutput,
    TableOfContents,
    TOCEntry,
    ToolDescription,
)
from chatbot.services.issue_plan_manager import IssuePlanManager, SectionIssueRequirement
from chatbot.services.section_context_compressor import SectionContextCompressor
from chatbot.utils.html_utils import (
    contains_unresolved_company_placeholder,
    is_valid_html_fragment,
)
from chatbot.utils.llm_params import (
    build_llm_call_kwargs,
    build_prompt_cache_key,
    is_prompt_cache_param_error,
)
from chatbot.utils.llm_runtime import call_llm_with_retries
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
    toc_json: str,
    target_section: TOCEntry,
    previous_sections_summary: str,
    last_full_section_html: str,
    issue_requirement: SectionIssueRequirement,
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
            "toc_json": toc_json,
            "previous_sections_summary": previous_sections_summary,
            "last_full_section_html": last_full_section_html,
            "target_issue_range": "2-3",
            "target_issue_total": issue_requirement.target_issue_total,
            "required_issue_count": issue_requirement.required_issue_count,
            "required_issue_label": issue_requirement.required_issue_label
            or "none",
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
    """
    ordered_sections = _flatten_toc_preorder(toc.sections)
    issue_plan_manager = IssuePlanManager(ordered_sections)
    context_compressor = SectionContextCompressor(
        max_summary_sections=config.section_summary_max_sections,
        max_summary_chars=config.section_summary_max_chars,
        max_last_section_chars=config.section_last_section_max_chars,
    )

    generated_sections: list[SectionOutput] = []
    total_issues_applied = 0
    target_issue_total = issue_plan_manager.target_issue_total

    if config.runtime_feedback:
        print(
            "[issues] "
            f"target_total={target_issue_total} "
            f"document_type={toc.document_type}",
            flush=True,
        )

    base_call_kwargs = build_llm_call_kwargs(
        temperature=config.section_temperature,
        top_p=config.top_p,
    )

    @llm.call(
        config.section_model,
        format=llm.format(SectionLLMOutput, mode="strict"),
        **base_call_kwargs,
    )
    def _section_call(prompt_text: str) -> str:
        return prompt_text

    active_section_call = _section_call
    if config.prompt_caching_enabled:
        cache_call_kwargs = build_llm_call_kwargs(
            temperature=config.section_temperature,
            top_p=config.top_p,
            prompt_cache_key=build_prompt_cache_key(
                config.prompt_cache_key_namespace,
                ["section", config.section_model, tool.name, toc.document_type],
            ),
            prompt_cache_retention=config.prompt_cache_retention,
        )

        @llm.call(
            config.section_model,
            format=llm.format(SectionLLMOutput, mode="strict"),
            **cache_call_kwargs,
        )
        def _section_call_cached(prompt_text: str) -> str:
            return prompt_text

        active_section_call = _section_call_cached

    toc_json = toc.model_dump_json()

    for section in ordered_sections:
        compressed_context = context_compressor.current_context()
        section_requirement = issue_plan_manager.requirement_for(
            section.id,
            total_issues_applied,
        )

        if config.runtime_feedback:
            print(
                "[issues] "
                f"section={section.id} "
                f"remaining={section_requirement.issues_remaining} "
                f"required={section_requirement.required_issue_count}",
                flush=True,
            )

        prompt = await _build_section_prompt(
            tool=tool,
            document_type=toc.document_type,
            toc_json=toc_json,
            target_section=section,
            previous_sections_summary=compressed_context.previous_sections_summary,
            last_full_section_html=compressed_context.last_full_section_html,
            issue_requirement=section_requirement,
        )

        section_output: SectionOutput | None = None

        for attempt in range(1, _MAX_SECTION_ATTEMPTS + 1):
            call_prompt = prompt
            if attempt > 1:
                call_prompt = (
                    f"{prompt}\n\n"
                    "Retry instruction: Return valid HTML fragment content in "
                    "html_content, use no unresolved placeholders like "
                    "[CompanyName] or [Company Name], and "
                    "match issues_applied exactly to required_issue_count and "
                    "required_issue_label."
                )

            stage_name = f"section:{toc.document_type}:{section.id}:attempt-{attempt}"
            try:
                response = await call_llm_with_retries(
                    active_section_call,
                    call_prompt,
                    config,
                    stage=stage_name,
                    model_id=config.section_model,
                )
            except Exception as exc:
                if (
                    not config.prompt_caching_enabled
                    or active_section_call is _section_call
                    or not is_prompt_cache_param_error(exc)
                ):
                    raise
                if config.runtime_feedback:
                    print(
                        "[cache] stage=section provider rejected cache params; "
                        "retrying without cache options",
                        flush=True,
                    )
                active_section_call = _section_call
                response = await call_llm_with_retries(
                    active_section_call,
                    call_prompt,
                    config,
                    stage=stage_name,
                    model_id=config.section_model,
                )

            candidate = response.parse()
            issues_match = issue_plan_manager.issues_match(
                section.id,
                candidate.issues_applied,
            )
            has_placeholder = contains_unresolved_company_placeholder(
                candidate.html_content
            )
            is_valid_html = False
            if issues_match and not has_placeholder:
                is_valid_html = is_valid_html_fragment(candidate.html_content)

            if is_valid_html and issues_match and not has_placeholder:
                section_output = SectionOutput(
                    section_id=section.id,
                    html_content=candidate.html_content,
                    issues_applied=candidate.issues_applied,
                )
                break

        if section_output is None:
            expected_issues = (
                "[]"
                if section_requirement.required_issue_label is None
                else f'["{section_requirement.required_issue_label}"]'
            )
            msg = (
                f"Model failed section '{section.id}' after "
                f"{_MAX_SECTION_ATTEMPTS} attempts. Expected issues_applied="
                f"{expected_issues}, valid HTML fragment, and no unresolved "
                "[CompanyName]/[Company Name] placeholder."
            )
            raise ValueError(msg)

        generated_sections.append(section_output)
        total_issues_applied += len(section_output.issues_applied)
        context_compressor.observe_section(section_output)

    issue_plan_manager.validate_document_totals(total_issues_applied)

    return generated_sections

