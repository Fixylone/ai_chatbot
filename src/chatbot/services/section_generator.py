"""Section-by-section HTML generation service."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import cast

from mirascope import llm

from chatbot.core.config import GenerationConfig
from chatbot.core.models import (
    SectionResponse,
    SectionOutput,
    TableOfContents,
    TOCEntry,
    ToolDescription,
)
from chatbot.services.issue_plan_manager import IssuePlanManager, SectionIssueRequirement
from chatbot.services.section_context_compressor import SectionContextCompressor
from chatbot.utils.html_utils import (
    is_valid_html_fragment,
)
from chatbot.utils.llm_runtime import call_llm_with_retries
from chatbot.utils.prompt_loader import render_prompt

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_MAX_SECTION_ATTEMPTS = 3


def _flatten_toc(entries: list[TOCEntry]) -> list[TOCEntry]:
    flat: list[TOCEntry] = []
    for entry in entries:
        flat.append(entry)
        if entry.children:
            flat.extend(_flatten_toc(entry.children))
    return flat


async def _build_section_prompt(
    tool: ToolDescription,
    document_type: str,
    toc_json: str,
    target_section: TOCEntry,
    previous_sections_summary: str,
    last_full_section_html: str,
    issue_req: SectionIssueRequirement,
) -> str:
    system_prompt = await render_prompt(_PROMPTS_DIR / "section_system.yaml")
    user_prompt = await render_prompt(
        _PROMPTS_DIR / "section_user.yaml",
        variables={
            "tool_name": tool.name,
            "tool_purpose": tool.purpose,
            "tool_category": tool.category,
            "tool_user_base": tool.typical_user_base,
            "document_type": document_type,
            "toc_json": toc_json,
            "previous_sections_summary": previous_sections_summary,
            "last_full_section_html": last_full_section_html,
            "section_id": target_section.id,
            "section_title": target_section.title,
            "inject_issue": "yes" if issue_req.required_issue_count > 0 else "no",
            "required_issue_label": issue_req.required_issue_label or "none",
        },
    )
    return f"{system_prompt}\n\n{user_prompt}"


async def generate_document_sections(
    config: GenerationConfig,
    tool: ToolDescription,
    toc: TableOfContents,
    *,
    temperature: float | None = None,
) -> list[SectionOutput]:
    """Generate all sections for a document sequentially.

    Each section receives compressed context from all prior sections
    to maintain coherence across the document.
    """
    ordered = _flatten_toc(toc.sections)
    issue_mgr = IssuePlanManager(ordered)
    compressor = SectionContextCompressor(
        max_summary_chars=config.section_summary_max_chars,
        max_last_section_chars=config.section_last_section_max_chars,
    )

    print(
        f"[issues] target_total={issue_mgr.target_issue_total} "
        f"document_type={toc.document_type}",
        flush=True,
    )

    temp = temperature if temperature is not None else config.section_temperature

    @llm.call(
        config.section_model,
        format=llm.format(SectionResponse, mode="strict"),
        temperature=temp,
        top_p=config.top_p,
    )
    def _section_call(prompt_text: str) -> str:
        return prompt_text

    toc_json = toc.model_dump_json()
    generated: list[SectionOutput] = []
    total_issues = 0

    for section in ordered:
        section_start = time.monotonic()
        ctx = compressor.current_context()
        req = issue_mgr.requirement_for(section.id, total_issues)

        print(
            f"[issues] section={section.id} "
            f"remaining={req.issues_remaining} required={req.required_issue_count}",
            flush=True,
        )

        prompt = await _build_section_prompt(
            tool=tool,
            document_type=toc.document_type,
            toc_json=toc_json,
            target_section=section,
            previous_sections_summary=ctx.previous_sections_summary,
            last_full_section_html=ctx.last_full_section_html,
            issue_req=req,
        )

        result: SectionOutput | None = None

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

            stage = f"section:{toc.document_type}:{section.id}:attempt-{attempt}"
            response = await call_llm_with_retries(
                _section_call, call_prompt, config,
                stage=stage, model_id=config.section_model,
            )

            candidate = cast(SectionResponse, response.parse())
            issues_ok = issue_mgr.issues_match(section.id, candidate.issues_applied)

            if issues_ok and is_valid_html_fragment(candidate.html_content):
                result = SectionOutput(
                    section_id=section.id,
                    html_content=candidate.html_content,
                    issues_applied=candidate.issues_applied,
                )
                break

        if result is None:
            expected = (
                "[]" if req.required_issue_label is None
                else f'["{req.required_issue_label}"]'
            )
            raise ValueError(
                f"Section '{section.id}' failed after {_MAX_SECTION_ATTEMPTS} "
                f"attempts. Expected issues_applied={expected}, valid HTML, "
                "no [CompanyName] placeholder."
            )

        generated.append(result)
        total_issues += len(result.issues_applied)
        compressor.observe_section(result)

        # Adaptive RPM pacing — only sleep if request was faster than the
        # minimum interval derived from the RPM limit.
        min_interval = 60.0 / config.rpm_limit
        elapsed = time.monotonic() - section_start
        gap = max(config.section_delay_seconds, min_interval) - elapsed
        if gap > 0:
            print(
                f"[pacing] sleeping {gap:.2f}s "
                f"(elapsed={elapsed:.2f}s, min_interval={min_interval:.2f}s)",
                flush=True,
            )
            await asyncio.sleep(gap)

    issue_mgr.validate_document_totals(total_issues)
    return generated

