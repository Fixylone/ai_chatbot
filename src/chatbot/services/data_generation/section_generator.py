"""Section-by-section HTML generation service."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import cast

from mirascope import llm

from chatbot.core.config import GenerationConfig
from chatbot.core.models import (
    SectionOutput,
    SectionResponse,
    TableOfContents,
    TOCEntry,
    ToolDescription,
)
from chatbot.services.data_generation.issue_plan_manager import (
    IssuePlanManager,
    SectionIssueRequirement,
)
from chatbot.utils.html_utils import (
    is_valid_html_fragment,
)
from chatbot.utils.llm_runtime import retry_llm_call
from chatbot.utils.prompt_loader import render_prompt

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


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
    previous_sections_html: str,
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
            "previous_sections_html": previous_sections_html or "none",
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

    Each section receives the full HTML of all prior sections
    to maintain coherence across the document.
    """
    ordered = _flatten_toc(toc.sections)
    issue_mgr = IssuePlanManager(ordered)

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
    previous_sections_html = ""
    total_issues = 0
    max_attempts = config.section_max_validation_retries

    for section in ordered:
        section_start = time.monotonic()
        req = issue_mgr.requirement_for(section.id, total_issues)

        print(
            f"[issues] section={section.id} "
            f"remaining={req.issues_remaining} required={req.required_issue_count} "
            f"label={req.required_issue_label or 'none'}",
            flush=True,
        )

        prompt = await _build_section_prompt(
            tool=tool,
            document_type=toc.document_type,
            toc_json=toc_json,
            target_section=section,
            previous_sections_html=previous_sections_html,
            issue_req=req,
        )

        result: SectionOutput | None = None

        for attempt in range(1, max_attempts + 1):
            call_prompt = prompt
            if attempt > 1:
                retry_instruction = (
                    "Retry instruction: Return HTML fragment content in "
                    "html_content, use no unresolved placeholders like "
                    "[CompanyName] or [Company Name], and "
                    f"wrap output in exactly one <section id=\"{section.id}\"> "
                    "element so html_content contains exactly one id "
                    "attribute total and no other id attributes, and "
                    "match issues_applied exactly to required_issue_count and "
                    "required_issue_label. Write issue_sentence BEFORE "
                    "html_content: 'none' when no issue is required, "
                    "otherwise compose a sentence that contains the visible "
                    "defect, then include that sentence verbatim in html_content."
                )
                call_prompt = f"{prompt}\n\n{retry_instruction}"

            response = await retry_llm_call(_section_call, call_prompt, config)

            candidate = cast(SectionResponse, response.parse())
            issues_ok = issue_mgr.issues_match(section.id, candidate.issues_applied)
            sentence = candidate.issue_sentence.strip()
            sentence_ok = False
            if req.required_issue_count == 0:
                sentence_ok = sentence.lower() == "none"
            else:
                sentence_ok = (
                    bool(sentence)
                    and sentence.lower() != "none"
                    and sentence in candidate.html_content
                )

            if (
                issues_ok
                and sentence_ok
                and is_valid_html_fragment(candidate.html_content)
            ):
                result = SectionOutput(
                    section_id=section.id,
                    html_content=candidate.html_content,
                    issues_applied=candidate.issues_applied,
                    issue_sentence=sentence or "none",
                )
                break

        if result is None:
            expected = (
                "[]"
                if req.required_issue_label is None
                else f'["{req.required_issue_label}"]'
            )
            raise ValueError(
                f"Section '{section.id}' failed after {max_attempts} "
                f"attempts. Expected issues_applied={expected}, valid HTML"
            )

        generated.append(result)
        if previous_sections_html:
            previous_sections_html = (
                f"{previous_sections_html}\n\n{result.html_content}"
            )
        else:
            previous_sections_html = result.html_content
        total_issues += len(result.issues_applied)

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
