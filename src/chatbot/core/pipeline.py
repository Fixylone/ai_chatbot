"""End-to-end generation pipeline orchestrator."""

from __future__ import annotations

import random

from chatbot.core.config import GenerationConfig
from chatbot.core.models import (
    DocumentIssueManifest,
    DocumentRecord,
    SectionIssueManifestEntry,
    ToolDescription,
)
from chatbot.services.ideation import generate_tool_ideation
from chatbot.services.section_generator import generate_document_sections
from chatbot.services.toc_generator import generate_table_of_contents
from chatbot.utils import to_snake_case
from chatbot.utils.html_utils import (
    assemble_document_html,
    build_html_filename,
    write_html_file,
)
from chatbot.utils.json_utils import (
    build_issue_manifest_filename,
    build_toc_json_filename,
    write_json_file,
)
from chatbot.utils.validation import ValidationReport, validate_all

_TEMP_JITTER = 0.05


def _jittered(base_temp: float) -> float:
    """Add small random jitter to a temperature for variety across documents."""
    return max(0.0, min(2.0, base_temp + random.uniform(-_TEMP_JITTER, _TEMP_JITTER)))


async def _generate_single_document(
    config: GenerationConfig,
    tool: ToolDescription,
    document_type: str,
) -> DocumentRecord:
    """Generate and persist one document (TOC JSON + HTML + issues manifest)."""
    tool_dir = config.output_dir / to_snake_case(tool.name)
    toc_path = tool_dir / build_toc_json_filename(document_type)
    html_path = tool_dir / build_html_filename(document_type)

    if html_path.exists():
        print(
            f"[pipeline] skipping {document_type} for {tool.name} — "
            f"already exists at {html_path}",
            flush=True,
        )
        return DocumentRecord(
            tool_name=tool.name,
            document_type=document_type,
            html_path=str(html_path),
            toc_path=str(toc_path),
            issues_manifest_path=str(tool_dir / build_issue_manifest_filename(document_type)),
            total_sections=0,
            issues_summary=[],
        )

    toc_temp = _jittered(config.toc_temperature)
    section_temp = _jittered(config.section_temperature)

    toc = await generate_table_of_contents(
        config, tool, document_type, temperature=toc_temp,
    )
    sections = await generate_document_sections(
        config, tool, toc, temperature=section_temp,
    )

    issues_path = tool_dir / build_issue_manifest_filename(document_type)

    await write_json_file(toc_path, toc.model_dump())
    await write_html_file(html_path, assemble_document_html(toc, sections))

    manifest = DocumentIssueManifest(
        tool_name=tool.name,
        document_type=document_type,
        total_issues=sum(len(s.issues_applied) for s in sections),
        sections=[
            SectionIssueManifestEntry(
                section_id=s.section_id,
                issues_applied=s.issues_applied,
            )
            for s in sections
        ],
    )
    await write_json_file(issues_path, manifest.model_dump())

    all_issues: list[str] = []
    for s in sections:
        all_issues.extend(s.issues_applied)

    return DocumentRecord(
        tool_name=tool.name,
        document_type=document_type,
        html_path=str(html_path),
        toc_path=str(toc_path),
        issues_manifest_path=str(issues_path),
        total_sections=len(sections),
        issues_summary=all_issues,
    )


async def run_pipeline(
    config: GenerationConfig,
) -> tuple[list[DocumentRecord], ValidationReport]:
    """Run full generation pipeline: ideation → TOC → sections → validate."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    expected_docs = config.num_tools * config.docs_per_tool
    print(
        f"[pipeline] starting generation for up to {expected_docs} documents "
        f"(tools={config.num_tools}, docs_per_tool={config.docs_per_tool})",
        flush=True,
    )

    ideation = await generate_tool_ideation(config)
    total_docs = sum(len(t.assigned_doc_types) for t in ideation.tools)
    print(
        f"[pipeline] ideation complete: {len(ideation.tools)} tools, "
        f"{total_docs} documents",
        flush=True,
    )

    records: list[DocumentRecord] = []
    doc_idx = 0

    for tool in ideation.tools:
        for doc_type in tool.assigned_doc_types:
            doc_idx += 1
            print(
                f"[pipeline] ({doc_idx}/{total_docs}) generating "
                f"{doc_type} for {tool.name}",
                flush=True,
            )
            record = await _generate_single_document(config, tool, doc_type)
            records.append(record)
            print(
                f"[pipeline] completed {record.document_type} "
                f"for {record.tool_name}",
                flush=True,
            )

    print("[pipeline] validating generated artifacts", flush=True)
    report = validate_all(config.output_dir)
    print(
        f"[pipeline] validation: {report.valid_files}/{report.total_files} valid",
        flush=True,
    )

    return records, report
