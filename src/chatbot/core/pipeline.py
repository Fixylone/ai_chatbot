"""End-to-end generation pipeline orchestrator."""

from __future__ import annotations

import re

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

_SLUG_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    """Convert free text to deterministic snake-case slug.

    Args:
        value: Raw text.

    Returns:
        Lowercase slug with underscore separators.
    """
    lowered = value.lower().strip().replace("&", " and ")
    collapsed = _SLUG_NON_ALNUM_PATTERN.sub("_", lowered)
    return collapsed.strip("_")


async def _generate_single_document(
    config: GenerationConfig,
    tool: ToolDescription,
    document_type: str,
) -> DocumentRecord:
    """Generate and persist one document pair (TOC JSON + HTML).

    Args:
        config: Generation configuration.
        tool: Tool context from ideation.
        document_type: Target document type.

    Returns:
        Document generation metadata.
    """
    toc = await generate_table_of_contents(config, tool, document_type)
    sections = await generate_document_sections(config, tool, toc)

    tool_dir = config.output_dir / _slugify(tool.name)
    toc_path = tool_dir / build_toc_json_filename(document_type)
    html_path = tool_dir / build_html_filename(document_type)
    issues_manifest_path = tool_dir / build_issue_manifest_filename(document_type)

    await write_json_file(toc_path, toc.model_dump())

    assembled_html = assemble_document_html(toc, sections)
    await write_html_file(html_path, assembled_html)

    issue_manifest = DocumentIssueManifest(
        tool_name=tool.name,
        document_type=document_type,
        total_issues=sum(len(section.issues_applied) for section in sections),
        sections=[
            SectionIssueManifestEntry(
                section_id=section.section_id,
                issues_applied=section.issues_applied,
            )
            for section in sections
        ],
    )
    await write_json_file(issues_manifest_path, issue_manifest.model_dump())

    issues_summary: list[str] = []
    for section in sections:
        issues_summary.extend(section.issues_applied)

    return DocumentRecord(
        tool_name=tool.name,
        document_type=document_type,
        html_path=str(html_path),
        toc_path=str(toc_path),
        issues_manifest_path=str(issues_manifest_path),
        total_sections=len(sections),
        issues_summary=issues_summary,
    )


async def run_pipeline(
    config: GenerationConfig,
) -> tuple[list[DocumentRecord], ValidationReport]:
    """Run full synthetic-data generation pipeline.

    Flow:
        1. Tool ideation.
        2. TOC generation per document.
        3. Section-by-section generation per document.
        4. HTML assembly and file persistence.
        5. Validation over output directory.

    Args:
        config: Pipeline configuration.

    Returns:
        Tuple with document metadata records and validation report.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.runtime_feedback:
        expected_docs = config.num_tools * config.docs_per_tool
        print(
            "[pipeline] "
            f"starting generation for up to {expected_docs} documents "
            f"(tools={config.num_tools}, docs_per_tool={config.docs_per_tool})",
            flush=True,
        )

    ideation = await generate_tool_ideation(config)
    records: list[DocumentRecord] = []
    total_docs = sum(len(tool.assigned_doc_types) for tool in ideation.tools)

    if config.runtime_feedback:
        print(
            "[pipeline] "
            f"ideation complete: {len(ideation.tools)} tools, {total_docs} documents",
            flush=True,
        )

    doc_index = 0

    for tool in ideation.tools:
        for document_type in tool.assigned_doc_types:
            doc_index += 1
            if config.runtime_feedback:
                print(
                    "[pipeline] "
                    f"({doc_index}/{total_docs}) generating "
                    f"{document_type} for {tool.name}",
                    flush=True,
                )

            record = await _generate_single_document(
                config=config,
                tool=tool,
                document_type=document_type,
            )
            records.append(record)

            if config.runtime_feedback:
                print(
                    "[pipeline] "
                    f"completed {record.document_type} for {record.tool_name}",
                    flush=True,
                )

    if config.runtime_feedback:
        print("[pipeline] validating generated artifacts", flush=True)

    report = validate_all(config.output_dir)

    if config.runtime_feedback:
        print(
            "[pipeline] "
            f"validation complete: {report.valid_files}/{report.total_files} valid",
            flush=True,
        )

    return records, report
