"""End-to-end generation pipeline orchestrator."""

from __future__ import annotations

import re

from chatbot.core.config import GenerationConfig
from chatbot.core.models import DocumentRecord, ToolDescription
from chatbot.services.ideation import generate_tool_ideation
from chatbot.services.section_generator import generate_document_sections
from chatbot.services.toc_generator import generate_table_of_contents
from chatbot.utils.html_utils import (
    assemble_document_html,
    build_html_filename,
    write_html_file,
)
from chatbot.utils.json_utils import (
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

    await write_json_file(toc_path, toc.model_dump())

    assembled_html = assemble_document_html(toc, sections)
    await write_html_file(html_path, assembled_html)

    issues_summary: list[str] = []
    for section in sections:
        issues_summary.extend(section.issues_applied)

    return DocumentRecord(
        tool_name=tool.name,
        document_type=document_type,
        html_path=str(html_path),
        toc_path=str(toc_path),
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

    ideation = await generate_tool_ideation(config)
    records: list[DocumentRecord] = []

    for tool in ideation.tools:
        for document_type in tool.assigned_doc_types:
            record = await _generate_single_document(
                config=config,
                tool=tool,
                document_type=document_type,
            )
            records.append(record)

    report = validate_all(config.output_dir)
    return records, report
