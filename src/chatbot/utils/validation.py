"""HTML and JSON validation utilities."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from chatbot.core.models import FileValidationResult, TableOfContents, ValidationReport
from chatbot.utils.html_utils import validate_html_document
from chatbot.utils.json_utils import parse_json_content


def validate_html(content: str) -> list[str]:
    """Validate HTML parseability and return parser errors.

    Args:
        content: HTML string to validate.

    Returns:
        List of validation error messages. Empty means valid.
    """
    return validate_html_document(content)


def validate_toc_json_content(content: str) -> list[str]:
    """Validate TOC JSON content against the Pydantic schema.

    Args:
        content: JSON content string.

    Returns:
        List of errors. Empty means valid.
    """
    payload, parse_errors = parse_json_content(content)
    if parse_errors:
        return parse_errors

    try:
        TableOfContents.model_validate(payload)
    except ValidationError as exc:
        return [f"Schema validation failed: {exc}"]

    return []


def validate_file(path: Path) -> FileValidationResult:
    """Validate a single HTML or JSON file.

    Args:
        path: File path.

    Returns:
        Validation result object.
    """
    content = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".html":
        errors = validate_html(content)
        return FileValidationResult(
            path=str(path),
            file_type="html",
            is_valid=not errors,
            errors=errors,
        )

    if path.suffix.lower() == ".json":
        errors = validate_toc_json_content(content)
        return FileValidationResult(
            path=str(path),
            file_type="json",
            is_valid=not errors,
            errors=errors,
        )

    return FileValidationResult(
        path=str(path),
        file_type="unknown",
        is_valid=False,
        errors=["Unsupported file extension"],
    )


def validate_all(output_dir: Path) -> ValidationReport:
    """Validate all generated HTML and TOC JSON artifacts.

    Args:
        output_dir: Root output directory.

    Returns:
        Aggregated validation report.
    """
    html_paths = sorted(output_dir.rglob("*.html"))
    json_paths = sorted(output_dir.rglob("toc_*.json"))

    results: list[FileValidationResult] = [
        validate_file(path) for path in [*html_paths, *json_paths]
    ]

    valid_files = sum(1 for result in results if result.is_valid)
    total_files = len(results)

    return ValidationReport(
        total_files=total_files,
        valid_files=valid_files,
        invalid_files=total_files - valid_files,
        html_files=len(html_paths),
        json_files=len(json_paths),
        results=results,
    )
