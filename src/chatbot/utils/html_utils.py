"""Centralized HTML utilities for generation, persistence, and validation."""

from __future__ import annotations

import asyncio
import re
from html import escape
from pathlib import Path
from typing import Any, cast

from lxml import etree as _etree  # type: ignore[import-untyped]

from chatbot.core.models import SectionOutput, TableOfContents
from chatbot.utils import slugify

etree = cast(Any, _etree)
_HTML_TAG_PATTERN = re.compile(r"<\s*[a-zA-Z][^>]*>")
_TAG_INVALID_RE = re.compile(r"Tag (\w+) invalid")
_HTML5_TAGS = frozenset({
    "article", "aside", "details", "figcaption", "figure", "footer",
    "header", "main", "mark", "nav", "section", "summary", "time",
})


def build_html_filename(document_type: str) -> str:
    """Convert document type into deterministic HTML filename."""
    return f"{slugify(document_type)}.html"


def assemble_document_html(
    toc: TableOfContents,
    sections: list[SectionOutput],
) -> str:
    """Assemble a full HTML document from generated section fragments."""
    title = f"{toc.tool_name} - {toc.document_type}"
    body_content = "\n\n".join(section.html_content for section in sections)

    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        f"  <title>{escape(title)}</title>\n"
        "  <style>\n"
        "    body { font-family: Georgia, serif; line-height: 1.55; "
        "margin: 2rem auto; max-width: 960px; padding: 0 1rem; }\n"
        "    h1, h2, h3, h4 { line-height: 1.25; }\n"
        "    p, li { color: #1a1a1a; }\n"
        "    .doc-meta { color: #555; margin-bottom: 1rem; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>{escape(toc.document_type)}</h1>\n"
        f"  <p class=\"doc-meta\"><strong>Tool:</strong> {escape(toc.tool_name)}</p>\n"
        f"{body_content}\n"
        "</body>\n"
        "</html>\n"
    )


async def write_html_file(path: Path, html_content: str) -> None:
    """Write HTML content to disk asynchronously."""
    path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, html_content, encoding="utf-8")


def validate_html_document(content: str) -> list[str]:
    """Validate HTML parseability and return parser errors."""
    parser = etree.HTMLParser(recover=True)

    try:
        etree.fromstring(content.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError as exc:
        return [str(exc)]

    errors: list[str] = []
    for entry in parser.error_log:
        msg = str(entry)
        m = _TAG_INVALID_RE.search(msg)
        if m and m.group(1).lower() in _HTML5_TAGS:
            continue
        errors.append(msg)
    return errors


def is_valid_html_fragment(content: str) -> bool:
    """Check whether content is an HTML fragment parseable when wrapped.

    This lightweight gate is intentionally more tolerant than final output
    validation, which runs on full assembled documents.
    """
    if not _HTML_TAG_PATTERN.search(content):
        return False

    wrapped = f"<!DOCTYPE html><html><body>{content}</body></html>"
    parser = etree.HTMLParser(recover=True)
    try:
        etree.fromstring(wrapped.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError:
        return False

    return True
