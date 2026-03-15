"""Step 7: HTML assembly service."""

from __future__ import annotations

import asyncio
from html import escape
from pathlib import Path

from chatbot.core.models import SectionOutput, TableOfContents


def assemble_document_html(
    toc: TableOfContents,
    sections: list[SectionOutput],
) -> str:
    """Assemble a full HTML document from generated section fragments.

    Args:
        toc: Table of contents metadata for the document.
        sections: Ordered section outputs.

    Returns:
        Full HTML document string.
    """
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


async def write_document_html(path: Path, html_content: str) -> None:
    """Write assembled HTML to disk asynchronously.

    Args:
        path: Destination file path.
        html_content: Full HTML content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, html_content, encoding="utf-8")


def build_html_filename(document_type: str) -> str:
    """Convert document type into deterministic filename.

    Args:
        document_type: Human-readable document type.

    Returns:
        Snake-case filename ending with ``.html``.
    """
    normalized = "_".join(document_type.lower().replace("&", "and").split())
    return f"{normalized}.html"
