"""Context compression for section-to-section continuity prompts.

Compresses previously generated sections into bullet summaries so the LLM
receives full document context without blowing up prompt size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chatbot.core.models import SectionOutput

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_HEADING_RE = re.compile(r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class CompressedSectionContext:
    previous_sections_summary: str
    last_full_section_html: str


class SectionContextCompressor:
    """Incrementally builds compressed context from accepted sections.

    Stores one bullet per section and always includes ALL of them in the
    summary (truncated only by a character limit as a safety net).
    """

    def __init__(
        self,
        max_summary_chars: int,
        max_last_section_chars: int,
    ) -> None:
        self._max_summary_chars = max_summary_chars
        self._max_last_section_chars = max_last_section_chars
        self._bullets: list[str] = []
        self._last_html: str = "none"

    def observe_section(self, section: SectionOutput) -> None:
        self._bullets.append(self._make_bullet(section))
        html = section.html_content.strip()
        if len(html) > self._max_last_section_chars:
            html = html[: self._max_last_section_chars].rstrip() + "\n<!-- truncated -->"
        self._last_html = html

    def current_context(self) -> CompressedSectionContext:
        if not self._bullets:
            return CompressedSectionContext("none", "none")

        summary = "\n".join(self._bullets)
        if len(summary) > self._max_summary_chars:
            summary = summary[: self._max_summary_chars].rstrip()

        return CompressedSectionContext(
            previous_sections_summary=summary,
            last_full_section_html=self._last_html,
        )

    def _make_bullet(self, section: SectionOutput) -> str:
        heading = self._extract_heading(section.html_content)
        if not heading:
            plain = self._strip_tags(section.html_content)
            heading = plain[:140].rstrip()
            if len(plain) > 140:
                heading += "..."
        return f"- {section.section_id}: {heading}"

    def _extract_heading(self, html: str) -> str:
        match = _HEADING_RE.search(html)
        if not match:
            return ""
        return self._strip_tags(match.group(1))[:120].strip()

    @staticmethod
    def _strip_tags(html: str) -> str:
        return _WS_RE.sub(" ", _TAG_RE.sub(" ", html)).strip()
