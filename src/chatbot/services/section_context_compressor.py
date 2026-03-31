"""Context compression for section-to-section continuity prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from chatbot.core.models import SectionOutput

_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_HEADING_PATTERN = re.compile(r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class CompressedSectionContext:
    """Compressed continuity context for a section prompt.

    Attributes:
        previous_sections_summary: Bullet summary for older generated sections.
        last_full_section_html: Full HTML for the most recent generated section.
    """

    previous_sections_summary: str
    last_full_section_html: str


class SectionContextCompressor:
    """Build compact continuity context for section generation prompts.

    The compressor stores per-section bullets incrementally and appends one
    bullet each time a new section is accepted.
    """

    def __init__(
        self,
        max_summary_sections: int,
        max_summary_chars: int,
        max_last_section_chars: int,
    ) -> None:
        """Create a compressor with configurable size limits.

        Args:
            max_summary_sections: Max number of older sections to summarize.
            max_summary_chars: Max characters allowed in summary block.
            max_last_section_chars: Max characters allowed for last section HTML.
        """
        self._max_summary_sections = max_summary_sections
        self._max_summary_chars = max_summary_chars
        self._max_last_section_chars = max_last_section_chars
        self._summary_bullets: list[str] = []
        self._last_full_section_html: str = "none"

    def observe_section(self, section: SectionOutput) -> None:
        """Append summary state for one newly accepted generated section."""
        self._summary_bullets.append(self._build_bullet(section))
        last_section_html = section.html_content.strip()
        if len(last_section_html) > self._max_last_section_chars:
            last_section_html = (
                f"{last_section_html[: self._max_last_section_chars].rstrip()}\n"
                "<!-- truncated -->"
            )
        self._last_full_section_html = last_section_html

    def current_context(self) -> CompressedSectionContext:
        """Return prompt context from currently accumulated summary state."""
        if not self._summary_bullets:
            return CompressedSectionContext(
                previous_sections_summary="none",
                last_full_section_html="none",
            )

        summary_lines = self._summary_bullets[-self._max_summary_sections :]
        summary = "\n".join(summary_lines)
        if len(summary) > self._max_summary_chars:
            summary = summary[: self._max_summary_chars].rstrip()

        return CompressedSectionContext(
            previous_sections_summary=summary,
            last_full_section_html=self._last_full_section_html,
        )

    def _build_bullet(self, section: SectionOutput) -> str:
        """Build one summary bullet from a section."""
        heading = self._extract_heading(section.html_content)
        plain_text = self._to_plain_text(section.html_content)
        snippet = plain_text[:140].rstrip()
        if len(plain_text) > 140:
            snippet = f"{snippet}..."
        summary_text = heading if heading else snippet
        return f"- {section.section_id}: {summary_text}"

    def _extract_heading(self, html_content: str) -> str:
        """Extract the first heading text from section HTML."""
        match = _HEADING_PATTERN.search(html_content)
        if not match:
            return ""
        heading_text = self._to_plain_text(match.group(1))
        return heading_text[:120].strip()

    def _to_plain_text(self, html_content: str) -> str:
        """Convert HTML to normalized plain text."""
        without_tags = _TAG_PATTERN.sub(" ", html_content)
        collapsed = _WHITESPACE_PATTERN.sub(" ", without_tags)
        return collapsed.strip()
