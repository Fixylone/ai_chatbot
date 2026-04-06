"""Unit tests for HTML utility functions."""

import asyncio
from pathlib import Path

from chatbot.core.models import SectionOutput, TableOfContents
from chatbot.utils.html_utils import (
    assemble_document_html,
    build_html_filename,
    is_valid_html_fragment,
    validate_html_document,
    write_html_file,
)


class TestHtmlUtils:
    """Tests for HTML utility behavior."""

    def test_build_html_filename_uses_snake_case(self) -> None:
        # Arrange
        document_type = "Compliance & Certifications"

        # Act
        filename = build_html_filename(document_type)

        # Assert
        assert filename == "compliance_and_certifications.html"

    def test_assemble_document_html_escapes_metadata_and_includes_sections(
        self,
    ) -> None:
        # Arrange
        toc = TableOfContents(
            document_type="Privacy <Policy>",
            tool_name="Tool & Co",
            sections=[],
        )
        sections = [
            SectionOutput(
                section_id="1",
                html_content="<section><h2>1. Intro</h2><p>Hello</p></section>",
            )
        ]

        # Act
        html = assemble_document_html(toc, sections)

        # Assert
        assert "<!DOCTYPE html>" in html
        assert "<title>Tool &amp; Co - Privacy &lt;Policy&gt;</title>" in html
        assert "<h2>1. Intro</h2>" in html

    def test_validate_html_document_returns_no_errors_for_valid_html(self) -> None:
        # Arrange
        valid_html = "<!DOCTYPE html><html><body><p>ok</p></body></html>"

        # Act
        errors = validate_html_document(valid_html)

        # Assert
        assert errors == []

    def test_is_valid_html_fragment_rejects_plain_text_and_accepts_tags(self) -> None:
        # Arrange
        plain = "just plain text"
        fragment = "<p>hello</p>"

        # Act
        plain_valid = is_valid_html_fragment(plain)
        fragment_valid = is_valid_html_fragment(fragment)

        # Assert
        assert plain_valid is False
        assert fragment_valid is True

    def test_write_html_file_creates_directory_and_writes_content(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        output_path = tmp_path / "nested" / "doc.html"
        content = "<p>saved</p>"

        # Act
        asyncio.run(write_html_file(output_path, content))

        # Assert
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == content
