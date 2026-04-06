"""Unit tests for JSON utility functions."""

import asyncio
from pathlib import Path

from chatbot.utils.json_utils import (
    build_issue_manifest_filename,
    build_toc_json_filename,
    parse_json_content,
    write_json_file,
)


class TestJsonUtils:
    """Tests for JSON utility behavior."""

    def test_build_toc_json_filename_uses_snake_case(self) -> None:
        # Arrange
        document_type = "Privacy Policy"

        # Act
        filename = build_toc_json_filename(document_type)

        # Assert
        assert filename == "toc_privacy_policy.json"

    def test_build_issue_manifest_filename_uses_snake_case(self) -> None:
        # Arrange
        document_type = "Service Level Agreement"

        # Act
        filename = build_issue_manifest_filename(document_type)

        # Assert
        assert filename == "issues_service_level_agreement.json"

    def test_parse_json_content_parses_valid_json(self) -> None:
        # Arrange
        raw = '{"a": 1, "b": [1, 2]}'

        # Act
        payload, errors = parse_json_content(raw)

        # Assert
        assert payload == {"a": 1, "b": [1, 2]}
        assert errors == []

    def test_parse_json_content_returns_error_for_invalid_json(self) -> None:
        # Arrange
        raw = '{"a": 1'

        # Act
        payload, errors = parse_json_content(raw)

        # Assert
        assert payload is None
        assert len(errors) == 1
        assert "Invalid JSON syntax" in errors[0]

    def test_write_json_file_creates_parent_and_writes_json(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        output_path = tmp_path / "nested" / "payload.json"
        payload = {"name": "example", "count": 2}

        # Act
        asyncio.run(write_json_file(output_path, payload))

        # Assert
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert '"name": "example"' in content
        assert '"count": 2' in content
