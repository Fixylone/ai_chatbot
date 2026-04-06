"""Unit tests for artifact validation helpers."""

import json
from pathlib import Path

from chatbot.utils.validation import (
    validate_all,
    validate_file,
    validate_issues_json_content,
    validate_toc_json_content,
)


class TestValidationFunctions:
    """Tests for validation utility behavior."""

    @staticmethod
    def _valid_toc_payload() -> dict[str, object]:
        return {
            "document_type": "Privacy Policy",
            "tool_name": "VaultSync",
            "sections": [
                {
                    "id": "1",
                    "title": "Introduction",
                    "children": [],
                }
            ],
        }

    @staticmethod
    def _valid_issues_payload() -> dict[str, object]:
        return {
            "tool_name": "VaultSync",
            "document_type": "Privacy Policy",
            "total_issues": 0,
            "sections": [
                {
                    "section_id": "1",
                    "issues_applied": [],
                    "issue_sentence": "none",
                }
            ],
        }

    def test_validate_toc_json_content_accepts_valid_payload(self) -> None:
        # Arrange
        raw = json.dumps(self._valid_toc_payload())

        # Act
        errors = validate_toc_json_content(raw)

        # Assert
        assert errors == []

    def test_validate_toc_json_content_rejects_invalid_schema(self) -> None:
        # Arrange
        raw = json.dumps({"tool_name": "VaultSync"})

        # Act
        errors = validate_toc_json_content(raw)

        # Assert
        assert len(errors) == 1
        assert "Schema validation failed" in errors[0]

    def test_validate_issues_json_content_accepts_valid_payload(self) -> None:
        # Arrange
        raw = json.dumps(self._valid_issues_payload())

        # Act
        errors = validate_issues_json_content(raw)

        # Assert
        assert errors == []

    def test_validate_file_rejects_unsupported_extension(self, tmp_path: Path) -> None:
        # Arrange
        unknown = tmp_path / "artifact.txt"
        unknown.write_text("content", encoding="utf-8")

        # Act
        result = validate_file(unknown)

        # Assert
        assert result.is_valid is False
        assert result.file_type == "unknown"
        assert result.errors == ["Unsupported file extension"]

    def test_validate_all_counts_html_and_json_files(self, tmp_path: Path) -> None:
        # Arrange
        html_path = tmp_path / "doc.html"
        toc_path = tmp_path / "toc_privacy_policy.json"
        issues_path = tmp_path / "issues_privacy_policy.json"

        html_path.write_text(
            "<!DOCTYPE html><html><body><p>ok</p></body></html>",
            encoding="utf-8",
        )
        toc_path.write_text(
            json.dumps(self._valid_toc_payload()),
            encoding="utf-8",
        )
        issues_path.write_text(
            json.dumps(self._valid_issues_payload()),
            encoding="utf-8",
        )

        # Act
        report = validate_all(tmp_path)

        # Assert
        assert report.total_files == 3
        assert report.valid_files == 3
        assert report.invalid_files == 0
        assert report.html_files == 1
        assert report.json_files == 2
