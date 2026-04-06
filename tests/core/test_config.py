"""Unit tests for configuration loading and validation."""

from pathlib import Path

import pytest

from chatbot.core.config import GenerationConfig, load_config


class TestGenerationConfig:
    """Tests for GenerationConfig validation and normalization."""

    def test_resolves_relative_output_dir_to_absolute_path(self) -> None:
        # Arrange
        relative_output = Path("tmp_data")

        # Act
        cfg = GenerationConfig(
            output_dir=relative_output,
            docs_per_tool=1,
            document_types=["Privacy Policy"],
        )

        # Assert
        assert cfg.output_dir.is_absolute()
        assert cfg.output_dir.name == "tmp_data"

    def test_rejects_docs_per_tool_larger_than_document_pool(self) -> None:
        # Arrange
        kwargs = {
            "docs_per_tool": 3,
            "document_types": ["Privacy Policy", "Terms of Service"],
        }

        # Act / Assert
        with pytest.raises(ValueError, match="docs_per_tool"):
            GenerationConfig(**kwargs)


class TestLoadConfig:
    """Tests for YAML + override config assembly."""

    def test_returns_defaults_when_config_file_is_missing(self, tmp_path: Path) -> None:
        # Arrange
        missing_path = tmp_path / "missing.yaml"

        # Act
        cfg = load_config(config_path=missing_path)

        # Assert
        assert isinstance(cfg, GenerationConfig)
        assert cfg.num_tools == 5
        assert cfg.docs_per_tool == 4

    def test_overrides_take_precedence_over_yaml_values(self, tmp_path: Path) -> None:
        # Arrange
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "num_tools: 2",
                    "docs_per_tool: 1",
                    "document_types:",
                    "  - Privacy Policy",
                    'output_dir: "data"',
                ]
            ),
            encoding="utf-8",
        )

        # Act
        cfg = load_config(config_path=cfg_path, overrides={"num_tools": 7})

        # Assert
        assert cfg.num_tools == 7
        assert cfg.docs_per_tool == 1
        assert cfg.document_types == ["Privacy Policy"]
