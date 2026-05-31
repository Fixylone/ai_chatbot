"""Unit tests for chat configuration loading and validation."""

from pathlib import Path

import pytest

from chatbot_console.core.config import ChatConfig, load_chat_config


class TestChatConfig:
    """Tests for ChatConfig validation and normalization behavior."""

    def test_resolves_relative_prompt_path_to_absolute(self) -> None:
        """Relative system prompt path should be normalized to absolute."""
        cfg = ChatConfig(
            system_prompt_path=Path("src/chatbot_console/prompts/chat_system.yaml")
        )

        assert cfg.system_prompt_path.is_absolute()
        assert cfg.system_prompt_path.name == "chat_system.yaml"

    def test_rejects_when_all_tools_are_disabled(self) -> None:
        """Config should fail when no tools are enabled."""
        with pytest.raises(ValueError, match="At least one tool"):
            ChatConfig(
                enable_get_current_date=False,
                enable_add_days_to_date=False,
            )


class TestLoadChatConfig:
    """Tests for YAML + override configuration assembly."""

    def test_returns_defaults_when_file_is_missing(self, tmp_path: Path) -> None:
        """Missing YAML should fall back to default config values."""
        missing_path = tmp_path / "missing.yaml"

        cfg = load_chat_config(config_path=missing_path)

        assert isinstance(cfg, ChatConfig)
        assert cfg.chat_model == "openai/l2-gpt-4.1-mini"
        assert cfg.max_history_messages == 24
        assert cfg.request_timeout_seconds == 30.0

    def test_overrides_take_precedence_over_yaml_values(self, tmp_path: Path) -> None:
        """Explicit overrides should win over YAML values."""
        cfg_path = tmp_path / "chat_config.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    'chat_model: "openai/example-model"',
                    "chat_temperature: 0.2",
                    "max_history_messages: 12",
                    "request_timeout_seconds: 45",
                ]
            ),
            encoding="utf-8",
        )

        cfg = load_chat_config(
            config_path=cfg_path,
            overrides={"chat_model": "openai/override-model"},
        )

        assert cfg.chat_model == "openai/override-model"
        assert cfg.chat_temperature == 0.2
        assert cfg.max_history_messages == 12
        assert cfg.request_timeout_seconds == 45
