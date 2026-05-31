"""Configuration loading for the console chatbot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SUMMARY_KEYWORDS = ["summary", "summarize", "recap", "overview"]
_DEFAULT_EXIT_COMMANDS = ["exit", "quit", "/exit"]


class ChatConfig(BaseSettings):
    """Central configuration for the console chatbot.

    Priority order:
    1. Explicit CLI overrides
    2. Environment variables (CHATBOT_CHAT_*)
    3. YAML file values
    4. Field defaults

    Attributes:
        chat_model: Model ID for assistant responses.
        chat_temperature: Sampling temperature.
        chat_top_p: Nucleus sampling parameter.
        max_history_messages: Maximum non-system messages retained in memory.
        system_prompt_path: YAML prompt template path for system behavior.
        enable_get_current_date: Enable current-date tool.
        enable_add_days_to_date: Enable date-offset tool.
        summary_keywords: Keywords that trigger explicit summary mode.
        exit_commands: Console commands that terminate the session.
        api_base_url: Optional OpenAI-compatible base URL.
        api_key: Optional API key override.
        request_timeout_seconds: Timeout for one model request in seconds.
        rate_limit_max_retries: Retry attempts for transient API errors.
        rate_limit_base_backoff_seconds: Base backoff duration.
        rate_limit_max_backoff_seconds: Maximum backoff duration.
    """

    model_config = SettingsConfigDict(
        env_prefix="CHATBOT_CHAT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chat_model: str = Field(default="openai/l2-gpt-4.1-mini", min_length=1)
    chat_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    chat_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_history_messages: int = Field(default=24, ge=6, le=200)
    system_prompt_path: Path = Field(
        default=Path("src/chatbot_console/prompts/chat_system.yaml")
    )

    enable_get_current_date: bool = True
    enable_add_days_to_date: bool = True

    summary_keywords: list[str] = Field(
        default_factory=lambda: _DEFAULT_SUMMARY_KEYWORDS
    )
    exit_commands: list[str] = Field(default_factory=lambda: _DEFAULT_EXIT_COMMANDS)

    api_base_url: str | None = None
    api_key: str | None = None
    request_timeout_seconds: float = Field(default=30.0, ge=5.0, le=300.0)

    rate_limit_max_retries: int = Field(default=0, ge=0, le=10)
    rate_limit_base_backoff_seconds: float = Field(default=1.5, ge=0.1, le=120.0)
    rate_limit_max_backoff_seconds: float = Field(default=30.0, ge=1.0, le=600.0)

    @field_validator("system_prompt_path", mode="after")
    @classmethod
    def _resolve_system_prompt_path(cls, value: Path) -> Path:
        """Resolve system prompt path relative to the project root.

        Args:
            value: Configured path.

        Returns:
            Absolute normalized path.
        """
        if not value.is_absolute():
            return (_PROJECT_ROOT / value).resolve()
        return value.resolve()

    @field_validator("summary_keywords", "exit_commands", mode="after")
    @classmethod
    def _normalize_text_lists(cls, value: list[str]) -> list[str]:
        """Normalize keyword-like list fields for stable matching.

        Args:
            value: Raw configured string list.

        Returns:
            Deduplicated, lowercase, trimmed values.
        """
        seen: set[str] = set()
        normalized: list[str] = []
        for item in value:
            clean = item.strip().lower()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
        return normalized

    @model_validator(mode="after")
    def _validate_tool_toggle(self) -> ChatConfig:
        """Ensure at least one tool stays enabled.

        Returns:
            Validated config instance.

        Raises:
            ValueError: If all tools are disabled.
        """
        if not self.enable_get_current_date and not self.enable_add_days_to_date:
            raise ValueError("At least one tool must be enabled.")
        return self


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Read YAML config values from disk.

    Args:
        path: Path to YAML config file.

    Returns:
        Mapping of config values or empty dict when unavailable/invalid.
    """
    resolved = path if path.is_absolute() else _PROJECT_ROOT / path
    if not resolved.exists():
        return {}

    with open(resolved, encoding="utf-8") as handle:
        raw: object = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        return {}

    return cast(dict[str, Any], raw)


def load_chat_config(
    config_path: Path | str = "chat_config.yaml",
    overrides: dict[str, Any] | None = None,
) -> ChatConfig:
    """Build validated ChatConfig from YAML and explicit overrides.

    Args:
        config_path: YAML config path.
        overrides: Explicit CLI-level overrides.

    Returns:
        Validated chat configuration.
    """
    yaml_values = _load_yaml_config(Path(config_path))
    merged: dict[str, Any] = {**yaml_values, **(overrides or {})}
    return ChatConfig(**merged)
