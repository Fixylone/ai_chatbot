"""Application configuration using Pydantic Settings.

Loads settings from (highest priority first):
    1. Explicit overrides dict (CLI flags)
    2. Environment variables (``CHATBOT_*`` prefix)
    3. ``config.yaml`` at project root
    4. Field defaults below
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ALL_DOCUMENT_TYPES: list[str] = [
    "Privacy Policy",
    "Terms of Service",
    "Data Processing Agreement",
    "Service Level Agreement",
    "Security Whitepaper",
    "Compliance & Certifications",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class GenerationConfig(BaseSettings):
    """Central configuration for the data-generation pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="CHATBOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM model identifiers (provider-specific ID passed to mirascope/litellm)
    toc_model: str = Field(default="openai/l2-gpt-4.1-mini", min_length=1)
    section_model: str = Field(default="openai/l2-gpt-4o-mini", min_length=1)
    ideation_model: str = Field(default="openai/l2-gpt-4.1-nano", min_length=1)

    # Sampling parameters
    toc_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    section_temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    ideation_temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

    # Generation scope
    num_tools: int = Field(default=5, ge=1, le=50)
    docs_per_tool: int = Field(default=4, ge=1, le=10)
    document_types: list[str] = Field(
        default_factory=lambda: list(ALL_DOCUMENT_TYPES),
    )

    # Runtime retry controls
    rate_limit_max_retries: int = Field(default=3, ge=0, le=10)
    rate_limit_base_backoff_seconds: float = Field(default=2.0, ge=0.1, le=120.0)
    rate_limit_max_backoff_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    runtime_feedback: bool = Field(default=True)

    # Optional prompt caching controls
    prompt_caching_enabled: bool = Field(default=False)
    prompt_cache_retention: Literal["in_memory", "24h"] = Field(
        default="in_memory"
    )
    prompt_cache_key_namespace: str = Field(default="chatbot", min_length=1)

    # Section context compression controls
    section_summary_max_sections: int = Field(default=8, ge=1, le=50)
    section_summary_max_chars: int = Field(default=1600, ge=200, le=10000)
    section_last_section_max_chars: int = Field(default=5000, ge=500, le=20000)

    # Output path (resolved relative to project root)
    output_dir: Path = Field(default=Path("data"))

    # API key — read from OPENAI_API_KEY env var (no CHATBOT_ prefix)
    openai_api_key: str = Field(default="")

    # -- Validators -----------------------------------------------------------

    @field_validator("document_types", mode="before")
    @classmethod
    def _coerce_document_types(cls, value: Any) -> list[str]:
        """Accept a comma-separated string or a list."""
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return list(value)

    @field_validator("output_dir", mode="after")
    @classmethod
    def _resolve_output_dir(cls, value: Path) -> Path:
        """Resolve relative paths from the project root."""
        if not value.is_absolute():
            return (_PROJECT_ROOT / value).resolve()
        return value.resolve()

    @model_validator(mode="after")
    def _check_docs_fit_pool(self) -> GenerationConfig:
        """Ensure docs_per_tool does not exceed available types."""
        if self.docs_per_tool > len(self.document_types):
            msg = (
                f"docs_per_tool ({self.docs_per_tool}) exceeds "
                f"document_types ({len(self.document_types)})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_model_diversity(self) -> GenerationConfig:
        """Require at least two distinct generation models."""
        model_ids = {
            self.toc_model.strip(),
            self.section_model.strip(),
            self.ideation_model.strip(),
        }
        if len(model_ids) < 2:
            msg = (
                "At least two distinct models are required across "
                "toc_model, section_model, and ideation_model."
            )
            raise ValueError(msg)
        return self


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Read a YAML config file; return empty dict if missing."""
    resolved = path if path.is_absolute() else _PROJECT_ROOT / path
    if not resolved.exists():
        return {}
    with open(resolved, encoding="utf-8") as fh:
        raw: object = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        return {}
    return cast(dict[str, Any], raw)


def load_config(
    config_path: Path | str = "config.yaml",
    overrides: dict[str, Any] | None = None,
) -> GenerationConfig:
    """Build a validated ``GenerationConfig`` from YAML + env + overrides."""
    yaml_values = _load_yaml_config(Path(config_path))
    merged: dict[str, Any] = {**yaml_values, **(overrides or {})}

    if "openai_api_key" not in merged:
        env_key = os.environ.get("OPENAI_API_KEY", "")
        if env_key:
            merged["openai_api_key"] = env_key

    return GenerationConfig(**merged)
