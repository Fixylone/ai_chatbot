"""Application configuration via Pydantic Settings.

Priority: CLI overrides > env vars (CHATBOT_*) > config.yaml > field defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

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

    # LLM model identifiers
    toc_model: str = Field(default="openai/l2-gpt-4.1-nano", min_length=1)
    section_model: str = Field(default="openai/l2-gpt-4.1-mini", min_length=1)
    ideation_model: str = Field(default="openai/l2-gpt-4.1-nano", min_length=1)

    # Sampling
    toc_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    section_temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    ideation_temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

    # Scope
    num_tools: int = Field(default=5, ge=1, le=50)
    docs_per_tool: int = Field(default=4, ge=1, le=10)
    document_types: list[str] = Field(
        default_factory=lambda: list(ALL_DOCUMENT_TYPES),
    )

    # Retry controls
    rate_limit_max_retries: int = Field(default=3, ge=0, le=10)
    rate_limit_base_backoff_seconds: float = Field(default=2.0, ge=0.1, le=120.0)
    rate_limit_max_backoff_seconds: float = Field(default=60.0, ge=1.0, le=600.0)

    # RPM pacing — adaptive delay keeps request rate under this limit
    rpm_limit: int = Field(default=60, ge=1, le=600)
    # Minimum floor delay between section calls (seconds); adaptive pacer
    # may add more to stay within rpm_limit.
    section_delay_seconds: float = Field(default=0.0, ge=0.0, le=30.0)

    # Section context compression
    section_summary_max_chars: int = Field(default=6000, ge=200, le=20000)
    section_last_section_max_chars: int = Field(default=5000, ge=500, le=20000)

    # Output
    output_dir: Path = Field(default=Path("data"))

    @field_validator("output_dir", mode="after")
    @classmethod
    def _resolve_output_dir(cls, value: Path) -> Path:
        if not value.is_absolute():
            return (_PROJECT_ROOT / value).resolve()
        return value.resolve()

    @model_validator(mode="after")
    def _check_docs_fit_pool(self) -> GenerationConfig:
        if self.docs_per_tool > len(self.document_types):
            msg = (
                f"docs_per_tool ({self.docs_per_tool}) exceeds "
                f"document_types ({len(self.document_types)})"
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
    return GenerationConfig(**merged)
