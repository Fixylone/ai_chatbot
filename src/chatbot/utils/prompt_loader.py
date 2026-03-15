"""Utilities for loading and rendering YAML prompt templates."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import yaml


def _load_prompt_template_sync(path: Path | str) -> str:
    """Load a prompt template from a YAML file.

    The YAML file must contain a top-level ``template`` key with string value.

    Args:
        path: Path to the YAML template file.

    Returns:
        Template string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If YAML shape is invalid.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {file_path}")

    with open(file_path, encoding="utf-8") as handle:
        raw: object = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Prompt template must be a mapping: {file_path}")

    raw_dict = cast(dict[str, Any], raw)

    template = raw_dict.get("template")
    if not isinstance(template, str):
        raise ValueError(
            f"Prompt template must include a string 'template' key: {file_path}"
        )

    return template


def _render_prompt_sync(
    path: Path | str,
    variables: dict[str, Any] | None = None,
) -> str:
    """Render a YAML prompt template using ``str.format`` variables.

    Args:
        path: Path to template YAML.
        variables: Template variables.

    Returns:
        Rendered prompt string.

    Raises:
        KeyError: If a required template variable is missing.
    """
    template = _load_prompt_template_sync(path)
    context = variables or {}

    try:
        return template.format(**context)
    except KeyError as exc:
        missing = exc.args[0] if exc.args else "<unknown>"
        raise KeyError(
            f"Missing prompt variable '{missing}' for template: {path}"
        ) from exc


async def render_prompt(
    path: Path | str,
    variables: dict[str, Any] | None = None,
) -> str:
    """Render a YAML prompt template asynchronously.

    Args:
        path: Path to template YAML.
        variables: Template variables.

    Returns:
        Rendered prompt string.
    """
    return await asyncio.to_thread(_render_prompt_sync, path, variables)
