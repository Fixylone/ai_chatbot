"""Centralized JSON utilities for generation and validation."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

_SLUG_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    """Convert free text to deterministic snake-case slug."""
    lowered = value.lower().strip().replace("&", " and ")
    collapsed = _SLUG_NON_ALNUM_PATTERN.sub("_", lowered)
    return collapsed.strip("_")


def build_toc_json_filename(document_type: str) -> str:
    """Build TOC JSON filename for one document type."""
    return f"toc_{_slugify(document_type)}.json"


def build_issue_manifest_filename(document_type: str) -> str:
    """Build per-document issues manifest filename."""
    return f"issues_{_slugify(document_type)}.json"


async def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload to disk asynchronously."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, ensure_ascii=True)
    await asyncio.to_thread(path.write_text, raw, encoding="utf-8")


def parse_json_content(content: str) -> tuple[Any | None, list[str]]:
    """Parse JSON content and return payload plus errors if any."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, [f"Invalid JSON syntax: {exc}"]

    return payload, []
