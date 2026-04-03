"""Centralized JSON utilities for generation and validation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from chatbot.utils import to_snake_case


def build_toc_json_filename(document_type: str) -> str:
    """Build TOC JSON filename for one document type."""
    return f"toc_{to_snake_case(document_type)}.json"


def build_issue_manifest_filename(document_type: str) -> str:
    """Build per-document issues manifest filename."""
    return f"issues_{to_snake_case(document_type)}.json"


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
