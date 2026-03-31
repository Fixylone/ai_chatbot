"""Shared utility functions."""

import re

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Convert free text to a deterministic snake_case slug."""
    lowered = value.lower().strip().replace("&", " and ")
    collapsed = _SLUG_PATTERN.sub("_", lowered)
    return collapsed.strip("_")
