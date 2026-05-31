"""Helpers for detecting summary requests in user input."""

from __future__ import annotations

import re

_SUMMARY_PATTERN = re.compile(r"\b(summary|summarize|recap|overview)\b", re.IGNORECASE)


def is_summary_request(user_input: str, keywords: list[str]) -> bool:
    """Return True when user input asks for a summary."""
    lowered = user_input.strip().lower()
    if not lowered:
        return False

    if _SUMMARY_PATTERN.search(lowered):
        return True

    return any(keyword in lowered for keyword in keywords)
