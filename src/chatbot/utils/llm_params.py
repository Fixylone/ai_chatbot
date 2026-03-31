"""Helpers for LLM call parameter selection in generation services."""

from __future__ import annotations

import hashlib
import re
from typing import Any

_KEY_SANITIZE_PATTERN = re.compile(r"[^a-z0-9_.:-]+")


def is_prompt_cache_param_error(exc: Exception) -> bool:
    """Detect provider errors caused by prompt-cache request params.

    Args:
        exc: Raised provider exception.

    Returns:
        True when error message indicates unsupported cache parameters.
    """
    text = str(exc).lower()
    return (
        "prompt_cache" in text
        or "prompt cache" in text
        or "unknown parameter" in text and "cache" in text
        or "unsupported" in text and "cache" in text
    )


def build_prompt_cache_key(namespace: str, parts: list[str]) -> str:
    """Build a stable cache key from namespace and parts.

    Args:
        namespace: Top-level key namespace.
        parts: Logical key components.

    Returns:
        A compact cache key suitable for provider APIs.
    """
    joined_parts = "|".join(part.strip().lower() for part in parts if part.strip())
    digest = hashlib.sha1(joined_parts.encode("utf-8")).hexdigest()[:16]
    cleaned_namespace = _KEY_SANITIZE_PATTERN.sub("-", namespace.lower()).strip("-")
    safe_namespace = cleaned_namespace or "chatbot"
    return f"{safe_namespace}:{digest}"


def build_llm_call_kwargs(
    temperature: float,
    top_p: float,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
) -> dict[str, Any]:
    """Build kwargs for Mirascope ``llm.call``.

    Args:
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        prompt_cache_key: Optional provider cache key.
        prompt_cache_retention: Optional provider cache retention policy.

    Returns:
        Decorator kwargs for one ``llm.call``.
    """
    call_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
    }
    if prompt_cache_key:
        call_kwargs["prompt_cache_key"] = prompt_cache_key
    if prompt_cache_retention:
        call_kwargs["prompt_cache_retention"] = prompt_cache_retention
    return call_kwargs
