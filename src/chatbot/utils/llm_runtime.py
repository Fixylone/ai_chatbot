"""Runtime helpers for LLM throttling and retry behavior."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from chatbot.core.config import GenerationConfig


def _get_attr_or_key(container: Any, key: str) -> Any:
    """Read value from dict-like or attribute-like objects."""
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


def _extract_cached_tokens(response: object) -> int | None:
    """Extract provider-reported cached token count when available."""
    usage = _get_attr_or_key(response, "usage")
    if usage is None:
        return None

    prompt_details = _get_attr_or_key(usage, "prompt_tokens_details")
    if prompt_details is None:
        return None

    cached_tokens = _get_attr_or_key(prompt_details, "cached_tokens")
    if isinstance(cached_tokens, int):
        return cached_tokens
    if isinstance(cached_tokens, float):
        return int(cached_tokens)
    return None


def _is_retryable_error(exc: Exception) -> bool:
    """Return True when exception text indicates transient provider failures."""
    text = str(exc).lower()
    return (
        "rate limit" in text
        or "429" in text
        or "temporarily unavailable" in text
        or "timeout" in text
    )


async def call_llm_with_retries(
    call_fn: Callable[[str], object],
    prompt_text: str,
    config: GenerationConfig,
    *,
    stage: str,
    model_id: str,
) -> object:
    """Execute one LLM call with simple retry handling."""
    max_attempts = config.rate_limit_max_retries + 1

    for attempt in range(max_attempts):
        if config.runtime_feedback:
            print(
                "[llm] "
                f"stage={stage} "
                f"model={model_id} "
                f"attempt={attempt + 1}/{max_attempts}",
                flush=True,
            )

        try:
            response = await asyncio.to_thread(call_fn, prompt_text)
            if config.runtime_feedback:
                cached_tokens = _extract_cached_tokens(response)
                if cached_tokens is not None:
                    print(
                        "[cache] "
                        f"stage={stage} "
                        f"model={model_id} "
                        f"cached_tokens={cached_tokens}",
                        flush=True,
                    )
            return response
        except Exception as exc:
            if not _is_retryable_error(exc) or attempt == max_attempts - 1:
                raise

            delay = min(
                config.rate_limit_max_backoff_seconds,
                config.rate_limit_base_backoff_seconds * (2**attempt),
            )

            if config.runtime_feedback:
                first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                print(
                    "[retry] "
                    f"stage={stage} "
                    f"model={model_id} "
                    f"reason={first_line[:180]} "
                    f"sleep={delay:.1f}s",
                    flush=True,
                )

            await asyncio.sleep(delay)

    # Defensive fallback; loop always returns or raises.
    raise RuntimeError("Unexpected retry loop termination")
