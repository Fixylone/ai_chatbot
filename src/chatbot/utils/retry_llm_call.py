"""Shared retry helper for LLM calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError
except Exception:  # pragma: no cover - optional dependency safety
    APIConnectionError = Exception
    APITimeoutError = Exception
    RateLimitError = Exception


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True

    text = str(exc).lower()
    return (
        "rate limit" in text
        or "429" in text
        or "temporarily unavailable" in text
        or "timeout" in text
    )


async def retry_llm_call[T](
    call_fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
    request_timeout_seconds: float | None = None,
) -> T:
    """Run an async LLM call with bounded exponential backoff retries."""
    max_attempts = max_retries + 1

    for attempt in range(max_attempts):
        try:
            if request_timeout_seconds is None:
                return await call_fn()
            return await asyncio.wait_for(call_fn(), timeout=request_timeout_seconds)
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_attempts - 1:
                raise

            delay = min(
                max_backoff_seconds,
                base_backoff_seconds * (2**attempt),
            )
            await asyncio.sleep(delay)

    raise RuntimeError("Unexpected retry loop exit")
