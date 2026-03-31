"""LLM call retry logic with exponential backoff."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from chatbot.core.config import GenerationConfig


def _is_retryable(exc: Exception) -> bool:
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
    """Execute an LLM call with exponential-backoff retry on transient errors."""
    max_attempts = config.rate_limit_max_retries + 1

    for attempt in range(max_attempts):
        print(
            f"[llm] stage={stage} model={model_id} "
            f"attempt={attempt + 1}/{max_attempts}",
            flush=True,
        )
        try:
            return await asyncio.to_thread(call_fn, prompt_text)
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_attempts - 1:
                raise

            delay = min(
                config.rate_limit_max_backoff_seconds,
                config.rate_limit_base_backoff_seconds * (2**attempt),
            )
            reason = str(exc).splitlines()[0][:180] if str(exc) else type(exc).__name__
            print(
                f"[retry] stage={stage} reason={reason} sleep={delay:.1f}s",
                flush=True,
            )
            await asyncio.sleep(delay)

    raise RuntimeError("Unexpected retry loop exit")
