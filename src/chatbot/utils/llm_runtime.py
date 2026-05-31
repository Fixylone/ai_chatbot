"""LLM call retry logic with exponential backoff."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from chatbot.core.config import GenerationConfig
from chatbot.utils.retry_llm_call import retry_llm_call as _shared_retry_llm_call


async def retry_llm_call(
    call_fn: Callable[[str], object],
    prompt_text: str,
    config: GenerationConfig,
) -> object:
    """Execute a sync LLM call with retry policy from config."""

    async def _invoke() -> object:
        return await asyncio.to_thread(call_fn, prompt_text)

    return await _shared_retry_llm_call(
        _invoke,
        max_retries=config.rate_limit_max_retries,
        base_backoff_seconds=config.rate_limit_base_backoff_seconds,
        max_backoff_seconds=config.rate_limit_max_backoff_seconds,
        request_timeout_seconds=None,
    )
