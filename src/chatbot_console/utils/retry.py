"""Async retry helper with exponential backoff for LLM calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from chatbot.utils.retry_llm_call import retry_llm_call as _shared_retry_llm_call
from chatbot_console.core.config import ChatConfig


async def retry_llm_call[T](
    call_fn: Callable[[], Awaitable[T]],
    config: ChatConfig,
) -> T:
    """Execute async LLM call with retry policy from config."""
    return await _shared_retry_llm_call(
        call_fn,
        max_retries=config.rate_limit_max_retries,
        base_backoff_seconds=config.rate_limit_base_backoff_seconds,
        max_backoff_seconds=config.rate_limit_max_backoff_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
    )
