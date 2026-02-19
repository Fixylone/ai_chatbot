"""LLM wrapper using the OpenAI chat-completion API via LangChain.

This module provides a concrete implementation of :class:`BaseLLM` backed by
any OpenAI-compatible model.  Swap the underlying provider by changing the
``llm_model`` setting without touching the rest of the codebase.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from chatbot.core.config import settings
from chatbot.core.interfaces import BaseLLM, Message
from chatbot.utils.logger import get_logger

logger = get_logger(__name__)

# Map our generic role strings to LangChain message classes.
_ROLE_MAP = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
}


class OpenAILLM(BaseLLM):
    """LangChain-backed OpenAI chat-completion wrapper."""

    def __init__(self) -> None:
        self._client = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )
        logger.info("llm_initialized", model=settings.llm_model)

    def generate(self, messages: list[Message]) -> str:
        """Send *messages* to the LLM and return the text reply."""
        lc_messages = [
            _ROLE_MAP.get(m.role, HumanMessage)(content=m.content) for m in messages
        ]
        response = self._client.invoke(lc_messages)
        content = response.content
        logger.debug("llm_response_received", chars=len(str(content)))
        return str(content)
