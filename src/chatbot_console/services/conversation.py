"""In-memory conversation state manager for one console session."""

from __future__ import annotations

from chatbot_console.core.models import ChatMessage, MessageRole


class ConversationManager:
    """Manage role-aware chat history with deterministic truncation.

    Args:
        system_prompt: Initial system instruction.
        max_history_messages: Maximum non-system messages retained.
    """

    def __init__(self, system_prompt: str, max_history_messages: int) -> None:
        self._max_history_messages = max_history_messages
        self._messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
        ]

    def add_user_message(self, content: str) -> None:
        """Append user message and enforce history limit.

        Args:
            content: User input text.
        """
        self._messages.append(ChatMessage(role=MessageRole.USER, content=content))
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """Append assistant message and enforce history limit.

        Args:
            content: Assistant response text.
        """
        self._messages.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=content)
        )
        self._trim_history()

    def add_tool_message(
        self,
        *,
        content: str,
        name: str,
        tool_call_id: str,
    ) -> None:
        """Append tool output and enforce history limit.

        Args:
            content: Tool output payload.
            name: Tool name.
            tool_call_id: Provider tool-call identifier.
        """
        self._messages.append(
            ChatMessage(
                role=MessageRole.TOOL,
                content=content,
                name=name,
                tool_call_id=tool_call_id,
            )
        )
        self._trim_history()

    def history(self) -> list[ChatMessage]:
        """Return full in-memory history.

        Returns:
            Copy of stored messages in chronological order.
        """
        return list(self._messages)

    def _trim_history(self) -> None:
        """Trim non-system history while preserving chronological order."""
        non_system = self._messages[1:]
        if len(non_system) <= self._max_history_messages:
            return

        self._messages = [self._messages[0], *non_system[-self._max_history_messages :]]
