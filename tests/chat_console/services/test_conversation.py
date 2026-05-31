"""Unit tests for in-memory conversation state management."""

from chatbot_console.core.models import MessageRole
from chatbot_console.services.conversation import ConversationManager


class TestConversationManager:
    """Tests for role ordering and deterministic truncation."""

    def test_keeps_system_message_and_trims_old_non_system_messages(self) -> None:
        """History should retain system message and trim oldest non-system messages."""
        manager = ConversationManager(
            system_prompt="System rules.",
            max_history_messages=4,
        )

        manager.add_user_message("u1")
        manager.add_assistant_message("a1")
        manager.add_user_message("u2")
        manager.add_assistant_message("a2")
        manager.add_user_message("u3")

        history = manager.history()

        assert history[0].role == MessageRole.SYSTEM
        assert [message.content for message in history[1:]] == ["a1", "u2", "a2", "u3"]

    def test_add_tool_message_stores_tool_metadata(self) -> None:
        """Tool messages should preserve name and tool_call_id metadata."""
        manager = ConversationManager(
            system_prompt="System rules.",
            max_history_messages=6,
        )

        manager.add_tool_message(
            content="tool=get_current_date; status=success; output=2026-05-26",
            name="get_current_date",
            tool_call_id="call_1",
        )

        tool_message = manager.history()[1]

        assert tool_message.role == MessageRole.TOOL
        assert tool_message.name == "get_current_date"
        assert tool_message.tool_call_id == "call_1"
