"""Unit tests for chat service orchestration."""

import asyncio
from collections.abc import Sequence
from typing import Any

from chatbot_console.core.config import ChatConfig
from chatbot_console.core.models import ChatMessage, ToolCallRequest
from chatbot_console.services.chat_service import ChatService
from chatbot_console.services.llm_gateway import AssistantGatewayResponse


class _FakeGateway:
    """Deterministic fake gateway for service tests."""

    def __init__(self, responses: list[AssistantGatewayResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    async def generate_response(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: str,
        summary_requested: bool,
    ) -> AssistantGatewayResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "tool_choice": tool_choice,
                "summary_requested": summary_requested,
            }
        )
        return self._responses.pop(0)


def test_process_user_message_without_tool_call_returns_assistant_reply() -> None:
    """No-tool turn should store and return assistant output directly."""
    gateway = _FakeGateway(
        [
            AssistantGatewayResponse(
                content="Sure, I can help with that.",
                tool_calls=[],
            )
        ]
    )
    service = ChatService(
        config=ChatConfig(),
        gateway=gateway,
        system_prompt="System rules.",
    )

    result = asyncio.run(service.process_user_message("Hello"))

    assert result.assistant_message == "Sure, I can help with that."
    assert result.tools_used == []
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["summary_requested"] is False


def test_process_user_message_executes_tool_and_requests_followup() -> None:
    """Tool request should execute local tool and trigger assistant followup call."""
    gateway = _FakeGateway(
        [
            AssistantGatewayResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="tool_1",
                        name="add_days_to_date",
                        arguments={"date": "2026-05-26", "days": 2},
                    )
                ],
            ),
            AssistantGatewayResponse(
                content="The resulting date is 2026-05-28.",
                tool_calls=[],
            ),
        ]
    )
    service = ChatService(
        config=ChatConfig(),
        gateway=gateway,
        system_prompt="System rules.",
    )

    result = asyncio.run(
        service.process_user_message("What date is two days after 2026-05-26?")
    )

    assert result.assistant_message == "The resulting date is 2026-05-28."
    assert result.tools_used == ["add_days_to_date"]
    assert len(gateway.calls) == 2
    assert gateway.calls[0]["tool_choice"] == "auto"
    assert gateway.calls[1]["tool_choice"] == "none"


def test_process_user_message_summary_request_disables_tools() -> None:
    """Summary requests should force no-tool mode for that turn."""
    gateway = _FakeGateway(
        [
            AssistantGatewayResponse(
                content="Summary of conversation so far.",
                tool_calls=[],
            )
        ]
    )
    service = ChatService(
        config=ChatConfig(),
        gateway=gateway,
        system_prompt="System rules.",
    )

    result = asyncio.run(service.process_user_message("Can you summarize this chat?"))

    assert result.assistant_message == "Summary of conversation so far."
    assert result.tools_used == []
    assert gateway.calls[0]["summary_requested"] is True
    assert gateway.calls[0]["tool_choice"] == "none"
    assert gateway.calls[0]["tools"] == []
