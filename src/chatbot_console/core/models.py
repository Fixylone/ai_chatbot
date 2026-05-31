"""Pydantic models for conversational runtime."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    """Supported chat message roles.

    Attributes:
        SYSTEM: Static behavior and policy instruction.
        USER: End-user input.
        ASSISTANT: Model-generated response.
        TOOL: External tool execution output.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """Single message entry in the in-memory conversation history.

    Attributes:
        role: Message role.
        content: Human-readable message content.
        name: Optional tool or assistant identifier.
        tool_call_id: Optional ID linking tool output to a tool call request.
    """

    role: MessageRole
    content: str = Field(min_length=1)
    name: str | None = None
    tool_call_id: str | None = None


class ToolCallRequest(BaseModel):
    """Tool call request parsed from an LLM response.

    Attributes:
        id: Provider tool-call identifier.
        name: Tool function name.
        arguments: Parsed JSON arguments for the tool call.
    """

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    """Normalized result payload from local tool execution.

    Attributes:
        name: Executed tool name.
        success: Whether the tool execution succeeded.
        output: Tool output or error message.
    """

    name: str
    success: bool
    output: str


class ChatTurnResult(BaseModel):
    """Result of processing one user turn.

    Attributes:
        assistant_message: Final assistant reply printed to the console.
        tools_used: Tool names used in this turn.
    """

    assistant_message: str = Field(min_length=1)
    tools_used: list[str] = Field(default_factory=list)
