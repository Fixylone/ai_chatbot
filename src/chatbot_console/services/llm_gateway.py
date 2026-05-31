"""Mirascope-backed chat gateway runtime."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from mirascope import llm
from pydantic import BaseModel, Field

from chatbot_console.core.config import ChatConfig
from chatbot_console.core.models import ChatMessage, ToolCallRequest
from chatbot_console.utils.prompt_loader import render_prompt

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class AssistantGatewayResponse(BaseModel):
    """Normalized assistant output returned by the gateway.

    Attributes:
        content: Assistant textual reply.
        tool_calls: Parsed tool calls requested by the model.
    """

    content: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)


class ChatGateway(Protocol):
    """Protocol for chat completion backends."""

    async def generate_response(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: str,
        summary_requested: bool,
    ) -> AssistantGatewayResponse:
        """Create one assistant response from role-aware messages.

        Args:
            messages: Ordered role messages.
            tools: Optional function tool schemas.
            tool_choice: Provider tool choice behavior.

        Returns:
            Normalized gateway response.
        """


class ToolCallGatewayResponse(BaseModel):
    """LLM-only tool-call payload schema.

    Attributes:
        id: Tool-call identifier.
        name: Tool function name.
        arguments_json: JSON object string for function arguments.
    """

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments_json: str


class GatewayLLMResponse(BaseModel):
    """LLM-only response schema for gateway completion.

    Attributes:
        content: Assistant text response.
        tool_calls: Tool call plans requested by the model.
    """

    content: str
    tool_calls: list[ToolCallGatewayResponse]


class MirascopeChatGateway:
    """Mirascope chat gateway adapter with optional function tools.

    Args:
        config: Chat configuration values.
    """

    def __init__(self, config: ChatConfig) -> None:
        self._config = config
        if config.api_key:
            os.environ["OPENAI_API_KEY"] = config.api_key
        if config.api_base_url:
            os.environ["OPENAI_BASE_URL"] = config.api_base_url

    async def generate_response(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: str,
        summary_requested: bool,
    ) -> AssistantGatewayResponse:
        """Call provider with role-aware messages and optional tool schemas.

        Args:
            messages: Conversation messages.
            tools: Optional tool schema list.
            tool_choice: Tool-choice mode.

        Returns:
            Normalized assistant response.
        """
        prompt_text = await self._build_prompt(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            summary_requested=summary_requested,
        )

        parsed = await asyncio.wait_for(
            asyncio.to_thread(
                self._invoke_structured_gateway,
                prompt_text,
                self._resolved_model_id(),
                self._config.chat_temperature,
                self._config.chat_top_p,
            ),
            timeout=self._config.request_timeout_seconds,
        )
        parsed_tool_calls: list[ToolCallRequest] = []

        for raw_tool_call in parsed.tool_calls:
            parsed_args: dict[str, Any]
            try:
                loaded = json.loads(raw_tool_call.arguments_json or "{}")
            except json.JSONDecodeError:
                loaded = {"raw_arguments": raw_tool_call.arguments_json}

            parsed_args = loaded if isinstance(loaded, dict) else {"value": loaded}

            parsed_tool_calls.append(
                ToolCallRequest(
                    id=raw_tool_call.id,
                    name=raw_tool_call.name,
                    arguments=parsed_args,
                )
            )

        return AssistantGatewayResponse(
            content=parsed.content.strip(),
            tool_calls=parsed_tool_calls,
        )

    @staticmethod
    def _invoke_structured_gateway(
        prompt_text: str,
        model_id: str,
        temperature: float,
        top_p: float,
    ) -> GatewayLLMResponse:
        """Run the full Mirascope structured call synchronously.

        This is executed in a worker thread so that event-loop timeouts
        reliably apply even if provider setup blocks before the first await.
        """

        @llm.call(
            model_id,
            format=llm.format(GatewayLLMResponse, mode="strict"),
            temperature=temperature,
            top_p=top_p,
        )
        def _gateway_call(prompt_input: str) -> str:
            return prompt_input

        response = _gateway_call(prompt_text)
        return cast(GatewayLLMResponse, response.parse())

    async def _build_prompt(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: list[dict[str, Any]] | None,
        tool_choice: str,
        summary_requested: bool,
    ) -> str:
        """Build a plain-text prompt from role messages and tool metadata.

        Args:
            messages: Conversation messages in chronological order.
            tools: Enabled tool schemas.
            tool_choice: Tool selection policy.

        Returns:
            Prompt string for LLM structured response generation.
        """
        history_lines: list[str] = []
        for message in messages:
            role_label = message.role.value.upper()
            name_suffix = f"[{message.name}]" if message.name else ""
            tool_suffix = (
                f"(tool_call_id={message.tool_call_id})"
                if message.tool_call_id
                else ""
            )
            history_lines.append(
                f"{role_label}{name_suffix}{tool_suffix}: {message.content}"
            )

        tools_json = json.dumps(tools or [], ensure_ascii=True)

        return await render_prompt(
            _PROMPTS_DIR / "gateway_request.yaml",
            variables={
                "tool_choice": tool_choice,
                "tools_json": tools_json,
                "summary_requested": str(summary_requested).lower(),
                "conversation_history": "\n".join(history_lines),
            },
        )

    def _resolved_model_id(self) -> str:
        """Return provider-prefixed model id for Mirascope."""
        if "/" in self._config.chat_model:
            return self._config.chat_model
        return f"openai/{self._config.chat_model}"
