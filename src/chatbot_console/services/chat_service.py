"""Main orchestration service for one interactive chat session."""

from __future__ import annotations

from chatbot_console.core.config import ChatConfig
from chatbot_console.core.models import ChatMessage, ChatTurnResult
from chatbot_console.services.conversation import ConversationManager
from chatbot_console.services.llm_gateway import (
    AssistantGatewayResponse,
    ChatGateway,
)
from chatbot_console.services.tool_registry import (
    execute_tool_call,
    get_enabled_tool_schemas,
)
from chatbot_console.utils.retry import retry_llm_call
from chatbot_console.utils.summary_checker import is_summary_request


class ChatService:
    """Process user turns with history, tools, and summary continuity.

    Args:
        config: Chat runtime configuration.
        gateway: LLM gateway implementation.
        system_prompt: Session-wide system prompt.
    """

    def __init__(
        self,
        config: ChatConfig,
        gateway: ChatGateway,
        system_prompt: str,
    ) -> None:
        self._config = config
        self._gateway = gateway
        self._conversation = ConversationManager(
            system_prompt=system_prompt,
            max_history_messages=config.max_history_messages,
        )

    async def process_user_message(self, user_input: str) -> ChatTurnResult:
        """Process one user turn and return assistant output.

        Args:
            user_input: Raw user message.

        Returns:
            ChatTurnResult with final assistant text and tool usage metadata.
        """
        self._conversation.add_user_message(user_input)

        summary_requested = is_summary_request(
            user_input=user_input,
            keywords=self._config.summary_keywords,
        )

        tools = [] if summary_requested else get_enabled_tool_schemas(self._config)
        tool_choice = "none" if summary_requested or not tools else "auto"

        initial_response = await self._request_assistant_response(
            tools=tools,
            tool_choice=tool_choice,
            summary_requested=summary_requested,
        )

        tools_used: list[str] = []
        final_response = initial_response

        for tool_call in initial_response.tool_calls:
            result = execute_tool_call(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                config=self._config,
            )
            tools_used.append(result.name)
            tool_payload = self._format_tool_message(
                result.name,
                result.success,
                result.output,
            )
            self._conversation.add_tool_message(
                content=tool_payload,
                name=result.name,
                tool_call_id=tool_call.id,
            )

        if initial_response.tool_calls:
            final_response = await self._request_assistant_response(
                tools=tools,
                tool_choice="none",
                summary_requested=summary_requested,
            )

        assistant_text = final_response.content.strip()
        if not assistant_text:
            raise ValueError("LLM returned empty assistant response.")

        self._conversation.add_assistant_message(assistant_text)

        return ChatTurnResult(
            assistant_message=assistant_text,
            tools_used=tools_used,
        )

    def history(self) -> list[ChatMessage]:
        """Get current in-memory chat history.

        Returns:
            Chronological message list.
        """
        return self._conversation.history()

    async def _request_assistant_response(
        self,
        *,
        tools: list[dict[str, object]],
        tool_choice: str,
        summary_requested: bool,
    ) -> AssistantGatewayResponse:
        """Call the gateway with retries for one assistant response.

        Args:
            tools: Tool schema list.
            tool_choice: Provider tool choice mode.
            summary_requested: Whether user requested a summary.

        Returns:
            AssistantGatewayResponse from the backend.
        """
        messages = self._conversation.history()

        async def _call() -> AssistantGatewayResponse:
            return await self._gateway.generate_response(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                summary_requested=summary_requested,
            )

        return await retry_llm_call(_call, self._config)

    @staticmethod
    def _format_tool_message(tool_name: str, success: bool, output: str) -> str:
        """Format a tool execution payload for tool-role messages.

        Args:
            tool_name: Executed tool name.
            success: Tool success state.
            output: Tool output payload.

        Returns:
            Tool-role message content.
        """
        status = "success" if success else "error"
        return f"tool={tool_name}; status={status}; output={output}"

