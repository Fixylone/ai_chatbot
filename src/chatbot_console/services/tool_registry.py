"""Tool registry and execution helpers for chat."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chatbot_console.core.config import ChatConfig
from chatbot_console.core.models import ToolExecutionResult

GET_CURRENT_DATE_TOOL = "get_current_date"
ADD_DAYS_TO_DATE_TOOL = "add_days_to_date"


class AddDaysArguments(BaseModel):
    """Validated arguments for add-days-to-date tool.

    Attributes:
        date_value: Base date in ISO format.
        days: Signed day offset.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    date_value: date = Field(alias="date")
    days: int = Field(ge=-365000, le=365000)


def get_enabled_tool_schemas(config: ChatConfig) -> list[dict[str, Any]]:
    """Build OpenAI-compatible function tool schema list.

    Args:
        config: Chat configuration with tool toggles.

    Returns:
        Function tool schema list.
    """
    tools: list[dict[str, Any]] = []

    if config.enable_get_current_date:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": GET_CURRENT_DATE_TOOL,
                    "description": "Return the current system date in ISO format.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            }
        )

    if config.enable_add_days_to_date:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": ADD_DAYS_TO_DATE_TOOL,
                    "description": (
                        "Add a signed number of days to a provided ISO date and "
                        "return the resulting ISO date."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Base date in YYYY-MM-DD format.",
                            },
                            "days": {
                                "type": "integer",
                                "description": (
                                    "Signed day offset, positive or negative."
                                ),
                            },
                        },
                        "required": ["date", "days"],
                        "additionalProperties": False,
                    },
                },
            }
        )

    return tools


def execute_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    config: ChatConfig,
) -> ToolExecutionResult:
    """Execute a single tool call and normalize the output.

    Args:
        tool_name: Requested tool name.
        arguments: Parsed tool argument object.
        config: Chat configuration with enable/disable switches.

    Returns:
        ToolExecutionResult with success flag and output payload.
    """
    match tool_name:
        case "get_current_date":
            if not config.enable_get_current_date:
                return ToolExecutionResult(
                    name=tool_name,
                    success=False,
                    output="Tool is disabled by configuration.",
                )
            return ToolExecutionResult(
                name=tool_name,
                success=True,
                output=date.today().isoformat(),
            )

        case "add_days_to_date":
            if not config.enable_add_days_to_date:
                return ToolExecutionResult(
                    name=tool_name,
                    success=False,
                    output="Tool is disabled by configuration.",
                )

            try:
                parsed = AddDaysArguments.model_validate(arguments)
            except ValidationError as exc:
                return ToolExecutionResult(
                    name=tool_name,
                    success=False,
                    output=f"Invalid tool arguments: {exc.errors()}",
                )

            calculated_date = parsed.date_value + timedelta(days=parsed.days)
            return ToolExecutionResult(
                name=tool_name,
                success=True,
                output=calculated_date.isoformat(),
            )

        case _:
            return ToolExecutionResult(
                name=tool_name,
                success=False,
                output="Unknown tool requested.",
            )
