"""Unit tests for tool registry and execution behavior."""

from datetime import date

from chatbot_console.core.config import ChatConfig
from chatbot_console.services import tool_registry


class _FixedDate(date):
    """Fixed date replacement for deterministic tests."""

    @classmethod
    def today(cls) -> date:
        """Return deterministic date for tests."""
        return cls(2026, 5, 26)


class TestToolRegistry:
    """Tests for local tool execution and validation behavior."""

    def test_get_current_date_returns_iso_date(
        self,
        monkeypatch,
    ) -> None:
        """Current-date tool should return system date in ISO format."""
        monkeypatch.setattr(tool_registry, "date", _FixedDate)
        cfg = ChatConfig()

        result = tool_registry.execute_tool_call(
            tool_name=tool_registry.GET_CURRENT_DATE_TOOL,
            arguments={},
            config=cfg,
        )

        assert result.success is True
        assert result.output == "2026-05-26"

    def test_add_days_to_date_supports_positive_and_negative_offsets(self) -> None:
        """Date-offset tool should support signed day deltas."""
        cfg = ChatConfig()

        plus_result = tool_registry.execute_tool_call(
            tool_name=tool_registry.ADD_DAYS_TO_DATE_TOOL,
            arguments={"date": "2026-05-26", "days": 5},
            config=cfg,
        )
        minus_result = tool_registry.execute_tool_call(
            tool_name=tool_registry.ADD_DAYS_TO_DATE_TOOL,
            arguments={"date": "2026-05-26", "days": -3},
            config=cfg,
        )

        assert plus_result.success is True
        assert plus_result.output == "2026-05-31"
        assert minus_result.success is True
        assert minus_result.output == "2026-05-23"

    def test_add_days_to_date_returns_error_for_invalid_arguments(self) -> None:
        """Invalid tool args should fail gracefully without exceptions."""
        cfg = ChatConfig()

        result = tool_registry.execute_tool_call(
            tool_name=tool_registry.ADD_DAYS_TO_DATE_TOOL,
            arguments={"date": "26-05-2026", "days": "abc"},
            config=cfg,
        )

        assert result.success is False
        assert "Invalid tool arguments" in result.output
