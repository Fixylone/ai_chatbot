"""Unit tests for summary request checker."""

from chatbot_console.utils.summary_checker import is_summary_request


class TestSummaryChecker:
    """Tests for natural-language summary detection behavior."""

    def test_detects_summary_request_with_builtin_terms(self) -> None:
        """Regex-backed terms should trigger summary detection."""
        assert is_summary_request("Can you summarize this chat?", []) is True

    def test_detects_summary_request_with_custom_keywords(self) -> None:
        """Configured custom keywords should trigger summary detection."""
        assert is_summary_request("Need a quick digest", ["digest"]) is True

    def test_returns_false_for_non_summary_text(self) -> None:
        """Non-summary text should not trigger summary behavior."""
        assert is_summary_request("What is 2 + 2?", ["digest"]) is False
