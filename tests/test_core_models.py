"""Smoke tests for the chatbot package skeleton.

These tests validate that the project is importable and that core data
structures behave as expected – no external services required.
"""

from __future__ import annotations

from chatbot import __version__
from chatbot.core.interfaces import (
    ChatRequest,
    ChatResponse,
    Document,
    Message,
)


def test_version_string() -> None:
    assert isinstance(__version__, str)
    assert __version__  # not empty


class TestDomainModels:
    def test_document_defaults(self) -> None:
        doc = Document(content="example text")
        assert doc.content == "example text"
        assert doc.metadata == {}

    def test_message_fields(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_chat_request_empty_history(self) -> None:
        req = ChatRequest(query="Is Notepad++ approved?")
        assert req.query == "Is Notepad++ approved?"
        assert req.history == []

    def test_chat_response_defaults(self) -> None:
        resp = ChatResponse(answer="Yes, it is approved.")
        assert "approved" in resp.answer
        assert resp.sources == []

    def test_chat_response_with_sources(self) -> None:
        doc = Document(content="Policy: Notepad++ is approved.", metadata={"page": 1})
        resp = ChatResponse(answer="Approved", sources=[doc])
        assert len(resp.sources) == 1
        assert resp.sources[0].metadata["page"] == 1
