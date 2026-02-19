"""Tests for the Settings configuration model."""

from __future__ import annotations

from chatbot.core.config import Settings


def test_default_settings() -> None:
    s = Settings()
    assert s.llm_model == "gpt-4o-mini"
    assert s.llm_temperature == 0.0
    assert s.retriever_top_k == 5
    assert s.api_port == 8000


def test_settings_override_via_env(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("RETRIEVER_TOP_K", "10")
    s = Settings()
    assert s.llm_model == "gpt-4o"
    assert s.retriever_top_k == 10
