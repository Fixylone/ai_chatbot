"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def test_client() -> TestClient:
    """Return a synchronous TestClient for the FastAPI app.

    The RAGService startup is intentionally **not** triggered here; individual
    tests that need a live service should mock it out.
    """
    from chatbot.api.main import create_app

    application = create_app()
    return TestClient(application, raise_server_exceptions=True)
