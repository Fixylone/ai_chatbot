"""Tests for the FastAPI routes (no external services required).

The RAGService startup hook is skipped by using ``TestClient`` without
triggering the lifespan events, so these tests remain fast and offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> TestClient:
    from chatbot.api.main import create_app

    return TestClient(create_app(), raise_server_exceptions=False)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_503_when_not_ready(client: TestClient) -> None:
    """Before startup, the RAG service is None → 503 expected."""
    payload = {"query": "Is Notepad++ approved?"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 503


def test_ingest_endpoint_returns_503_when_not_ready(client: TestClient) -> None:
    payload = {"documents": [{"content": "test doc", "metadata": {}}]}
    response = client.post("/ingest", json=payload)
    assert response.status_code == 503
