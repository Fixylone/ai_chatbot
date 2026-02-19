"""Domain models and abstract interfaces for the chatbot core.

This module defines the canonical data structures and the abstract contracts
that every concrete implementation (LLM wrapper, retriever, …) must satisfy.
Adding new back-ends only requires implementing these interfaces – the rest of
the application stays unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A single piece of source material ingested into the knowledge base."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """A single turn in a conversation (user or assistant)."""

    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class ChatRequest:
    """Payload sent by the caller to the chatbot."""

    query: str
    history: list[Message] = field(default_factory=list)


@dataclass
class ChatResponse:
    """Response returned by the chatbot."""

    answer: str
    sources: list[Document] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class BaseRetriever(ABC):
    """Contract for any document-retrieval back-end."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Return the *top_k* most relevant documents for *query*."""


class BaseLLM(ABC):
    """Contract for any language-model back-end."""

    @abstractmethod
    def generate(self, messages: list[Message]) -> str:
        """Generate a text completion given a list of *messages*."""


class BaseChatbot(ABC):
    """High-level contract for the chatbot itself."""

    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a user request and return a grounded response."""
