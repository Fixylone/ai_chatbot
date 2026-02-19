"""FastAPI application factory and HTTP routes.

Run with::

    uvicorn chatbot.api.main:app --reload

Or via the project entry-point::

    chatbot serve
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from chatbot.core.interfaces import ChatRequest, ChatResponse, Document, Message
from chatbot.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas (Pydantic v2)
# ---------------------------------------------------------------------------


class MessageSchema(BaseModel):
    role: str
    content: str


class ChatRequestSchema(BaseModel):
    query: str
    history: list[MessageSchema] = Field(default_factory=list)


class DocumentSchema(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)


class ChatResponseSchema(BaseModel):
    answer: str
    sources: list[DocumentSchema] = Field(default_factory=list)


class IngestRequestSchema(BaseModel):
    documents: list[DocumentSchema]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Lazy import to avoid loading heavy ML deps during testing
    from chatbot.services.rag import RAGService  # noqa: PLC0415

    rag: RAGService | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        nonlocal rag
        rag = RAGService()
        logger.info("app_started")
        yield

    application = FastAPI(
        title="AI Chatbot – Whitelisting Assistant",
        description=(
            "Semantic search chatbot that assists software whitelisting teams "
            "with approval verification decisions."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @application.post(
        "/chat",
        response_model=ChatResponseSchema,
        tags=["chat"],
        status_code=status.HTTP_200_OK,
    )
    async def chat(body: ChatRequestSchema) -> ChatResponseSchema:
        """Ask the chatbot a question."""
        if rag is None:
            raise HTTPException(status_code=503, detail="Service not ready")

        request = ChatRequest(
            query=body.query,
            history=[Message(role=m.role, content=m.content) for m in body.history],
        )
        try:
            result: ChatResponse = rag.chat(request)
        except Exception as exc:
            logger.error("chat_error", error=str(exc))
            raise HTTPException(status_code=500, detail="Internal error") from exc

        return ChatResponseSchema(
            answer=result.answer,
            sources=[
                DocumentSchema(content=d.content, metadata=d.metadata)
                for d in result.sources
            ],
        )

    @application.post(
        "/ingest",
        tags=["knowledge-base"],
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def ingest(body: IngestRequestSchema) -> dict[str, int]:
        """Add documents to the knowledge base."""
        if rag is None:
            raise HTTPException(status_code=503, detail="Service not ready")

        docs = [Document(content=d.content, metadata=d.metadata) for d in body.documents]
        try:
            rag.ingest(docs)
        except Exception as exc:
            logger.error("ingest_error", error=str(exc))
            raise HTTPException(status_code=500, detail="Internal error") from exc

        return {"ingested": len(docs)}

    return application


app = create_app()
