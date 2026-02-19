"""Retrieval-Augmented Generation (RAG) pipeline.

This module wires together a document retriever and an LLM to produce grounded
answers.  Documents are stored in a local ChromaDB vector store and retrieved
via semantic similarity search.

Typical usage
-------------
    from chatbot.services.rag import RAGService

    service = RAGService()
    service.ingest([Document(content="…", metadata={"source": "policy.pdf"})])
    response = service.chat(ChatRequest(query="Is Notepad++ approved?"))
"""

from __future__ import annotations

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from chatbot.core.config import settings
from chatbot.core.interfaces import (
    BaseChatbot,
    BaseRetriever,
    ChatRequest,
    ChatResponse,
    Document,
    Message,
)
from chatbot.services.llm import OpenAILLM
from chatbot.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are an AI assistant that helps software whitelisting teams verify "
    "approval decisions. Answer questions strictly based on the provided context. "
    "If the context does not contain enough information, say so clearly."
)


# ---------------------------------------------------------------------------
# ChromaDB-backed retriever
# ---------------------------------------------------------------------------


class ChromaRetriever(BaseRetriever):
    """Semantic similarity retriever backed by a persistent ChromaDB store."""

    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )
        self._store = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self._embeddings,
        )
        logger.info("retriever_initialized", persist_dir=settings.chroma_persist_dir)

    def add_documents(self, documents: list[Document]) -> None:
        """Embed and persist *documents* into the vector store."""
        lc_docs = [
            {"page_content": d.content, "metadata": d.metadata} for d in documents
        ]
        self._store.add_texts(
            texts=[d["page_content"] for d in lc_docs],
            metadatas=[d["metadata"] for d in lc_docs],
        )
        logger.info("documents_ingested", count=len(documents))

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Return the *top_k* most semantically similar documents."""
        results = self._store.similarity_search(query, k=top_k)
        docs = [Document(content=r.page_content, metadata=r.metadata) for r in results]
        logger.debug("documents_retrieved", query=query, count=len(docs))
        return docs


# ---------------------------------------------------------------------------
# RAG chatbot
# ---------------------------------------------------------------------------


class RAGService(BaseChatbot):
    """End-to-end RAG chatbot for whitelisting approval assistance."""

    def __init__(self) -> None:
        self._retriever = ChromaRetriever()
        self._llm = OpenAILLM()

    def ingest(self, documents: list[Document]) -> None:
        """Add *documents* to the knowledge base."""
        self._retriever.add_documents(documents)

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Retrieve relevant context and generate a grounded answer."""
        sources = self._retriever.retrieve(request.query, top_k=settings.retriever_top_k)

        context = "\n\n".join(
            f"[Source {i + 1}]\n{doc.content}" for i, doc in enumerate(sources)
        )
        system_msg = Message(role="system", content=_SYSTEM_PROMPT)
        context_msg = Message(
            role="system",
            content=f"Use the following context to answer the user's question:\n\n{context}",
        )
        messages: list[Message] = [
            system_msg,
            context_msg,
            *request.history,
            Message(role="user", content=request.query),
        ]

        answer = self._llm.generate(messages)
        logger.info("chat_completed", query=request.query[:80])
        return ChatResponse(answer=answer, sources=sources)
