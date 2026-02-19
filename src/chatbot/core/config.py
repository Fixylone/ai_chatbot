"""Application settings loaded from environment variables / .env file.

Usage
-----
    from chatbot.core.config import settings

    print(settings.openai_api_key)
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the AI chatbot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o-mini", description="Chat model name")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Vector store / RAG
    chroma_persist_dir: str = Field(
        default="data/chroma",
        description="Directory where ChromaDB stores its data",
    )
    retriever_top_k: int = Field(default=5, ge=1, le=50)

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_reload: bool = Field(default=False)


# Module-level singleton – import this instead of instantiating Settings yourself.
settings = Settings()
