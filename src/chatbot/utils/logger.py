"""Structured logger factory.

Usage
-----
    from chatbot.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("server_started", port=8000)
"""

from __future__ import annotations

import logging
import sys

import structlog


def _configure_structlog() -> None:
    """Set up structlog with a human-friendly console renderer."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


_configured = False


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for *name*.

    Calling this function is idempotent – structlog is configured at most once
    per interpreter session.
    """
    global _configured  # noqa: PLW0603
    if not _configured:
        _configure_structlog()
        _configured = True
    return structlog.get_logger(name)
