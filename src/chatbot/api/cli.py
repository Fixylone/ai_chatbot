"""Command-line interface entry point.

Usage
-----
    chatbot serve            # Start the FastAPI HTTP server
    chatbot --help           # Show available commands
"""

from __future__ import annotations

import argparse
import sys

from chatbot.utils.logger import get_logger

logger = get_logger(__name__)


def _serve(args: argparse.Namespace) -> None:
    """Start the Uvicorn HTTP server."""
    import uvicorn  # noqa: PLC0415

    from chatbot.core.config import settings  # noqa: PLC0415

    host = args.host or settings.api_host
    port = args.port or settings.api_port

    logger.info("starting_server", host=host, port=port)
    uvicorn.run(
        "chatbot.api.main:app",
        host=host,
        port=port,
        reload=settings.api_reload,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chatbot",
        description="AI Chatbot – whitelisting approval assistant",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Start the HTTP API server")
    serve_p.add_argument("--host", default=None, help="Bind host (default from settings)")
    serve_p.add_argument("--port", type=int, default=None, help="Bind port (default from settings)")
    serve_p.set_defaults(func=_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point registered in pyproject.toml."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
