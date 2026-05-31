"""Console entrypoint for the chatbot application."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer

from chatbot_console.core.config import load_chat_config
from chatbot_console.services.chat_service import ChatService
from chatbot_console.services.llm_gateway import MirascopeChatGateway
from chatbot_console.utils.prompt_loader import render_prompt

app = typer.Typer(add_completion=False, help="Interactive console chatbot")


def _merge_overrides(**kwargs: Any) -> dict[str, Any]:
    """Drop None values and keep only explicit CLI overrides.

    Args:
        **kwargs: Potential override values.

    Returns:
        Dict containing only explicitly set values.
    """
    return {key: value for key, value in kwargs.items() if value is not None}


async def _run_chat_session(
    config_path: Path,
    overrides: dict[str, Any],
) -> None:
    """Run interactive one-session chat loop.

    Args:
        config_path: Chat YAML config path.
        overrides: Explicit CLI overrides.
    """
    config = load_chat_config(config_path=config_path, overrides=overrides)
    system_prompt = await render_prompt(config.system_prompt_path)

    service = ChatService(
        config=config,
        gateway=MirascopeChatGateway(config),
        system_prompt=system_prompt,
    )

    typer.echo("Chatbot Console")
    typer.echo("Type your message. Use exit/quit to end the session.")

    exit_commands = set(config.exit_commands)

    while True:
        try:
            raw_input = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nSession ended.")
            return

        if not raw_input:
            continue

        if raw_input.lower() in exit_commands:
            typer.echo("Session ended.")
            return

        try:
            result = await service.process_user_message(raw_input)
            typer.echo(f"assistant> {result.assistant_message}")
        except Exception as exc:
            typer.echo(f"assistant> Error: {type(exc).__name__}: {exc}")


@app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    config: Path = typer.Option(
        Path("chat_config.yaml"),
        help="Path to chat YAML config.",
    ),
    chat_model: str | None = typer.Option(None, help="Override chat model id."),
    chat_temperature: float | None = typer.Option(
        None,
        help="Override chat temperature.",
    ),
    chat_top_p: float | None = typer.Option(None, help="Override chat top_p."),
    max_history_messages: int | None = typer.Option(
        None,
        help="Max non-system messages to retain.",
    ),
    system_prompt_path: Path | None = typer.Option(
        None,
        help="Override system prompt YAML path.",
    ),
    enable_get_current_date: bool | None = typer.Option(
        None,
        "--enable-get-current-date/--disable-get-current-date",
        help="Enable or disable get_current_date tool.",
    ),
    enable_add_days_to_date: bool | None = typer.Option(
        None,
        "--enable-add-days-to-date/--disable-add-days-to-date",
        help="Enable or disable add_days_to_date tool.",
    ),
    api_base_url: str | None = typer.Option(
        None,
        help="Optional OpenAI-compatible base URL.",
    ),
    api_key: str | None = typer.Option(None, help="Optional API key override."),
) -> None:
    """Start the interactive console chatbot session.

    Args:
        ctx: Typer context.
        config: YAML config path.
        chat_model: Optional model override.
        chat_temperature: Optional temperature override.
        chat_top_p: Optional top_p override.
        max_history_messages: Optional history cap.
        system_prompt_path: Optional prompt template path override.
        enable_get_current_date: Optional tool toggle.
        enable_add_days_to_date: Optional tool toggle.
        api_base_url: Optional base URL override.
        api_key: Optional API key override.
    """
    if ctx.invoked_subcommand is not None:
        return

    overrides = _merge_overrides(
        chat_model=chat_model,
        chat_temperature=chat_temperature,
        chat_top_p=chat_top_p,
        max_history_messages=max_history_messages,
        system_prompt_path=system_prompt_path,
        enable_get_current_date=enable_get_current_date,
        enable_add_days_to_date=enable_add_days_to_date,
        api_base_url=api_base_url,
        api_key=api_key,
    )

    asyncio.run(_run_chat_session(config, overrides))


def main() -> None:
    """Launch Typer application.

    This function is used by the console script entrypoint.
    """
    app()
