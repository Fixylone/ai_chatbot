"""CLI entry points for generation and validation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer

from chatbot.core.config import load_config
from chatbot.core.pipeline import run_pipeline
from chatbot.utils.validation import validate_all

app = typer.Typer(add_completion=False, help="Synthetic legal data generator")


def _merge_overrides(**kwargs: Any) -> dict[str, Any]:
    """Drop ``None`` values and keep only explicit CLI overrides."""
    return {key: value for key, value in kwargs.items() if value is not None}


@app.command()
def generate(
    config: Path = typer.Option(Path("config.yaml"), help="Path to YAML config."),
    toc_model: str | None = typer.Option(None, help="Override TOC model id."),
    section_model: str | None = typer.Option(
        None,
        help="Override section model id.",
    ),
    ideation_model: str | None = typer.Option(
        None,
        help="Override ideation model id.",
    ),
    toc_temperature: float | None = typer.Option(None, help="TOC temperature."),
    section_temperature: float | None = typer.Option(
        None,
        help="Section temperature.",
    ),
    ideation_temperature: float | None = typer.Option(
        None,
        help="Ideation temperature.",
    ),
    top_p: float | None = typer.Option(None, help="Nucleus sampling top_p."),
    num_tools: int | None = typer.Option(None, help="Number of tools to generate."),
    docs_per_tool: int | None = typer.Option(
        None,
        help="Number of documents per tool.",
    ),
    rate_limit_max_retries: int | None = typer.Option(
        None,
        help="Retries for rate limit (429) failures.",
    ),
    rate_limit_base_backoff_seconds: float | None = typer.Option(
        None,
        help="Base retry backoff seconds for 429 handling.",
    ),
    rate_limit_max_backoff_seconds: float | None = typer.Option(
        None,
        help="Maximum retry backoff seconds for 429 handling.",
    ),
    runtime_feedback: bool | None = typer.Option(
        None,
        help="Enable live runtime feedback while generation is running.",
    ),
    prompt_caching_enabled: bool | None = typer.Option(
        None,
        help="Enable provider prompt caching request parameters.",
    ),
    prompt_cache_retention: str | None = typer.Option(
        None,
        help="Prompt cache retention policy: in_memory or 24h.",
    ),
    prompt_cache_key_namespace: str | None = typer.Option(
        None,
        help="Namespace used when building prompt cache keys.",
    ),
    section_summary_max_sections: int | None = typer.Option(
        None,
        help="Max previous sections summarized in compressed context.",
    ),
    section_summary_max_chars: int | None = typer.Option(
        None,
        help="Max characters in previous-sections summary.",
    ),
    section_last_section_max_chars: int | None = typer.Option(
        None,
        help="Max characters kept for last full section HTML context.",
    ),
    output_dir: Path | None = typer.Option(None, help="Output directory path."),
    document_type: list[str] | None = typer.Option(
        None,
        "--document-type",
        help="Repeat option to override document type pool.",
    ),
) -> None:
    """Generate full synthetic dataset for Phase 1."""
    overrides = _merge_overrides(
        toc_model=toc_model,
        section_model=section_model,
        ideation_model=ideation_model,
        toc_temperature=toc_temperature,
        section_temperature=section_temperature,
        ideation_temperature=ideation_temperature,
        top_p=top_p,
        num_tools=num_tools,
        docs_per_tool=docs_per_tool,
        rate_limit_max_retries=rate_limit_max_retries,
        rate_limit_base_backoff_seconds=rate_limit_base_backoff_seconds,
        rate_limit_max_backoff_seconds=rate_limit_max_backoff_seconds,
        runtime_feedback=runtime_feedback,
        prompt_caching_enabled=prompt_caching_enabled,
        prompt_cache_retention=prompt_cache_retention,
        prompt_cache_key_namespace=prompt_cache_key_namespace,
        section_summary_max_sections=section_summary_max_sections,
        section_summary_max_chars=section_summary_max_chars,
        section_last_section_max_chars=section_last_section_max_chars,
        output_dir=output_dir,
        document_types=document_type,
    )

    cfg = load_config(config_path=config, overrides=overrides)
    records, report = asyncio.run(run_pipeline(cfg))

    typer.echo(
        f"Generated {len(records)} documents. "
        f"Validation: {report.valid_files}/{report.total_files} valid files."
    )

    if report.invalid_files > 0:
        typer.echo("Invalid files detected:")
        for result in report.results:
            if result.is_valid:
                continue
            typer.echo(f"- {result.path}")
            for error in result.errors:
                typer.echo(f"    * {error}")
        raise typer.Exit(code=1)


@app.command()
def validate(
    output_dir: Path = typer.Option(Path("data"), help="Directory to validate."),
) -> None:
    """Validate generated HTML and JSON artifacts."""
    report = validate_all(output_dir)
    typer.echo(
        f"Validation summary: {report.valid_files}/{report.total_files} valid "
        f"(html={report.html_files}, json={report.json_files})."
    )

    if report.invalid_files > 0:
        typer.echo("Invalid files:")
        for result in report.results:
            if result.is_valid:
                continue
            typer.echo(f"- {result.path}")
            for error in result.errors:
                typer.echo(f"    * {error}")
        raise typer.Exit(code=1)
