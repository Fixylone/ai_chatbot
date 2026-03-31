# ai_chatbot

Phase 1 synthetic legal/compliance document generator.

This project generates synthetic legal-style datasets for fictional software
tools, including:

- Structured TOC files (`toc_*.json`)
- Per-document issue manifests (`issues_*.json`)
- Full HTML documents (`*.html`)
- Programmatic validation for HTML and generated JSON integrity

## What Phase 1 does

1. Generates fictional tool descriptions.
2. Assigns document types per tool.
3. Generates structured TOC JSON per document.
4. Generates section-by-section HTML content using TOC + prior section context.
5. Generates section-level issue manifest JSON per document.
6. Validates all generated outputs.

## Project structure

The project follows a `src` layout:

- `src/chatbot/core`: config, models, pipeline orchestration
- `src/chatbot/services`: ideation, TOC, and section generation
- `src/chatbot/utils`: prompt loading, HTML/JSON helpers, validation
- `src/chatbot/prompts`: YAML prompt templates
- `data`: generated artifacts

## Configuration

Default settings are in `config.yaml` and can be overridden via CLI flags or
environment variables (`CHATBOT_*`).

Current default model strategy:

- `toc_model: l2-gpt-4.1-mini` (reasoning-focused TOC generation)
- `section_model: l2-gpt-4o-mini` (cost-efficient section drafting)
- `ideation_model: l2-gpt-4.1-nano` (strong tool ideation quality/cost)

Validation rule: at least two distinct model IDs are required across these
three generation stages.

## Usage

Install dependencies in your active environment, then run:

```bash
chatbot generate --config config.yaml
```

Validate output directory:

```bash
chatbot validate --output-dir data
```

Example overrides:

```bash
chatbot generate \
	--toc-model l2-gpt-4.1-mini \
	--section-model l2-gpt-4o-mini \
	--ideation-model l2-gpt-4.1-nano \
	--num-tools 5 \
	--docs-per-tool 4
```

## Output layout

Generated files are organized per tool:

```text
data/
	<tool_name_slug>/
		toc_privacy_policy.json
		issues_privacy_policy.json
		privacy_policy.html
		...
```

## Notes

- Prompt templates are externalized in YAML files (not inline code strings).
- TOC and section generation use strict structured outputs.
- Section generation enforces 2-3 total intentional quality issues per document.
- Runtime resilience is retry-only (bounded exponential backoff for transient
  failures).
- Section continuity context is compressed into:
	- summary of previous sections
	- full HTML of the most recent generated section
- Optional prompt caching controls are available in `config.yaml`:
	- `prompt_caching_enabled`
	- `prompt_cache_retention`
	- `prompt_cache_key_namespace`
