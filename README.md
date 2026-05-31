# ai_chatbot

This repository now contains two independent console applications:

- Phase 1 synthetic legal/compliance document generator
- Phase 2 interactive LLM chatbot console application

Phase 1 and Phase 2 are intentionally separated so each can be run and tested
independently.

## Phase 1 Overview

The pipeline generates synthetic legal-style datasets for fictional software
tools, including:

- Structured TOC files (`toc_*.json`)
- Per-document issue manifests (`issues_*.json`)
- Full HTML documents (`*.html`)
- Programmatic validation for HTML and generated JSON integrity

## What Phase 1 does

1. Generates fictional tool descriptions.
2. Assigns document types per tool.
3. Generates structured TOC JSON per document.
4. Generates section-by-section HTML content using TOC and full prior-section
   context.
5. Generates section-level issue manifest JSON per document.
6. Validates all generated outputs.

## Project structure

The project follows a `src` layout:

- `src/chatbot/core`: config, models, pipeline orchestration
- `src/chatbot/services/data_generation`: ideation, TOC, and section generation
- `src/chatbot/utils`: prompt loading, HTML/JSON helpers, validation
- `src/chatbot/prompts`: YAML prompt templates
- `tests`: unit tests
- `data`: generated artifacts

## Configuration

Defaults are in `config.yaml` and can be overridden via:

- CLI flags
- Environment variables with `CHATBOT_` prefix

Current default model IDs:

- `toc_model: openai/l2-gpt-4.1-nano`
- `section_model: openai/l2-gpt-4.1-mini`
- `ideation_model: openai/l2-gpt-4.1-nano`

## First-time setup

### 1) Install dependencies

From repository root:

```bash
uv sync --all-groups
```

### 2) Set credentials

Set the environment variables required by your model gateway/provider.

PowerShell example:

```powershell
$env:OPENAI_API_KEY = "<your_key_here>"
```

### 3) Quick sanity checks

```bash
uv run chatbot --help
uv run pytest -q
```

## Phase 1 Usage

Generate dataset:

```bash
uv run chatbot generate --config config.yaml
```

Validate generated output:

```bash
uv run chatbot validate --output-dir data
```

Example generation override:

```bash
uv run chatbot generate \
  --config config.yaml \
  --num-tools 5 \
  --docs-per-tool 4 \
  --section-max-validation-retries 5 \
  --toc-max-validation-retries 3
```

## Phase 2 Overview

Phase 2 provides an interactive conversational console chatbot that:

- Maintains system/user/assistant/tool roles
- Preserves in-memory message history for one session run
- Uses tool calls for date logic
- Supports natural-language summary requests during the conversation

Required tools included:

- `get_current_date` returns current date in `YYYY-MM-DD`
- `add_days_to_date` applies a signed day offset to a provided date

## Phase 2 Configuration

Phase 2 defaults are in `chat_config.yaml` and can be overridden via:

- CLI flags
- Environment variables with `CHATBOT_CHAT_` prefix

## Phase 2 Usage

Start interactive chatbot:

```bash
uv run chatbot-console
```

Use a different config file:

```bash
uv run chatbot-console --config chat_config.yaml
```

Example overrides:

```bash
uv run chatbot-console \
  --chat-model openai/l2-gpt-4.1-mini \
  --chat-temperature 0.3 \
  --max-history-messages 30 \
  --enable-get-current-date \
  --enable-add-days-to-date
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
- Runtime resilience uses bounded exponential backoff for transient failures.
- Phase 2 chat history is in-memory only for the active process run.
