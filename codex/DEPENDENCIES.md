# Dependency Management (uv)

## 1. Package Tooling

- **Strict Rule:** This project exclusively uses **`uv`** (by Astral) for managing dependencies and virtual environments.
- **Prohibition:** Do not use standard `pip install`, `poetry`, or `pipenv` commands.

## 2. Initialization and Installation

- To create a virtual environment and synchronize dependencies from the `pyproject.toml` file, use the command:
  `uv sync`
- To add a new package to the project, use the command:
  `uv add <package_name>`
- To add a development-only package (dev dependency), use:
  `uv add --dev <package_name>`

## 3. Running Scripts and Tools

- Always use `uv run` to execute Python scripts, servers, or tests to ensure they run within the correct environment.
- **FastAPI:** To start the development server, use:
  `uv run uvicorn src.chatbot.api.main:app --reload`
- **Testing:** To run the `pytest` suite:
  `uv run pytest`
- **Linting:** To run `ruff` for code style and linting:
  `uv run ruff check`

## 4. Modern Features (Python 3.13+)

- Ensure you are using the latest `uv` version to leverage experimental features like JIT or free-threaded builds if required for synthetic data generation.
- All dependencies must be compatible with Python 3.13+.
