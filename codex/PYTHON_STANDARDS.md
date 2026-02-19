# Python Coding Standards (2026)

## 1. Syntax & Style
- **Follow PEP 8:** Use `snake_case` for functions/variables and `PascalCase` for classes.
- **Line Length:** Strictly 88 characters (Black/Ruff standard).
- **Docstrings:** Use Google-style docstrings for all public methods and classes.
- **F-Strings:** Use f-strings for all string formatting.

## 2. Type Safety (Crucial)
- **Strict Typing:** All function signatures MUST include type hints (e.g., `def get_data(id: int) -> dict:`).
- **Pydantic v2:** Use Pydantic models for data validation and configuration instead of raw dictionaries.
- **Generics:** Use `typing.Generic` and `TypeVar` where appropriate to maintain type safety.

## 3. Modern Features (Python 3.10+)
- **Pattern Matching:** Use `match` statements instead of complex `if/elif` chains.
- **Asyncio:** Always use `async/await` for I/O-bound tasks (API calls, DB queries).
- **Deferred Annotations:** Leverage PEP 649 for faster startup and cleaner introspection.