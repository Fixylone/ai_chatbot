# Project Architecture Guidelines

## 1. Directory Layout (The `src` Layout)
Follow the modern `src` layout to prevent accidental imports and ensure testability:
```text
project_root/
├── pyproject.toml        # The modern replacement for requirements.txt
├── src/
│   └── chatbot/          # Main package
│       ├── core/         # Core logic, interfaces
│       ├── services/     # RAG logic, LLM wrappers
│       ├── api/          # FastAPI or CLI entry points
│       └── utils/        # Loggers, helpers
├── tests/                # Pytest suite
└── data/                 # Local data storage