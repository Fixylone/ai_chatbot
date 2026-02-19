**Focus:** Speed and RAG-specific logic.

```markdown
# Performance & RAG Best Practices

## 1. Optimization
- **Vectorization:** Use `NumPy` or `Pandas` for any data manipulation—avoid manual `for` loops.
- **JIT & Free-Threading:** In Python 3.13+, leverage the experimental JIT or free-threaded build if performing CPU-heavy synthetic data generation.
- **Batching:** Always batch embedding calls to reduce API latency.

## 2. RAG Logic (Phase 3 & 4)
- **Chunking:** Use recursive character splitting with a 10% overlap unless semantic boundaries are detected.
- **Metadata:** Every chunk must retain its source file name and page number as metadata.
- **Agents:** Use `LangGraph` for agentic loops. Prefer "Stateful Agents" that can backtrack if a search result is irrelevant.

## 3. Evaluation (Phase 5)
- Use the **RAGAS** framework for automated evaluation.
- Metrics to track: `Faithfulness`, `Answer Relevancy`, and `Context Precision`.