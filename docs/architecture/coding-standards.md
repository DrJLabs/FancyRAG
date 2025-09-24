# Coding Standards

## Core Standards
- **Languages & Runtimes:** Python 3.12.
- **Style & Linting:** `ruff` for linting + import sorting; `black` formatting with line length 100.
- **Test Organization:** Mirror `src` layout (`tests/unit/...`, `tests/integration/...`).

## Critical Rules
- **Secrets Handling:** Never log API keys or passwords; mask values before logging.
- **Driver Lifecycle:** Use context managers for Neo4j sessions and ensure Qdrant clients are closed gracefully.
- **Retry Limits:** Cap OpenAI retries (max 5) and log token usage to protect against runaway cost.

## Logging Guidance
- Prefer structured logging via `structlog` or JSON-formatted `logging` handlers.
- Include correlation ID, operation name, duration, and status in log entries.

## Testing Expectations
- Run `pytest` for unit/integration suites; add CI workflow to gate merges.
- Provide sandbox fixtures or mocks for Neo4j, Qdrant, and OpenAI dependencies.
