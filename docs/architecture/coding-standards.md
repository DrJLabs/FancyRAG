# Coding Standards

## Core Standards
- **Languages & Runtimes:** Python 3.12.
- **Style & Linting:** `ruff` for linting + import sorting; `black` formatting with line length 100.
- **Dependencies:** Install `neo4j-graphrag[experimental,openai,qdrant]`, `neo4j>=5.28`, `qdrant-client>=1.8`, `openai>=1.31` inside the project virtualenv.
- **Test Organization:** Mirror `src` layout (`tests/unit/...`, `tests/integration/...`).

## Critical Rules
- **Secrets Handling:** Never log API keys or passwords; mask values before logging. Scripts must read credentials from `.env`/environment and allow CLI overrides without echoing secrets.
- **Driver Lifecycle:** Use context managers for Neo4j sessions and ensure Qdrant clients are closed gracefully. Reuse a single driver/client per script run.
- **Retry Limits:** Cap OpenAI retries (max 3, configurable via `OPENAI_MAX_ATTEMPTS`) and log token usage to protect against runaway cost. Apply exponential backoff for 429/503 responses.
- **Idempotency:** Vector index creation, KG ingestion, and Qdrant upsert scripts must be safe to rerun; guard writes with `MERGE`/upsert semantics.

## Script Conventions
- Provide `argparse` CLIs with sensible defaults sourced from `.env` (e.g., `--dimensions`, `--collection`, `--question`).
- Emit structured logs using `structlog` or JSON-formatted `logging` with `operation`, `status`, `duration_ms`, and key identifiers (`index_name`, `collection`).
- Exit non-zero on failure and raise `SystemExit` with concise remediation hints.
- Flag long-running operations with progress counters (e.g., batch upsert of embeddings) and configurable batch sizes.
- Centralise OpenAI interactions through `cli.openai_client.SharedOpenAIClient` to inherit retries, telemetry, redaction, and fallback guardrails.

## Logging Guidance
- Prefer structured logging via `structlog` or JSON-formatted `logging` handlers.
- Include correlation ID, operation name, duration, and status in log entries.
- Log the Neo4j index name, Qdrant collection, and OpenAI model version for traceability.

## Testing Expectations
- Run `pytest` for unit/integration suites; add CI workflow to gate merges.
- Provide sandbox fixtures or mocks for Neo4j, Qdrant, and OpenAI dependencies. Integration smoke test should spin up the compose stack, seed sample content, and assert a grounded answer.
- Ingestion QA must remain fully covered by tests: ensure unit coverage for metric collectors/threshold evaluators and integration coverage proving that failed gates roll back partial graph writes.
