# Architecture Overview

## Technical Summary
The Neo4j GraphRAG solution is a Python 3.12 CLI that orchestrates knowledge graph ingestion, vector synchronization, and retrieval with the official `neo4j-graphrag` library. It integrates with managed Neo4j for graph storage, Qdrant for vectors, and OpenAI for generation and embeddings. Core patterns include a modular ingestion pipeline, idempotent vector writes, and a retrieval flow that enriches vector hits with graph context.

## High-Level Components
- CLI orchestrator that houses subcommands for ingest, vectors, and search operations.
- Knowledge Graph Builder leveraging `SimpleKGPipeline` to populate Neo4j.
- Vector upsert service handling embedding, batching, and Qdrant payload consistency.
- Retrieval engine that joins Qdrant hits with Neo4j entities and delegates answer generation to OpenAI models.
- Workspace bootstrap script (`scripts/bootstrap.sh`) that provisions the Python 3.12 virtual environment, installs pinned dependencies, and validates `neo4j_graphrag` imports before operators run CLI workflows. Run it from the repository root (`bash scripts/bootstrap.sh`) and activate the resulting virtualenv (`source .venv/bin/activate`) prior to completing the `.env` configuration story.

## Diagram
```mermaid
graph TD
    Operator[Operator CLI]
    CLI[Python CLI]
    GraphRAG[Neo4j GraphRAG]
    Neo4j[(Neo4j DB)]
    Qdrant[(Qdrant Collection)]
    OpenAI[(OpenAI APIs)]

    Operator --> CLI --> GraphRAG
    GraphRAG --> Neo4j
    GraphRAG --> Qdrant
    GraphRAG --> OpenAI
    Qdrant <-->|neo4j_id| Neo4j
```

## Change Log
| Date       | Version | Description                               | Author    |
|------------|---------|-------------------------------------------|-----------|
| 2025-09-24 | 0.1     | Seeded architecture overview shard        | Codex CLI |
| 2025-09-25 | 0.2     | Documented environment configuration workflow | James      |
| 2025-09-25 | 0.3     | Added workspace diagnostics guidance       | James      |

## Environment Configuration
- Copy `.env.example` to `.env` immediately after running `scripts/bootstrap.sh` and before executing CLI commands.
- Populate the placeholders with environment-specific values: `OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4o-mini`, optional `gpt-4.1-mini`), `OPENAI_EMBEDDING_MODEL`, `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `QDRANT_URL`, and `QDRANT_API_KEY`.
- Keep `.env` git-ignored; never commit real credentials or paste secrets into shared channels.
- Ensure the values align with managed service endpoints (Neo4j Bolt URI and Qdrant HTTPS URL) documented in the architecture.

## Workspace Verification
- After bootstrapping and populating `.env`, validate the environment with `PYTHONPATH=src python -m cli.diagnostics workspace --write-report` (or add `--verify` when running `scripts/bootstrap.sh`).
- The diagnostics command imports `neo4j_graphrag`, `neo4j`, `qdrant_client`, `openai`, `structlog`, and `pytest`, failing fast when dependencies are missing or misconfigured.
- A structured report is written to `artifacts/environment/versions.json` capturing Python runtime, package versions (via `importlib.metadata`), the SHA-256 of `requirements.lock`, and the current git commit for audit trails.
- Output is redacted automatically—no environment variables or secrets are persisted—allowing the report to be shared with operators and CI systems.
- Rerun diagnostics whenever dependencies change (`pip install`/`pip-compile` updates) or prior to releasing automation changes so drift is detected early.
- Sample report payload:
  ```json
  {
    "generated_at": "2025-09-25T15:20:20Z",
    "python": {"version": "3.12.3", "executable": ".venv/bin/python"},
    "packages": [
      {"module": "neo4j_graphrag", "distribution": "neo4j-graphrag", "version": "1.10.0"},
      {"module": "neo4j", "distribution": "neo4j", "version": "5.28.2"},
      {"module": "qdrant_client", "distribution": "qdrant-client", "version": "1.15.1"},
      {"module": "openai", "distribution": "openai", "version": "1.109.1"},
      {"module": "structlog", "distribution": "structlog", "version": "24.4.0"},
      {"module": "pytest", "distribution": "pytest", "version": "8.4.2"}
    ],
    "lockfile": {"sha256": "<64-char hex>", "path": "requirements.lock", "exists": true},
    "git": {"sha": "f4b0f87c"}
  }
  ```
