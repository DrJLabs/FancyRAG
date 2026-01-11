# Technical Assumptions

- Repository structure: Monorepo with CLI scripts, Docker Compose files, configuration modules, and documentation.
- Service architecture: Python CLI orchestrator interacts with local Dockerized Neo4j services by default; environment variables can redirect to managed clusters later. Qdrant is legacy/optional for export workflows only.
- Testing requirements: Unit coverage for helper utilities plus an integration smoke test that seeds sample text, runs the pipeline, and verifies a grounded answer.
- Tooling preferences:
  - Python 3.12 with optional `pip-tools` or `uv` for lockfile management.
  - Neo4j Python driver 5.x, OpenAI SDK 1.x. Qdrant client ≥ 1.8 only when running legacy export workflows.
  - Logging via `structlog` or standard `logging` with JSON output.
- Infrastructure expectations: Compose stack listens on localhost (`7474`, `7687`) for Neo4j; volumes persisted under `./.data/neo4j`; API keys injected through `.env` without committing secrets. Qdrant (`6333`) and `./.data/qdrant` only apply to the legacy stack.
