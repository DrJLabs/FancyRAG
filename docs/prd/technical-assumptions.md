# Technical Assumptions

- Repository structure: Monorepo with CLI scripts, infrastructure automation, and documentation.
- Service architecture: Single Python CLI orchestrator interacting with external Neo4j and Qdrant services.
- Testing requirements: Unit tests for helper modules plus integration smoke tests against sandbox services.
- Tooling preferences:
  - Python 3.12 with optional `pip-tools` or `uv` for lockfile management.
  - Neo4j Python driver 5.x, Qdrant client 1.10+, OpenAI SDK 1.x.
  - Logging via `structlog` or standard `logging` with JSON output.
- Infrastructure expectations: Qdrant reachable over internal network with API key; Neo4j accessible over Bolt; TLS termination handled by existing edge components.
