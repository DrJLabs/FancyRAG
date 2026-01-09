# Epic 1 – Hybrid Retrieval Foundation

## Overview
Epic 1 establishes the baseline infrastructure required to move from a vector-only Neo4j deployment to a hybrid retrieval stack that combines vector and full-text search. The work spans Neo4j indexing, ingestion pipeline resiliency, and a new FastMCP service that exposes secured tools for downstream consumers (e.g., ChatGPT connectors). Completing this epic gives the team a production-ready retrieval surface that other epics can extend with OAuth hardening, operational playbooks, and user integrations.

## Objectives
- Stand up a reproducible Neo4j indexing workflow that covers both vector and full-text indexes for `Chunk` nodes.
- Deliver a FastMCP server that wraps `HybridCypherRetriever`, exposing `search` and `fetch` tools backed by the local embedding service.
- Package the MCP server for containerized deployment alongside Neo4j so the stack can be run locally, in staging, and in production.
- Document the critical configuration (index names, embedding defaults, environment variables) so DevOps and developer agents can execute the pipeline without hand-holding.

## Scope
### In Scope
- Scripting and automation for vector and full-text index creation.
- FastMCP server implementation, including query embeddings through the local OpenAI-compatible API.
- Dockerfile and Docker Compose integration to run Neo4j and the MCP server on the `rag-net` network.
- `.env.example` updates covering index, embedding, and MCP configuration variables.

### Out of Scope
- OAuth enablement, monitoring/alerting, and ChatGPT connector onboarding (Epic 2).
- Production secret rotation and infrastructure-as-code modules (captured in later epics).
- UX or front-end surfaces; this epic focuses solely on backend services.

## Success Criteria
- Index creation scripts are idempotent and run cleanly through Makefile targets (`make index`, `make fulltext-index`).
- MCP server responds to authenticated requests with combined semantic + lexical results inside the target latency (≤1.5s for `top_k=5`).
- Docker Compose can launch both services (`docker compose up -d neo4j mcp`) and pass health checks (`/mcp/health`, sample `search` call).
- README/operations documentation describes end-to-end bootstrap within 30 minutes, aligned with PRD success metric.

## Primary Stories

| Story | Goal | Key Acceptance Criteria | Dependencies |
|-------|------|-------------------------|--------------|
| **1.1 Scripted Index Automation** | Provide repeatable setup for vector + full-text indexes. | 1. GraphRAG helper creates `chunk_text_fulltext`; reruns do not fail. 2. Makefile includes `fulltext-index` target analogous to `index`. 3. `.env.example` lists index variables (`INDEX_NAME`, `FULLTEXT_INDEX_NAME`, label/property overrides). | Requires existing `tools/run_pipeline.py` and `Makefile` available. |
| **1.2 Hybrid MCP Server** | Expose `search`/`fetch` tools powered by `HybridCypherRetriever`. | 1. Server config pulls index names and embedding credentials from environment. 2. Query embeddings use `OpenAIEmbeddings` against local base URL. 3. Retrieval response returns `text`, `metadata`, `score_vector`, `score_fulltext`. | Depends on Story 1.1 index names; needs `.env.local` entries for embedding service. |
| **1.3 Containerized Deployment** | Package MCP and integrate with Docker Compose. | 1. Dockerfile builds Python image with required deps (`fastmcp`, `neo4j-graphrag`, `python-dotenv`). 2. `docker-compose.yml` adds `mcp` service on `rag-net`, port `8080`. 3. `docker compose up -d neo4j mcp` succeeds and logs confirm server is ready. | Requires code from Story 1.2 and updated configuration from 1.1. |


**Status (2025-10-09)**: Stories 1.1 and 1.2 completed (QA + PO PASS). Story 1.3 Containerized Deployment is in active development.

## Dependencies & Inputs
- **Neo4j credentials**: Provided via `.env.local` (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`).
- **Embedding service**: Local OpenAI-compatible endpoint accessible at `EMBEDDING_API_BASE_URL`; ensure sample corpus is available for ingest.
- **GraphRAG library**: `neo4j-graphrag[experimental,openai]` already listed in `pyproject.toml`.
- **Docker environment**: Docker/Docker Compose must be installed on target machines; availability confirmed via README.

## Operational Runbook Snapshot
- **Build**: `docker compose build mcp` (reads `uv.lock`, installs via `uv sync --frozen`).
- **Start / Stop**: `make up` launches `neo4j` + `mcp` on `rag-net`; `make down` tears them down and prunes orphans.
- **Health**: `curl http://localhost:8080/mcp/health` mirrors the Compose healthcheck; use `docker compose ps` to confirm both services are `healthy`.
- **Logs**: `make logs` tails structured JSON output from Neo4j and FastMCP.
- **Auth for smoke tests**: Set `MCP_STATIC_TOKEN=<value>` in the Compose env file to enable deterministic bearer-token access without contacting Google OAuth.
- **Networking**: `mcp` joins `rag-net`, exposes port `8080`, and resolves the host stub via `host.docker.internal` (added through `extra_hosts`).
- **Port overrides**: set `MCP_PUBLISHED_PORT=<port>` before `make up` if host `8080` is occupied (defaults to `8080`).
- **Parallel stacks**: override `NEO4J_HTTP_PORT` / `NEO4J_BOLT_PORT` when another Neo4j container already binds the default host ports.

## Risks & Mitigations
- **Embedding API latency/outage**: Implement retries and document fallback to OpenAI-hosted embeddings or queuing (captured in PRD risk mitigation). Validate with smoke test in Story 1.2.
- **Index mismatch**: Ensure environment variables default to `text_embeddings` and `chunk_text_fulltext`; add preflight check in scripts to confirm existence.
- **Container image drift**: Document build command and push to registry as part of CI/CD (Epic 2 will handle full pipeline, but Story 1.3 should record exact `Dockerfile` and build instructions).

## Validation & Testing
- Unit tests for index script (e.g., verifying helper invocation using Neo4j test session mock).
- Integration smoke test (`tests/integration/test_container_smoke.py`): builds the Docker image, runs `docker compose up -d neo4j mcp`, seeds Neo4j data, polls `/mcp/health`, and posts to `/mcp/search` with a static bearer token.
- Manual verification: `curl http://localhost:8080/mcp/search -H "Authorization: Bearer <token>" -d '{"query":"sample"}'` using a development token stub.

## Timeline Guidance
1. Story 1.1 complete (2025-10-09); leverage established index naming and readiness automation.
2. Implement Story 1.2 immediately after, leveraging the established index variables.
3. Finish with Story 1.3 to containerize and validate runtime parity between local and staging.

## Exit Criteria
- All three stories marked ready with passing tests and documentation updates.
- Docker Compose instructions confirmed in README.
- Baseline smoke test script added (even if manual) showing index creation → ingest → hybrid query flow.

## Open Questions
- Additional MCP endpoints (beyond `search`/`fetch`) and advanced tuning (e.g., custom alpha weighting) are explicitly deferred to later epics.
- Use any concise synthetic dataset (e.g., 2–3 short paragraphs covering both lexical keywords and semantic variants) for CI/staging smoke tests; update fixtures as needed when expanding coverage.
