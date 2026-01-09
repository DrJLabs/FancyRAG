# Fancryrag Hybrid NeoRAG Upgrade Product Requirements Document (PRD)

## Goals and Background Context

### Goals
- Deliver hybrid (vector + full-text) retrieval so the NeoRAG deployment supports both semantic and lexical matching across Chunk nodes.
- Expose the retriever through a FastMCP server secured with Google OAuth, enabling ChatGPT connectors to call hybrid search and fetch tools.
- Maintain compatibility with the existing ingestion pipeline, local embedding API, and Neo4j deployment workflows.
- Provide repeatable infrastructure (Makefile targets, Docker services, helper scripts) so operators can bootstrap or reset indexes reliably.

### Background Context
The current project stands up a Neo4j 5.18 instance with a vector index (`text_embeddings`) and an ingestion pipeline that calls a local OpenAI-compatible embedding API. Retrieval still relies on vector-only lookups, limiting recall for keyword-heavy prompts. We must introduce hybrid retrieval while keeping the lightweight infrastructure footprint. This requires a full-text index on `Chunk.text`, a FastMCP server that leverages `HybridCypherRetriever`, and updated container orchestration so the retriever runs alongside Neo4j. OAuth is mandatory because the MCP server will be exposed over HTTPS (`neo.mcp.drjlabs.com`) for ChatGPT integrations.

### Change Log
| Date       | Version | Description                               | Author |
|------------|---------|-------------------------------------------|--------|
| 2025-10-08 | 0.1     | Initial draft covering hybrid upgrade PRD | John   |

## Requirements

### Functional Requirements
1. **FR1**: Provide a script (`scripts/create_fulltext_index.py`) that creates a Neo4j full-text index on `Chunk.text` using `neo4j_graphrag.indexes.create_fulltext_index`, defaulting to the index name `chunk_text_fulltext`.
2. **FR2**: Extend the Makefile with a `fulltext-index` target that invokes the new script via `docker compose exec neo4j python scripts/create_fulltext_index.py`.
3. **FR3**: Update configuration samples (`.env.example`) to include `INDEX_NAME`, `FULLTEXT_INDEX_NAME`, and defaults for full-text label/property so operators can override them consistently.
4. **FR4**: Implement a FastMCP server (`servers/mcp_hybrid_google.py`) that instantiates `HybridCypherRetriever`, exposes `search` and `fetch` tools, and reuses the existing local embedding API via `OpenAIEmbeddings`.
5. **FR5**: Integrate Google OAuth into the FastMCP server using `fastmcp.server.auth.providers.google.GoogleProvider`, sourcing credentials from environment variables and advertising OAuth metadata routes.
6. **FR6**: Add a Dockerfile that packages the MCP server with required dependencies (`fastmcp`, `neo4j-graphrag`, `python-dotenv`) and exposes port 8080.
7. **FR7**: Extend `docker-compose.yml` with an `mcp` service that depends on `neo4j`, shares the `rag-net` network, loads `.env.local`, and maps port 8080 for external access.
8. **FR8**: Document and automate operational tasks (install deps, start services, create indexes, ingest data, verify MCP health) so the hybrid flow can be reproduced end-to-end.
9. **FR9**: Preserve and validate the existing ingestion pipeline (`tools/run_pipeline.py`, `pipelines/kg_ingest.yaml`) to ensure it still writes chunks, embeddings, and relationships consumed by the hybrid retriever.
10. **FR10**: Provide guidance for ChatGPT connector setup, including OAuth redirect URL (`/auth/callback`) and MCP endpoint (`/mcp`), ensuring the hybrid tools appear post-authorization.

### Non-Functional Requirements
1. **NFR1**: Secure all MCP endpoints with Google OAuth 2.0; unauthenticated access must be rejected with appropriate HTTP status codes.
2. **NFR2**: The MCP server must respond to search requests within 1.5 seconds for a `top_k` of 5 on datasets up to 50k chunks, assuming Neo4j and the embedding API are reachable within the same network.
3. **NFR3**: Services must log structured JSON lines (at minimum include timestamp, level, event) to stdout/stderr so container logs can be centralized.
4. **NFR4**: All configuration must be environment-driven with documented defaults; no secrets or API keys are checked into source control.
5. **NFR5**: Dockerized services must start with a single `docker compose up -d neo4j mcp` command without manual build steps beyond `uv sync` and `.env.local` preparation.
6. **NFR6**: Index creation scripts must be idempotent—re-running them should not fail if indexes already exist.

## Technical Assumptions
- **Languages & Runtime**: Python 3.12 managed via Astral `uv`.
- **Core Libraries**: `neo4j-graphrag[experimental,openai]` for pipeline and retrievers, `fastmcp` for MCP server, `neo4j` Python driver, `python-dotenv` for environment loading.
- **Neo4j Deployment**: Single container (`neo4j:5.18`) with APOC enabled, connected to the `rag-net` bridge network, default database `neo4j`.
- **Embedding Service**: External OpenAI-compatible API reachable at `EMBEDDING_API_BASE_URL`, providing the `local-embedding-768` model with cosine similarity (768 dimensions).
- **Authentication**: Google OAuth Web Application credentials stored in `.env.local`; FastMCP handles token verification and metadata exposure.
- **Hosting Expectations**: Production deployment fronted by HTTPS at `https://neo.mcp.drjlabs.com`; local development uses port-forwarded `http://localhost:8080`.
- **MCP Integration**: ChatGPT connectors consume the MCP server via `streamable-http` transport; tokens are obtained through the Google OAuth flow.

## Data & Indexing Considerations
- **Graph Schema**: Ingestion pipeline writes `Chunk` nodes with properties including `text`, `embedding`, and relationships produced by the entity extractor.
- **Vector Index**: Already established as `text_embeddings` (cosine, 768 dim) using `createNodeIndex`.
- **Full-Text Index**: New `chunk_text_fulltext` index over `Chunk.text`, enabling lexical recall; environment variables allow alternative label/property overrides.
- **Retrieval Query**: Default query returns text content, node metadata, vector score, and full-text score; configurable via `RETRIEVAL_QUERY`.
- **Data Volume**: Designed for small-to-medium corpora (< 50k chunks) with ingestion performed through existing pipeline (`make ingest f=<file>`).

## Success Metrics
- Hybrid search returns combined lexical/semantic results with measurable recall improvements (target: ≥20% increase in relevant document hits for keyword-centric queries compared to vector-only baseline).
- OAuth-protected MCP server passes manual penetration check (all endpoints require bearer tokens).
- End-to-end bootstrap (dependencies → compose → index creation → ingestion → hybrid query) completed within 30 minutes on a fresh environment.

## Release Criteria
- All functional and non-functional requirements validated in a staging environment with sample data.
- `docker compose up -d neo4j mcp` followed by `make index` and `make fulltext-index` succeeds without manual intervention.
- `curl` health and search requests return expected payloads when authorized, and 401/403 responses when unauthenticated.
- Documentation in README (or dedicated operations guide) updated to reflect hybrid workflow.

## Risks & Mitigations
- **Risk**: Misconfigured OAuth credentials block MCP access.  
  **Mitigation**: Provide clear environment variable documentation, add startup validation that logs missing credentials explicitly. Assign credential provisioning to platform engineer; document rotation steps shared with Product Owner.
- **Risk**: Full-text index sync delays degrade freshness.  
  **Mitigation**: Document reindex process post-ingestion; optionally schedule nightly re-creation.
- **Risk**: Custom embedding API downtime breaks query embeddings.  
  **Mitigation**: Implement retries and surface errors in MCP responses; document fallback to OpenAI-hosted embeddings or queue requests if service unavailable.
- **Risk**: Hybrid retriever performance issues with large graphs.  
  **Mitigation**: Allow tuning of `top_k`, `alpha`, and query template via environment variables; capture metrics for query time.

## Open Questions
- Do we need additional MCP tools (e.g., node neighborhood expansion) beyond `search` and `fetch`?
- Should the MCP server expose adjustable alpha weighting via request parameters?
- What monitoring/alerting stack will observe Neo4j index health and MCP availability in production?

## Milestones & Epics

### Epic 1: Hybrid Retrieval Foundation
Deliver the foundational infrastructure to support hybrid search across Neo4j.

- **Story 1.1**: As a Neo4j operator, I want a scripted way to create both vector and full-text indexes so that I can bootstrap retrieval consistently.  
  **Acceptance Criteria**  
  1. Script creates `chunk_text_fulltext` via GraphRAG helper, no errors on repeated runs.  
  2. Makefile exposes `fulltext-index` target documented alongside `index`.  
  3. `.env.example` lists index-related variables with defaults.

- **Story 1.2**: As a backend engineer, I want a FastMCP server that wraps `HybridCypherRetriever` so that ChatGPT can run hybrid search with minimal configuration.  
  **Acceptance Criteria**  
  1. Server exposes `search` (hybrid) and `fetch` tools using shared Neo4j driver.  
  2. Query embeddings use environment-configured OpenAI-compatible service.  
  3. Retrieval query returns text, metadata, vector and full-text scores.

- **Story 1.3**: As a deployment engineer, I want the MCP server containerized and orchestrated with Neo4j so that the stack runs via Docker Compose.  
  **Acceptance Criteria**  
  1. Dockerfile builds Python image with FastMCP and project sources.  
  2. `docker-compose.yml` includes `mcp` service on `rag-net` with port 8080.  
  3. `docker compose up -d neo4j mcp` starts successfully and logs confirm server readiness.

### Epic 2: OAuth Enablement & Operational Hardening
Secure the MCP server and document operational workflows.

- **Story 2.1**: As a security engineer, I want Google OAuth protecting MCP endpoints so that only authorized users can query hybrid search.  
  **Acceptance Criteria**  
  1. GoogleProvider configured with client ID/secret and base URL.  
  2. Unauthenticated requests return 401/403, authenticated requests succeed.  
  3. OAuth metadata (`/.well-known/oauth-protected-resource`) exposes issuer & DCR endpoints.

- **Story 2.2**: As an operator, I want documented end-to-end runbooks so that I can ingest data and validate hybrid retrieval in under 30 minutes.  
  **Acceptance Criteria**  
  1. README (or new ops guide) includes dependency install, compose, index creation, ingestion, and verification steps.  
  2. Example `curl` commands for health and search documented with token placeholder.  
  3. Checklist for environment variables (Neo4j, embeddings, OAuth) included.

- **Story 2.3**: As an integration engineer, I want guidance for ChatGPT connector setup so that stakeholders can connect without ambiguity.  
  **Acceptance Criteria**  
  1. Document connector URL, OAuth redirect, and required scopes.  
  2. Instructions confirm hybrid tools appear after connector authorization.  
  3. Notes capture troubleshooting tips for common OAuth or network errors.

## Next Steps
- Prepare Google OAuth credentials and populate `.env.local`.
- Story 1.1 completed on 2025-10-09 (full-text index automation); kick off Story 1.2 Hybrid MCP Server.
- Coordinate with DevOps to provision HTTPS endpoint (`neo.mcp.drjlabs.com`) ahead of MCP deployment.
- Define CI/CD pipeline stages (lint, unit, integration smoke) and staging deployment workflow before first implementation PR.
