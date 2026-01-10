# Fancryrag Tech Stack

The table below defines the approved technologies for the hybrid NeoRAG upgrade. Agents must adhere to these choices unless an architecture update is approved.

## Core Languages & Runtime
- **Python**: 3.12 (managed with Astral `uv`)
- **Shell**: POSIX-compliant Bash for automation scripts

## Frameworks & Libraries
| Layer | Technology | Version/Notes | Purpose |
|-------|------------|---------------|---------|
| Graph Processing | `neo4j-graphrag[experimental,openai]` | Latest compatible with Neo4j 5.18 | Ingestion pipeline, retrievers, index helpers |
| Database Driver | `neo4j` Python driver | `>=5.18,<6` | Bolt connectivity to Neo4j |
| MCP Runtime | `fastmcp` | Latest | Hybrid retrieval server with OAuth |
| Configuration | `python-dotenv` | `>=1.0.1,<2` | Loads `.env.local` for scripts and servers |
| Embeddings | `OpenAIEmbeddings` via GraphRAG | Uses external base URL | Query & ingestion embeddings |
| Testing | `pytest`, `pytest-asyncio`, `testcontainers` (planned) | Pin when added | Automated test suite |
| Lint/Format | `ruff` | Add to dev dependencies | Linting and code formatting |

## Data & Storage
- **Neo4j**: Version 5.18 with APOC plugin; hosted via Docker container `neo4j:5.18`.
- **Vector Index**: `text_embeddings` (cosine, 768 dimensions) built with GraphRAG helper.
- **Full-Text Index**: `chunk_text_fulltext` targeting `Chunk.text`.
- **Volumes**: Docker volumes `neo4j-data`, `neo4j-logs`, `neo4j-plugins` for persistence.

## External Services
- **Embedding API**: Custom OpenAI-compatible endpoint (`EMBEDDING_API_BASE_URL`, default `http://localhost:20010/v1`).
- **LLM Provider**: OpenAI `gpt-5-mini` for extraction (configurable via environment).
- **OAuth Provider**: Google OAuth 2.0 web application credentials.
- **ChatGPT Connector**: Consumes MCP server over HTTPS (`BASE_URL`).

## Tooling & Operations
- **Package Manager**: Astral `uv`.
- **Container Orchestration**: Docker Compose.
- **CI/CD (planned)**: GitHub Actions with lint/test/build stages.
- **Metrics & Observability (planned)**: Docker logs, Neo4j metrics; future Prometheus/OpenTelemetry integration.

## Environments
- **Local Development**: `.env.local` for credentials, `docker compose up -d neo4j mcp`.
- **Staging/Production**: Same container images deployed behind TLS-terminating ingress, with secrets injected via cloud secret manager.

Any deviation from this stack requires an update to this document and approval in architecture review.
