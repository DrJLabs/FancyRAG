# Neo4j GraphRAG Product Requirements Document (PRD)

## Goals and Background Context
### Goals
- Deliver a self-contained GraphRAG evaluation stack that runs on a single Linux host using Docker Compose for Neo4j and Qdrant plus the official `neo4j-graphrag` package.
- Provide scripted workflows that create the Neo4j vector index, build the knowledge graph, mirror embeddings into Qdrant, and execute retrieval checks end-to-end.
- Maintain compatibility with production-bound settings so the same scripts can later point at managed services without code changes.
- Improve maintainability by decomposing the `scripts/kg_build.py` monolith into a modular `src/fancyrag/` package with clear testing guardrails.

### Background Context
Earlier iterations assumed existing managed Neo4j and Qdrant deployments. That scope stalled adoption because teams could not safely experiment without access to those services. Version 1 focuses on a project-owned stack: Dockerized Neo4j 5.26.12 (APOC Core) and Qdrant 1.15.4 (see [Version Matrix](../README.md#version-matrix)), paired with Python 3.12 tooling that relies exclusively on the public `neo4j-graphrag` APIs (`SimpleKGPipeline`, `create_vector_index`, `QdrantNeo4jRetriever`). Once the local workflow is proven, the same scripts will target managed infrastructure by overriding connection variables.

### Change Log
| Date       | Version | Description                                                    | Author     |
|------------|---------|----------------------------------------------------------------|------------|
| 2025-09-24 | 0.1     | Rebuilt PRD in BMAD format; aligned goals, requirements, epics | Codex CLI  |
| 2025-09-28 | 0.2     | Narrowed scope to local Docker stack + scripted minimal path   | Codex CLI  |
| 2025-10-02 | 0.3     | Added refactor roadmap and linked planning artifacts          | Codex CLI  |

## Requirements
### Functional
- **FR1:** Provide `docker-compose.neo4j-qdrant.yml` that starts Neo4j 5.26.12 (APOC enabled) and Qdrant 1.15.4 with persistent volumes and configurable credentials.
- **FR2:** Ship a reproducible Python 3.12 environment that installs `neo4j-graphrag[experimental,openai,qdrant]`, validates imports, and documents required `.env` variables.
- **FR3:** Implement a script that creates the Neo4j vector index on `Chunk.embedding` with configurable dimensions (default 1536) while reusing `neo4j_graphrag.indexes.create_vector_index`.
- **FR4:** Implement a KG builder script that runs `SimpleKGPipeline` against sample content (text or PDF) and persists Document/Chunk nodes plus embeddings.
- **FR5:** Implement export and retrieval scripts that batch embeddings into Qdrant, preserve join keys, and exercise `GraphRAG.search()` using `QdrantNeo4jRetriever` to return grounded answers.

### Non Functional
- **NFR1:** The end-to-end workflow (compose up → scripts) completes in under 60 minutes on a developer laptop with 16 GB RAM.
- **NFR2:** Secrets remain outside version control; scripts read credentials from `.env` and support overrides via environment variables.
- **NFR3:** Logging is structured (JSON or key-value) and includes durations and statuses for each major step.
- **NFR4:** Scripts exit non-zero on failure and provide actionable error messages for retry attempts.

## User Interface Design Goals
- **Overall UX Vision:** Operate entirely via documented CLI commands and scripts; outputs must be copy-paste friendly.
- **Key Interaction Paradigms:** Command-line prompts, environment variables, and scripted pipelines; no GUI.
- **Core Screens and Views:** N/A (CLI only); focus on terminal output sections for install, ingestion, retrieval verification, and operational status.
- **Accessibility:** N/A — CLI inherits terminal accessibility features.
- **Branding:** Plain-text output with optional structured logging.
- **Target Device and Platforms:** Linux/macOS shell environments.

## Technical Assumptions
- **Repository Structure:** Monorepo containing CLI scripts, Docker Compose files, configuration modules, and documentation.
- **Service Architecture:** Python CLI orchestrator interacting with locally managed Neo4j and Qdrant containers by default; connection variables allow redirection to managed services later.
- **Testing Requirements:** Unit tests for helper functions; integration smoke test that runs the retrieval script against the local Docker stack with seeded sample content.
- **Additional Technical Assumptions and Requests:**
  - Python 3.12 interpreter with option to pin dependencies via `pip-tools` or `uv`.
  - Qdrant reachable at `http://localhost:6333` when using the default compose file; API key optional for local use.
  - Neo4j reachable over Bolt `neo4j://localhost:7687` with credentials sourced from `.env`.
  - Logging via `structlog` or standard library logging configured for JSON output.

## Epic List
| Epic | Title                               | Goal Statement                                                                         |
|------|-------------------------------------|----------------------------------------------------------------------------------------|
| 1    | Environment & Workspace             | Establish reproducible Python environment and configuration surface for local stack.   |
| 2    | Local GraphRAG Minimal Path         | Deliver Dockerized Neo4j/Qdrant plus scripts covering KG build, vector sync, retrieval. |
| 3    | Ingestion Quality Hardening         | Add adaptive chunking, QA telemetry, and semantic enrichment for reliable ingestion.    |
| 4    | FancyRAG `kg_build.py` Refactor     | Break the KG build monolith into modular packages with dedicated tests and guardrails. |

## Epic Details
### Epic 1 — Environment & Workspace
**Goal:** Provide a dependable development shell that standardizes Python versioning, installs `neo4j-graphrag[experimental,openai,qdrant]`, and documents `.env` variables that default to the local Docker stack.
- Create and document virtual environment bootstrap (`python3 -m venv`, `pip install "neo4j-graphrag[experimental,openai,qdrant]"`).
- Record package versions and validate imports via `scripts/bootstrap.sh` diagnostics.
- Publish sample `.env` template listing required variables and pointing at the compose defaults (`neo4j://localhost:7687`, `http://localhost:6333`).

### Epic 2 — Local GraphRAG Minimal Path
**Goal:** Ensure Docker Compose, KG builder, vector index, and Qdrant export/retrieval scripts deliver an end-to-end GraphRAG demonstration locally.
- Add `docker-compose.neo4j-qdrant.yml` with persistent volumes and documented environment overrides.
- Provide a script that runs `create_vector_index` and a `SimpleKGPipeline` ingestion with sample content.
- Provide export and retrieval scripts that push embeddings to Qdrant and execute `GraphRAG.search()` to verify grounded answers.


### Epic 3 — Ingestion Quality Hardening
**Goal:** Raise ingestion fidelity with adaptive chunking, QA gating, and semantic enrichment so telemetry can block bad data before retrieval.
- Introduce chunking presets and deterministic directory ingestion with metadata captured for observability.
- Expand QA evaluation to emit JSON/Markdown reports and enforce thresholds tied to CI checks.
- Wire semantic enrichment toggles that leverage GraphRAG `LLMEntityRelationExtractor` while maintaining rollback controls.
- Status: Delivered 2025-10-02 (see `docs/bmad/focused-epics/ingestion-quality/epic.md`).

### Epic 4 — FancyRAG `kg_build.py` Monolith Decomposition
**Goal:** Break the `scripts/kg_build.py` script into a `src/fancyrag/` package with dedicated modules for CLI wiring, pipeline orchestration, QA utilities, Neo4j helpers, and configuration.
- Follow the guardrails defined in the project brief and architecture addendum (single responsibility modules, typed interfaces, 200–400 LOC targets).
- Keep `scripts/kg_build.py` as a thin CLI bridge that defers to `src/fancyrag/cli/kg_build_main.py`.
- Stand up per-module unit tests plus a CLI smoke test that proves parity with the pre-refactor behaviour.
- Planning artifacts: `docs/prd/projects/fancyrag-kg-build-refactor/`, `docs/architecture/projects/fancyrag-kg-build-refactor.md`, and `docs/bmad/focused-epics/kg-build-refactor/epic.md`.
- Status: Planned (kick-off after Epic 3 hardening).

