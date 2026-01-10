# Codebase Evaluation (FancyRAG)

## Scope

This review covers the current repository structure, core runtime paths, and the Neo4j GraphRAG ingestion + MCP server workflow.
It focuses on architecture clarity, build reproducibility, runtime safety, and operational ergonomics.

Key entrypoints reviewed:
- `servers/mcp_hybrid_google.py`
- `src/fancryrag/mcp/runtime.py`
- `src/fancyrag/kg/pipeline.py`
- `pipelines/kg_ingest.yaml` + `tools/run_pipeline.py`
- `Dockerfile`, `docker-compose.yml`, `Makefile`

## Current Architecture Summary

- **MCP server**: `servers/mcp_hybrid_google.py` boots FastMCP and delegates search/fetch to `fancryrag.mcp.runtime`.
- **Hybrid retrieval**: `src/fancryrag/mcp/runtime.py` uses `neo4j_graphrag.retrievers.HybridCypherRetriever` and issues extra vector/fulltext queries to provide normalized score breakdowns.
- **Ingestion pipeline**: `src/fancyrag/kg/pipeline.py` wraps `SimpleKGPipeline`, adds source discovery, chunking, DB reset support, semantic enrichment, and QA reports.
- **Config-driven pipeline**: `pipelines/kg_ingest.yaml` + `tools/run_pipeline.py` use the upstream `PipelineRunner` and explicit component wiring.
- **Containerized stack**: Docker/Compose runs Neo4j + MCP, with a Makefile for common ops (indexing, ingestion, smoke tests).

## Strengths

- **Clear operational path**: README + Makefile document a complete local workflow for Neo4j + MCP.
- **Good test coverage**: dedicated tests for runtime, config, pipeline, and smoke flows.
- **Structured logging**: consistent event-style logs in the server runtime.
- **Dependency isolation**: `uv` used for repeatable environments and Docker layering.

## Improvement Opportunities

### High Priority

1. **Build reproducibility (uv.lock)**
   - **Issue**: `uv.lock` is required by the Dockerfile (`COPY pyproject.toml uv.lock`) but is ignored in `.gitignore`, so fresh clones won’t build with `--frozen`.
   - **Impact**: container builds fail or become non-reproducible across environments.
   - **Suggestion**: commit `uv.lock` (preferred for reproducible builds) or update Dockerfile to generate it and avoid `--frozen` when missing.
   - **Files**: `Dockerfile`, `.gitignore`.

2. **Package naming + namespace collisions**
   - **Issue**: both `src/fancyrag/` and `src/fancryrag/` exist; additional top‑level packages (`src/cli`, `src/config`, `src/_compat`) are very generic and likely unintentionally exported by setuptools.
   - **Impact**: import confusion, accidental namespace conflicts, ambiguous public API surface.
   - **Suggestion**: consolidate to a single canonical package (and optionally provide a compatibility alias), and explicitly declare exported packages in `pyproject.toml`.
   - **Files**: `src/fancyrag/*`, `src/fancryrag/*`, `src/cli/*`, `src/config/*`, `src/_compat/*`, `pyproject.toml`.

### Medium Priority

3. **Two parallel ingestion paths**
   - **Issue**: there are two ingestion approaches: `SimpleKGPipeline` in `src/fancyrag/kg/pipeline.py` and a config‑driven `PipelineRunner` via `pipelines/kg_ingest.yaml` and `tools/run_pipeline.py`.
   - **Impact**: behavioral drift and maintenance overhead.
   - **Suggestion**: choose one canonical ingestion path and align tools/Makefile/docs around it. If advanced customization is needed, migrate to `Pipeline`/`PipelineRunner` and model pre/post steps as components.
   - **Files**: `src/fancyrag/kg/pipeline.py`, `pipelines/kg_ingest.yaml`, `tools/run_pipeline.py`, `Makefile`.

4. **Index configuration drift risk**
   - **Issue**: `make index` hardcodes 1024 dimensions and a fixed index name, while config and embedder models can be changed independently.
   - **Impact**: mismatched embeddings/index dimensions and brittle ops.
   - **Suggestion**: derive index name/dimensions from config or env variables and validate at startup.
   - **Files**: `Makefile`, `scripts/create_vector_index.py`, `src/fancryrag/config.py`.

5. **Local secrets hygiene**
   - **Issue**: `.env.local` is present but not explicitly ignored in `.gitignore`.
   - **Impact**: accidental secrets commit risk.
   - **Suggestion**: add `.env.local` to `.gitignore` and document safe local overrides.
   - **Files**: `.gitignore`, `.env.example`.

### Low Priority / Quality of Life

6. **MCP request validation**
   - **Issue**: runtime tools accept raw inputs; validation for `top_k` and ratio is implicit.
   - **Suggestion**: add explicit schema validation (min/max bounds) at MCP tool definition.
   - **Files**: `src/fancryrag/mcp/runtime.py`.

7. **Telemetry and tracing**
   - **Issue**: logs are structured but lack request IDs/correlation across components.
   - **Suggestion**: add per‑request IDs in MCP handlers and propagate to logs (and optionally to Neo4j writes).
   - **Files**: `src/fancryrag/mcp/runtime.py`, `servers/mcp_hybrid_google.py`.

8. **Configuration ergonomics**
   - **Issue**: some env vars (e.g., `NEO4J_DATABASE`, `MCP_BASE_URL`) are required even when defaults are sufficient locally.
   - **Suggestion**: allow safe defaults or derive from URI/host for local dev to reduce config friction.
   - **Files**: `src/fancryrag/config.py`.

## Suggested Next Steps

1. Decide the canonical ingestion pipeline path (`SimpleKGPipeline` vs `PipelineRunner`) and remove the alternate path or document it as experimental.
2. Fix package naming and explicitly scope exported packages to prevent collisions.
3. Commit `uv.lock` (or update Dockerfile to generate it) to ensure Docker builds are reproducible.
4. Align index creation with embedder model dimensions and config values.

## Open Questions

- Which ingestion path do you want to treat as the default (SimpleKGPipeline wrapper vs config‑driven PipelineRunner)?
- Should we keep both `fancyrag` and `fancryrag` as public imports, or consolidate to one and add a compatibility shim?
- Do you want strict reproducibility (commit `uv.lock`) or flexible installs (generate lock during build)?
