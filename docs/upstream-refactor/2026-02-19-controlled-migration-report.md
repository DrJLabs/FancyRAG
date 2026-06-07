# Controlled Upstream Migration Report

Date: 2026-02-19
Branch: `chore/upstream-controlled-migration`
Base: `fork` (tracking `upstream/main`)

## Objective

Rebase project work onto the latest upstream code while preserving only the local functionality that remains valuable, and avoid re-introducing obsolete or already-upstreamed custom behavior.

## Evaluation Method

1. Compared repository snapshots `fork..main` at file level and area counts.
2. Reviewed custom runtime/app code added on `main` (`src/fancyrag`, `src/cli`, `src/config`, scripts, compose, server).
3. Cross-checked recent upstream additions on `upstream/main` for overlap/redundancy.
4. Classified custom deltas into:
- `KEEP` (project-specific value)
- `DROP` (process/docs noise or overrides that weaken upstream baseline)
- `KEEP but DEPRIORITIZE` (still useful, but now partly redundant upstream)

## Findings

### High-level divergence

- `main` and `fork` are unrelated-history branches with large structural divergence.
- Main risk in direct history rewrite: immediate conflicts from first replay commit (`AGENTS.md`, `.gitignore`), confirming controlled file-level migration is safer than root rebase.

### What was worth migrating from `main`

`KEEP` (migrated)
- Local stack operations and runtime:
- `Dockerfile`
- `docker-compose.yml`
- `docker-compose.smoke.yml`
- `docker-compose.neo4j-qdrant.yml`
- `Makefile`
- `.env.example`
- Application/runtime layer (additive; no upstream core overwrite):
- `src/fancyrag/**`
- `src/fancryrag/__init__.py` (compat namespace shim)
- `src/cli/**`
- `src/config/**`
- `src/_compat/**`
- MCP server entrypoint and query:
- `servers/mcp_hybrid_google.py`
- `queries/hybrid_retrieval.cypher`
- Operational scripts:
- `scripts/**`
- Pipeline/tooling assets:
- `pipelines/kg_ingest.yaml`
- `tools/run_pipeline.py`
- `tools/discover_classes.py`
- Supporting CI workflows for local-stack ops:
- `.github/workflows/container-scan.yml`
- `.github/workflows/fulltext-index-smoke.yml`
- `.github/workflows/local-stack-smoke.yml`
- `.github/workflows/openai-allowlist-audit.yml`
- `.github/workflows/openai-probe-smoke.yml`
- Selected tests covering migrated local functionality:
- local stack, scripts, CLI, `fancyrag`, MCP runtime, smoke fixtures

`DROP` (intentionally not migrated)
- `.bmad-core/**`, `.bmad-infrastructure-devops/**`, AI/planning process artifacts
- PR/issue template reshuffles not required for runtime
- Main-branch deletions/replacements of upstream package sources (`src/neo4j_graphrag/**`)
- Main branch broad doc replacement set that would overwrite upstream docs baseline

`KEEP but DEPRIORITIZE` (migrated as wrappers, not upstream patches)
- Structured semantic output wrapper code:
- `src/fancyrag/kg/semantic_llm.py`
- `src/fancyrag/kg/structured_output.py`
- Rationale: upstream already added structured-output support; local wrappers still add project-specific behavior (fallback and local controls), but these should be revisited for simplification.

## Upstream redundancy analysis

Recent upstream commits already cover portions of prior custom work:

- `57f80e6` Add support for structured output (`LLMInterfaceV2` / extractor)
- `02d4d82` Enable structured output for schema extraction
- `9f48cc3` Async embedding support improvements
- `69f0241` Rate-limit retry behavior improvements
- `8532940` Vertex AI init fix

Conclusion:
- We should not port old core-library customizations into `src/neo4j_graphrag`.
- Keep local value as additive orchestration/runtime modules on top of upstream.
- Treat custom semantic wrappers as temporary compatibility layer, not core differentiation.

## Concrete End-to-End Migration Plan

### Goals

1. Preserve upstream `neo4j_graphrag` as source of truth on this branch.
2. Reintroduce only local application/runtime capabilities needed for FancyRAG operations.
3. Keep dependency and lock state consistent for reproducible `uv` and Docker flows.
4. Leave migration auditable with explicit keep/drop rationale.

### Tasks

1. Create migration branch from pristine upstream mirror.
- Definition of done:
- Branch created from `fork`; no direct rewrite of `main`.

2. Import additive local code/runtime assets from `main`.
- Definition of done:
- Local runtime packages, server, scripts, compose files, and query templates present.
- Upstream `src/neo4j_graphrag/**` untouched.

3. Update packaging/install surface for additive modules.
- Definition of done:
- `pyproject.toml` includes required runtime dependencies and packages.
- FancyRAG modules are installable without replacing upstream core package.

4. Refresh lockfile for reproducibility.
- Definition of done:
- `uv.lock` regenerated successfully after dependency changes.

5. Capture migration rationale and implementation state in repository docs.
- Definition of done:
- This report committed with keep/drop/redundancy analysis and task outcomes.

## Implementation Status

- Task 1: Completed
- Task 2: Completed
- Task 3: Completed
- Task 4: Completed
- Task 5: Completed

## Final State Summary

The branch now uses latest upstream as baseline and layers in project-specific FancyRAG runtime/tooling as additive code. The migration intentionally avoids reapplying obsolete upstream-overlapping core modifications, reducing maintenance burden while preserving required local stack functionality.
