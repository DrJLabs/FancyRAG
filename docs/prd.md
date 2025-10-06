# Neo4j GraphRAG Brownfield Enhancement PRD

## Intro Project Analysis and Context

### Existing Project Overview
**Analysis Source:** Document-project output available at `docs/brownfield-architecture.md` (generated 2025-10-06). This document captures the verified runtime topology, module boundaries, and operational guardrails for the current Neo4j + Qdrant stack.

**Current Project State:**
- Local Docker Compose stack provisions Neo4j 5.26.12 (APOC) and Qdrant 1.15.4 with health-gated startup and persistent bind mounts (`docker-compose.neo4j-qdrant.yml`).
- Python 3.12 toolchain orchestrates ingestion via the FancyRAG package (`src/fancyrag/`) and CLI entry points in `scripts/`. Core modules now isolate caching splitters, QA evaluation/reporting, and Neo4j query helpers (Stories 4.3–4.6 delivered 2025-10-04).
- Structured artifacts (vector index, ingestion logs, QA reports, Qdrant exports, retrieval traces) land under `artifacts/local_stack/` and power Story 4.5 QA dashboards.
- Tests cover unit suites for splitters/QA/DB helpers plus `tests/integration/local_stack/test_minimal_path_smoke.py`, which drives the end-to-end “minimal path” workflow through Docker and OpenAI.

### Available Documentation Analysis
Using existing project analysis from document-project output.
- [x] Tech Stack Documentation (`docs/architecture/tech-stack.md`)
- [x] Source Tree/Architecture (`docs/brownfield-architecture.md`, `docs/architecture/source-tree.md`)
- [x] Coding Standards (`docs/architecture/coding-standards.md`)
- [x] API Documentation (Neo4j/Qdrant script docs within `docs/architecture/projects/*`)
- [x] External API Documentation (OpenAI usage captured in `docs/architecture/overview.md`)
- [ ] UX/UI Guidelines (CLI-only scope; not applicable)
- [x] Technical Debt Documentation (`docs/brownfield-architecture.md#Technical Debt and Constraints`)
- Other: `docs/bmad/focused-epics/kg-build-refactor/`

### Enhancement Scope Definition
- **Enhancement Type:**
  - [ ] New Feature Addition
  - [x] Major Feature Modification
  - [x] Integration with New Systems
  - [x] Performance/Scalability Improvements
  - [ ] UI/UX Overhaul
  - [x] Technology Stack Upgrade
  - [x] Bug Fix and Stability Improvements
  - [ ] Other: _n/a_
- **Enhancement Description:** Extend the Neo4j GraphRAG minimal-path service so the modular FancyRAG ingestion pipeline, automation scripts, and retrieval tooling operate as a cohesive, production-ready service with telemetry, configuration, and managed-environment parity built on the newly refactored modules.
- **Impact Assessment:**
  - [ ] Minimal Impact
  - [ ] Moderate Impact
  - [x] Significant Impact (substantial existing code changes)
  - [x] Major Impact (architectural changes required)

### Goals and Background Context
- Establish a hardened ingestion service surface that packages Docker bootstrapping, ingestion pipeline execution, and retrieval checks behind reproducible commands.
- Deliver telemetry and QA evidence that downstream evaluators (QA/Test, PM) can trust without manual log wrangling.
- Preserve compatibility with local developer workflows while preparing configuration toggles for managed Neo4j/Qdrant deployments.

Earlier work (PRD v0.3) proved the minimal path by containerising Neo4j/Qdrant and scripting ingestion. The refactor (Stories 4.3–4.6) decomposed the pipeline into reusable modules. The next wave must operationalise these modules: automate stack lifecycle, unify telemetry artefacts, and codify configuration/compatibility requirements so service operators can run or extend the stack without re-discovering architecture details.

### Change Log
| Change | Date | Version | Description | Author |
|--------|------|---------|-------------|--------|
| Harden FancyRAG service planning (this document) | 2025-10-06 | 0.4 | Regenerated PRD using brownfield template, aligned with `docs/brownfield-architecture.md`, and outlined service hardening epic/stories. | Codex CLI |
| Refactor roadmap + planning artefacts | 2025-10-02 | 0.3 | Added refactor roadmap and linked planning artefacts. | Codex CLI |
| Local stack focus | 2025-09-28 | 0.2 | Narrowed scope to local Docker stack + scripted minimal path. | Codex CLI |
| Initial BMAD-format PRD | 2025-09-24 | 0.1 | Rebuilt PRD in BMAD format; aligned goals, requirements, epics. | Codex CLI |

## Requirements
These requirements are grounded in the validated system analysis above. Please review and confirm they align with your understanding of the current stack.

### Functional Requirements
- **FR1:** Provide a consolidated CLI workflow (`scripts/bootstrap.sh`, `scripts/check_local_stack.sh`, orchestrated make/venv targets) that bootstraps Python deps, configures environment variables, and validates Docker health before ingestion begins.
- **FR2:** Extend the FancyRAG pipeline configuration so operators can select chunking presets, override splitter parameters, and toggle semantic enrichment without modifying source code, while maintaining ingestion parity.
- **FR3:** Guarantee ingestion runs emit versioned, sanitized telemetry bundles (structured logs, QA Markdown/JSON, counts, durations) stored under `artifacts/` with deterministic naming so QA and PM checkpoints can consume them automatically, backed by a documented manifest schema shared with downstream consumers.
- **FR4:** Ensure Neo4j vector index provisioning, Qdrant export, and retrieval scripts support both local Compose endpoints and managed-service endpoints via `.env` overrides while preserving provenance metadata (chunk IDs, checksums, git commits).
- **FR5:** Maintain up-to-date documentation shards (architecture, source tree, operator playbooks) that reflect the refactored modules and provide runbooks for the minimal path and future managed rollouts.

### Non-Functional Requirements
- **NFR1:** End-to-end minimal path (bootstrap → compose up → ingestion → export → retrieval smoke) must complete within 45 minutes on a 16 GB RAM developer laptop.
- **NFR2:** Pipeline memory utilisation must remain within 1.5× current peak (measured during Story 4.6) when semantic enrichment is disabled; semantic mode may increase usage but must be documented with thresholds.
- **NFR3:** All scripts must exit non-zero on failure and include actionable error messaging with remediation hints (OpenAI retry, Neo4j import dependencies, Docker status).
- **NFR4:** Telemetry artefacts must redact secrets and absolute host paths, following `cli.sanitizer.scrub_object` policies, and remain reproducible across reruns (timestamp suffix collision handling).

### Compatibility Requirements
- **CR1:** CLI interfaces (`scripts/*.py`, `scripts/*.sh`) must remain backward compatible with Story 2.x automation (no flag removals; new flags default to existing behaviour).
- **CR2:** Neo4j schema (Document, Chunk, relationships, vector index) must remain intact; ingestion upgrades cannot drop or repurpose labels/properties without a migration plan.
- **CR3:** Qdrant payload schema must retain current field names (`chunk_id`, `chunk_uid`, `source_path`, `relative_path`, `checksum`) so existing dashboards and QA scripts continue to function.
- **CR4:** OpenAI integration must continue to support `gpt-4.1-mini` primary and `gpt-4o-mini` fallback models, respecting retry/backoff guardrails from `config/settings.py`.

## Technical Constraints and Integration Requirements

### Existing Technology Stack
**Languages:** Python 3.12, Bash, YAML (Compose/CI).

**Frameworks:** FancyRAG (bespoke package), `neo4j_graphrag`, `structlog`, `pytest`, `qdrant-client`.

**Database:** Neo4j 5.26.12 (vector index on `Chunk.embedding`), Qdrant 1.15.4.

**Infrastructure:** Docker Compose (`graphrag-local` network, bind mounts under `.data/`), Git-managed repo with BMAD workflows, OpenAI API connectivity.

**External Dependencies:** OpenAI API (chat + embeddings), Neo4j Bolt/HTTP endpoints, Qdrant HTTP/gRPC endpoints, optional pandas/numpy extras for driver functionality.

### Integration Approach
**Database Integration Strategy:** Continue using shared helper module (`src/fancyrag/db/neo4j_queries.py`) for resets, provenance, and rollback; add database selector support for managed clusters without altering default local database usage.

**API Integration Strategy:** Parameterise CLI scripts via `.env` and environment variables so switching between local and managed endpoints requires no code change. Preserve ingest run keys for cross-store tracing.

**Frontend Integration Strategy:** Not applicable—CLI-centric product. Maintain Markdown/JSON artefact output for consumers.

**Testing Integration Strategy:** Expand `tests/integration/local_stack/test_minimal_path_smoke.py` coverage to include semantic-enabled runs and managed-endpoint dry runs (mocked), while keeping unit suites isolated via fixtures and cached path controls.

### Code Organization and Standards
**File Structure Approach:** Keep CLI entry points thin (`scripts/*.py` delegating into `src/fancyrag/`), maintain module boundaries documented in `docs/brownfield-architecture.md`, and add dedicated telemetry/config modules as needed.

**Naming Conventions:** Follow established snake_case modules, PascalCase classes, and `kg_build` naming. New flags/artefacts should use kebab-case CLI options and snake_case JSON keys.

**Coding Standards:** Adhere to `docs/architecture/coding-standards.md` (type hints, Ruff/Rusty lints, test coverage) and ensure new modules stay within 200–400 LOC guardrails.

**Documentation Standards:** Update sharded docs under `docs/prd/` and `docs/architecture/` with concise diffs, include change logs, and cross-link to artefacts referenced in acceptance criteria.

### Deployment and Operations
**Build Process Integration:** `scripts/bootstrap.sh --verify` remains canonical for environment prep; augment with Make targets (`make stack-up`, `make minimal-path`) to orchestrate sequential commands.

**Deployment Strategy:** Local-first via Docker Compose; future managed deployment toggled via environment overrides and documented manual steps. Provide rollback instructions for ingestion resets (`--reset-database`, Qdrant collection recreation).

**Monitoring and Logging:** Continue structured logging via `_compat.structlog`; add optional Prometheus client hook (already in `requirements-dev.txt`) for future exporters; ensure logs capture component, duration, and status tags.

**Configuration Management:** Centralise environment defaults in `.env` (template) and `src/config/settings.py`; document required overrides for managed hosts and maintain sample `.env` entries for QA/CI usage.

### Risk Assessment and Mitigation
**Technical Risks:** Optional dependencies (neo4j driver extras, openai) can drift; mitigate via locked requirements and bootstrap diagnostics. Semantic enrichment increases OpenAI cost and latency; default disabled with documentation.

**Integration Risks:** Managed service endpoints may introduce TLS/auth variations; plan staging dry runs and add config validation to scripts before connecting. Ensure Qdrant collection schemas stay compatible when migrating.

**Deployment Risks:** Docker volume growth can exhaust disk; document cleanup commands. Failing semantic runs must roll back partial graph writes (handled via ingest run keys).

**Mitigation Strategies:** Maintain smoke tests, add config validation steps, keep rollback scripts validated via integration tests, and monitor dependency upgrades through pinned lockfiles and CI checks.

## Epic and Story Structure
Based on the current module decomposition and outstanding service work, this enhancement should proceed as a **single epic** focused on service hardening. The stories share the same pipelines and must land in a controlled sequence to avoid breaking delivered functionality. Please confirm this matches your understanding before execution.

**Epic Structure Decision:** Single epic (“Local GraphRAG Service Hardening”) with rationale: modular components already shipped (Stories 4.3–4.6); remaining work is tightly coupled service/operations improvements best coordinated under one epic to manage sequencing and compatibility.

## Epic 5: FancyRAG Service Hardening
**Epic Goal:** Operationalise the refactored FancyRAG pipeline into a reliable service workflow that anyone can run end-to-end with built-in telemetry, configuration safeguards, and rollback procedures.

**Integration Requirements:** Preserve CLI compatibility, maintain provenance metadata across Neo4j and Qdrant, ensure QA artefacts remain consumable, and keep Docker stack commands idempotent.

### Story 5.1 Decompose the Pipeline Orchestrator
As a FancyRAG maintainer,
I want the monolithic pipeline orchestration split into composable functions and lightweight classes,
so that each ingestion phase can be tested and iterated independently without re-reading global state.

#### Acceptance Criteria
1. Extract helper functions/classes for `resolve_settings()`, `discover_sources()`, `build_clients()`, `ingest_source()`, `run_semantic_enrichment()`, and `perform_qa()`, leaving `run_pipeline()` as a thin coordinator.
2. Each helper accepts explicit parameters (no direct `os.environ` usage) and returns typed results that can be unit tested in isolation.
3. Update unit tests to cover each helper (mocks/stubs acceptable) and extend integration smoke to ensure behaviour matches the previous orchestration.
4. Document the new call graph in `docs/architecture/projects/fancyrag-kg-build-refactor.md` and ensure story shards reference the extracted functions.

#### Integration Verification
- IV1: Existing CLI entry points (`scripts/kg_build.py` and `fancyrag.cli.kg_build_main`) continue to succeed with no flag changes.
- IV2: Ingestion, semantic enrichment, and QA outputs remain identical for the smoke dataset.
- IV3: Unit tests demonstrate helpers can be executed with mocked clients/settings without hitting the network.

### Story 5.2 Centralise Typed Settings
As a configuration steward,
I want all service credentials and tunables represented by typed settings classes,
so that operators configure FancyRAG through a single validated surface instead of scattered environment lookups.

#### Acceptance Criteria
1. Introduce `FancyRAGSettings` with nested `OpenAISettings`, `Neo4jSettings`, and `QdrantSettings` (Pydantic `BaseSettings` or equivalent) loading from `.env`.
2. Replace direct `ensure_env()` / `os.environ[...]` usage in the pipeline and scripts with the typed settings objects.
3. Update documentation (`README.md`, `.env.example`, `docs/brownfield-architecture.md`) to describe the consolidated configuration surface.
4. Add unit tests validating settings fallback, overrides, and error paths (missing values, invalid URLs).
5. Publish an `.env` migration checklist (docs/brownfield-architecture.md addendum + story shard) mapping legacy variables to the new settings, including upgrade commands and rollback guidance for existing operators.

#### Integration Verification
- IV1: CLI scripts honour existing environment variables when present and emit actionable errors when required values are missing.
- IV2: Managed-endpoint overrides (hosted Neo4j/Qdrant) can be supplied through the new settings without code edits.
- IV3: Telemetry/logging confirm settings are initialised exactly once per run.
- IV4: Operators following the documented `.env` migration steps can upgrade an existing configuration without editing source code or breaking the minimal-path smoke.

### Story 5.3 Automate Stack Lifecycle Workflows
As an operations engineer,
I want a single automation surface that bootstraps, ingests, validates, and rolls back the stack,
so that anyone can run the FancyRAG service workflow reproducibly without hand-crafted command sequences.

#### Acceptance Criteria
1. Provide a consolidated automation entry point (`make` target or wrapper script) that chains bootstrap, ingestion, export, evaluation, and teardown steps with clear logging.
2. Implement guarded rollback/cleanup commands that restore Docker services, Neo4j/Qdrant data, and artifacts when runs fail or are cancelled.
3. Wire automation to accept configurable presets (dataset path, chunking profile, telemetry toggle) while defaulting to the smoke dataset.
4. Document the workflow in `docs/brownfield-architecture.md` (or successor shard) and update developer onboarding instructions to reference the new command surface.
5. Extend integration smoke coverage (e.g., `tests/integration/local_stack/test_minimal_path_smoke.py`) to exercise the automation entry point end-to-end.

#### Integration Verification
- IV1: A single command executes the full bootstrap → ingest → export → evaluation → teardown path without manual intervention.
- IV2: Rollback/cleanup commands leave Docker services and data volumes in a known-good state for subsequent runs.
- IV3: CI pipelines adopt the automation entry point, demonstrating parity with current manual scripts.

### Story 5.4 Harden Pipeline Configuration Presets
As a pipeline maintainer,
I want ingestion presets (chunking, caching, semantic enrichment, evaluation) defined as versioned configuration profiles,
so that operators can choose the right balance of fidelity, cost, and latency without patching Python code.

#### Acceptance Criteria
1. Introduce preset definitions (YAML/TOML/JSON) that declare chunk sizing, cache behaviour, enrichment toggles, and evaluation settings, and load them via the typed settings contract.
2. Provide guardrails that validate preset compatibility (e.g., ensuring cache dimensions match vector index size) and surface actionable errors when constraints are violated.
3. Update CLI flags and automation workflows to select presets explicitly, defaulting to a documented “smoke” profile.
4. Refresh project documentation (`docs/prd.md`, `docs/brownfield-architecture.md`, and newcomer guides) to describe available presets, override workflows, and migration notes for operators.
5. Add regression tests covering preset resolution, validation failures, and runtime overrides.

#### Integration Verification
- IV1: Operators can switch between presets (smoke, full, enrichment-heavy, evaluation-lite) without editing Python modules.
- IV2: Preset validation catches incompatible combinations before the pipeline runs and returns descriptive error messages.
- IV3: Automation and CI demonstrate successful runs using at least two presets.

### Story 5.5 Integrate RAG Evaluation Harness
As a quality engineer,
I want automated retrieval and faithfulness metrics produced alongside ingestion QA,
so that regressions in answer quality are caught before release.

#### Acceptance Criteria
1. Adopt a RAG evaluation framework (e.g., RAGAS) and define baseline metrics (precision, recall, faithfulness, correctness).
2. Sample queries (configurable) after each ingestion run and record evaluation results in a versioned scorecard artefact under `artifacts/local_stack/`.
3. Integrate evaluation thresholds into CI (fail the pipeline when metrics drop below configured floors).
4. Update QA gates and documentation to reference the new scorecard outputs.

#### Integration Verification
- IV1: Evaluation runs can be executed locally with a seeded sample dataset.
- IV2: CI pipeline demonstrates failing behaviour when metrics breach thresholds.
- IV3: Documentation instructs operators on extending the query set and adjusting thresholds.

### Story 5.6 Instrument Observability with OpenTelemetry
As an SRE,
I want ingestion stages to emit structured traces and metrics,
so that performance bottlenecks and errors can be diagnosed quickly.

#### Acceptance Criteria
1. Wrap major pipeline stages (settings load, source discovery, embeddings, graph writes, semantic enrichment, QA, evaluation) in OpenTelemetry spans with correlated logs.
2. Expose OTLP exporter configuration via the new settings classes; provide a default console exporter for local testing.
3. Emit key metrics (latency, throughput, cache hit ratio) and document how to view them (e.g., Jaeger/Grafana instructions).
4. Add automated tests or smoke checks ensuring spans are emitted when telemetry is enabled.

#### Integration Verification
- IV1: Local runs can output traces/metrics to a developer-friendly backend (console/Jaeger).
- IV2: Telemetry can be disabled without impacting existing logging.
- IV3: Documentation includes troubleshooting guidance and example dashboards for the emitted signals.

---
Please review the revised story sequence and confirm before invoking the Scrum Master to draft detailed implementation tasks.
