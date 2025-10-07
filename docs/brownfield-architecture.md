# Neo4j GraphRAG Brownfield Architecture

## Introduction
- **Scope:** Captures the current state of the local Neo4j + Qdrant evaluation stack described in `docs/prd.md` (version 0.3, updated 2025-10-02). Focus is on the Docker-first "minimal path" workflow that ingests documents into Neo4j, mirrors embeddings into Qdrant, and serves retrieval via the GraphRAG retriever.
- **Audience:** Agents implementing Stories 4.x (caching, QA, query helpers) and future service enhancements. Target skill level: senior developer familiar with Python data tooling.
- **Key Guiding Principle:** Reflect reality. All behaviors below were verified against the code and scripts that ship in this repository as of 2025-10-06.

## Runtime Components
- **Docker Compose stack** (`docker-compose.neo4j-qdrant.yml`)
  - `neo4j` service: Neo4j 5.26.12 with APOC enabled, binds `.data/neo4j/{data,logs,import}` via host mounts and advertises Bolt/HTTP using `NEO4J_*` env overrides.
  - `qdrant` service: Qdrant 1.15.4 persisting to `.data/qdrant/storage`; health check polls `/readyz` to guard script execution.
  - `smoke-tests` service: Python 3.12 container wired for CI; runs against the live containers once both health checks pass.
- **Python orchestration layer** (`src/fancyrag/` and `scripts/`)
  - CLI entry points execute inside the developer host (recommended via `.venv` provisioned by `scripts/bootstrap.sh`).
  - Shared utilities (_compat structlog shim, sanitized logging, config guards) live alongside the FancyRAG package and are re-used by every script.
- **Artifacts and volumes**
  - Structured JSON/Markdown artifacts land under `artifacts/local_stack/` (vector index, KG build, Qdrant export/query).
  - Docker bind mounts keep graph data and logs between runs; cleaning the stack requires explicit `scripts/check_local_stack.sh --down --destroy-volumes`.

## Code and Module Layout
- **Scripts** (`scripts/`)
  - `create_vector_index.py`: Idempotently provisions `chunks_vec` using `neo4j_graphrag.indexes` with retry/backoff and structured logging.
  - `kg_build.py`: Thin wrapper that delegates to `fancyrag.cli.kg_build_main.main`.
  - `export_to_qdrant.py`: Streams chunk embeddings out of Neo4j and upserts them to Qdrant with batch sanity checks.
  - `ask_qdrant.py`: Embeds a question, performs hybrid retrieval through `QdrantNeo4jRetriever`, and prints sanitized matches.
  - `check_local_stack.sh`: Manages Docker lifecycle (config/up/status/down), handles health polling, and ensures necessary ports respond before scripts run.
  - Support scripts (`bootstrap.sh`, `audit_openai_allowlist.py`, `check_docs.py`) enforce developer environment and documentation quality.
- **FancyRAG package** (`src/fancyrag/`)
  - `kg/pipeline.py`: Core ingestion orchestration including chunk discovery, OpenAI calls, Neo4j writes, semantic enrichment hooks, rollback helpers, and QA gating.
  - `splitters/caching_fixed_size.py`: Caching wrapper around GraphRAG's fixed-size splitter with scoped memoization to keep chunk IDs stable across retries.
  - `qa/evaluator.py` & `qa/report.py`: Compute ingestion QA metrics (missing embeddings, orphan chunks, checksum mismatches, semantic stats) and emit Markdown + JSON reports.
  - `db/neo4j_queries.py`: Centralizes Cypher used for provenance, cleanup, and metrics, including rollback semantics keyed by ingestion run UUIDs.
  - `config/schema.py`: Loads the project KG schema from `scripts/config/kg_schema.json`, falling back to stub types when `neo4j_graphrag` is absent.
  - `utils/env.py` / `utils/paths.py`: Resolve `.env` files, enforce required environment variables, and map paths relative to the repo root for provenance logging.
- **CLI utilities** (`src/cli/`)
  - `openai_client.py`: Shared OpenAI client with telemetry, retry/backoff, deterministic fallback model selection, and sanitized logging.
  - `utils.py`: Guards embedding dimensionality (`ensure_embedding_dimensions`) to keep Neo4j vector index compatibility.
  - `telemetry.py`, `sanitizer.py`, `diagnostics.py`: Provide metrics emission, masking helpers, and environment diagnostics consumed across scripts.

## Data Graph and Storage Model
- **Neo4j Graph**
  - Documents and chunks are stored as `Document` and `Chunk` nodes. Each ingest assigns a deterministic checksum (`hashlib.sha256`) and attaches provenance (relative path, git commit) via `_ensure_document_relationships`.
  - Relationships: `(:Document)-[:HAS_CHUNK]->(:Chunk)` created for every chunk. The ingestion pipeline ensures orphan chunks are deleted during rollback.
  - Semantic enrichment (optional) writes additional entity/relationship nodes tagged with `ingest_run_key`; QA thresholds default to zero to fail the run on any anomaly.
- **Vector Index**
  - `scripts/create_vector_index.py` targets label `Chunk`, property `embedding`, similarity `cosine`, with default 1536 dimensions. Validation mismatches raise `VectorIndexMismatchError` prompting manual remediation.
- **Qdrant Collection**
  - Export script ensures `chunks_main` collection exists (or recreates it with `--recreate-collection`). Point IDs default to the chunk's numeric index, falling back to UID strings when absent.
  - Payload mirrors Neo4j provenance fields (source path, git commit, checksum) for cross-store validation.
- **Artifacts**
  - Per-run JSON logs live in `artifacts/local_stack/{create_vector_index,kg_build,export_to_qdrant,ask_qdrant}.json`.
  - QA reports are written to `artifacts/ingestion/<timestamp>/`, including Markdown summaries used by Story 4.5 QA automation.

## Pipeline Execution Flow (`fancyrag.kg.pipeline.run_pipeline`)
1. **Pre-flight**
   - Collects chunking parameters from profile presets (`PROFILE_PRESETS`) or CLI overrides.
   - Enforces presence of `OPENAI_API_KEY`, Neo4j credentials, and optionally `NEO4J_DATABASE`.
2. **Source Resolution**
   - Single-file mode reads `options.source` (default `docs/samples/pilot.txt`). Directory mode (`--source-dir`) expands include globs per preset and skips binary/empty files.
   - Every source is assigned a checksum and stored relative to repo root for downstream logging.
3. **Client Construction**
   - Loads `OpenAISettings` (actor `kg_build`) which restricts chat models to `gpt-4.1-mini` with optional fallback to `gpt-4o-mini`; embedding model defaults to `text-embedding-3-small`.
   - Instantiates `SharedOpenAIClient`, `SharedOpenAIEmbedder`, and `SharedOpenAILLM` for consistent telemetry and retry handling.
   - Builds a scoped `CachingFixedSizeSplitter` with chunk size/overlap derived from presets.
4. **Neo4j Execution Loop**
   - Optionally resets the database on first iteration when `--reset-database` is set.
   - Executes `SimpleKGPipeline.run_async` (from `neo4j_graphrag`) wrapped in structured logging. Missing dependencies raise a guarded RuntimeError with installation guidance.
   - Writes document ↔ chunk relationships and collects chunk metadata (`ChunkMetadata`) for QA and export.
5. **QA & Logging**
   - Builds QA source payloads (`QaSourceRecord`) and evaluates thresholds via `IngestionQaEvaluator`.
   - Writes sanitized run logs capturing counts, chunk stats, OpenAI model usage, and ingest run keys for rollback.
6. **Semantic Enrichment (optional)**
   - When `--enable-semantic` is passed, `LLMEntityRelationExtractor` executes with bounded concurrency (`--semantic-max-concurrency`, default 5). Results feed `semantic_totals` and QA semantic checks.
7. **Result**
   - Returns a dict containing run identifiers, QA status, metrics, and artifact paths. CLI wrappers relay failures with actionable stderr messages while logging full context through structlog.

## Testing and Automation
- **Integration smoke test** (`tests/integration/local_stack/test_minimal_path_smoke.py`)
  - Spins up the compose stack, runs the minimal script sequence (`create_vector_index.py` → `kg_build.py` → `export_to_qdrant.py` → `ask_qdrant.py`), then tears everything down. Skips automatically when Docker or API keys are missing.
- **Unit coverage**
  - `tests/unit/` exercises splitters, OpenAI client guardrails, QA evaluator, schema resolution fallbacks, and vector index utilities.
  - `tests/fixtures/` provides schema and embedding stubs to allow offline test execution.
- **CLI Diagnostics**
  - `scripts/bootstrap.sh --verify` captures interpreter metadata to `artifacts/diagnostics/versions.json` (when run) and ensures `neo4j-graphrag` extras install cleanly.
- **QA Gate YMLs** (`docs/qa/gates/*.yml`)
  - Stories 2.5, 4.3, 4.5 encode post-development QA expectations and should be checked when assessing readiness.

## Configuration and Secrets
- `.env` (developer-local) supplies OpenAI, Neo4j, and Qdrant credentials. `FancyRAGSettings` (defined in `src/config/settings.py`) loads these values via Pydantic, emitting actionable `ValueError` messages when required keys are missing or malformed before any network calls execute. Legacy helpers such as `fancyrag.utils.env.ensure_env` now delegate to the typed surface and fall back to raw environment reads only when validation cannot run (e.g., partial test fixtures).
- `FancyRAGSettings.openai` enforces the chat-model allowlist (`gpt-4.1-mini` baseline, optional `gpt-4o-mini` fallback), validates embedding overrides, and records base URL overrides with masked logging. Override knobs remain the same (`OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_MAX_ATTEMPTS`, etc.) but are now typed and cached per actor.
- `FancyRAGSettings.neo4j` and `FancyRAGSettings.qdrant` normalise connection details (URI validation, optional database names, API keys). Docker defaults still map to `bolt://localhost:7687` and `http://localhost:6333`; managed deployments must override the same environment keys, which are now surfaced through typed accessors.

## Observability and Telemetry
- Structlog instrumentation (backed by `_compat.structlog`) standardizes log fields across scripts. JSON artifacts are scrubbed via `cli.sanitizer.scrub_object` to remove credentials.
- OpenAI telemetry (`cli.telemetry.OpenAIMetrics`) captures latency, token counts, retry details, and fallback usage for later export.
- QA Markdown reports highlight anomalies (missing embeddings, orphan chunks, semantic failures). These reports feed Story 4.5's evaluator dashboards (`docs/stories/4.5.qa-report.md`).
- Docker health checks ensure automation waits for ready services; `check_local_stack.sh --status --wait` combines Compose status, HTTP probing, and Bolt connectivity tests.

## Technical Debt and Constraints (Current Reality)
- **Dependency sensitivity:** `neo4j_graphrag` optional modules (semantic extractor, writer) must be installed; otherwise the pipeline raises explicit RuntimeErrors. This is acceptable for local smoke but blocks semantic enrichment until dependencies are satisfied.
- **Hard-coded model allowlist:** Adding new OpenAI models requires code changes plus policy review; short term workaround is limited to gpt-4.1-mini ↔ gpt-4o-mini fallback.
- **Manual data cleanup:** `.data/neo4j` and `.data/qdrant` can accumulate gigabytes across runs. Developers must use `check_local_stack.sh --down --destroy-volumes` or delete directories manually.
- **Retry surfaces:** Vector index creation retries up to three times with fixed backoff but does not yet randomize jitter; repeated cluster contention could still fail CI runs.
- **Semantic QA gaps:** When semantic enrichment is enabled the evaluator only checks counts, not content accuracy. Future work may need richer validation or human-in-the-loop review.
- **Qdrant schema drift:** Export script recreates collections but does not version payload schema; downstream consumers should monitor for field additions.

## Maintainability Improvement Plan (Epic 5 – October 2025)
- **Pipeline decomposition:** Extract helper functions/classes (`resolve_settings`, `discover_sources`, `build_clients`, `ingest_source`, `run_semantic_enrichment`, `perform_qa`) so `run_pipeline()` becomes a thin coordinator with unit-testable pieces.
- **Typed settings surface:** Introduce a consolidated `FancyRAGSettings` (Pydantic) wrapping OpenAI, Neo4j, and Qdrant configuration, replacing direct environment lookups across CLI and pipeline modules.
- **Adapter interfaces:** Define interfaces for embeddings, vector storage, KG writing, LLM, and semantic extraction; implement adapters backed by GraphRAG/OpenAI so orchestration depends on abstractions.
- **Caching layer:** Persist embeddings and LLM responses keyed by `(model, checksum)` with TTL/version controls to reduce cost and latency on reingests; expose telemetry for hit/miss ratios.
- **Automated RAG evaluation:** Integrate a harness (e.g., RAGAS) generating precision/recall/faithfulness scorecards after each ingest and enforcing thresholds in CI.
- **Observability:** Instrument ingestion stages with OpenTelemetry spans and metrics, configurable exporters, and documented dashboards for diagnosing performance issues.

## Upcoming Enhancements & Impact (from PRD FR1–FR5)
- **FR1 – Compose durability:** Work continues in `docker-compose.neo4j-qdrant.yml` and `scripts/check_local_stack.sh`; future story changes should coordinate with `.data/` volume expectations and health checks.
- **FR2 – Python environment:** `scripts/bootstrap.sh`, `requirements.lock`, and `src/config/settings.py` will absorb dependency upgrades to keep GraphRAG extras aligned with OpenAI + Neo4j client versions.
- **FR3 – Vector index automation:** Enhancements touch `scripts/create_vector_index.py` and `tests/unit/scripts/test_create_vector_index.py`. Respect the retry/backoff guardrails and structured logging contract.
- **FR4 – KG builder modularization:** Stories updating `src/fancyrag/kg/pipeline.py`, `src/fancyrag/splitters/`, and `src/fancyrag/qa/` must preserve provenance logging (chunk checksums, git commit) and QA gating defaults.
- **FR5 – Qdrant export & retrieval:** Changes will involve `scripts/export_to_qdrant.py`, `scripts/ask_qdrant.py`, and downstream retriever tests. Keep payload parity with Neo4j metadata to maintain rollback guarantees.

## Useful Commands
```bash
# Verify developer environment and capture diagnostics
scripts/bootstrap.sh --verify

# Start the local stack and wait for readiness
scripts/check_local_stack.sh --up
scripts/check_local_stack.sh --status --wait

# Run the minimal ingestion path end-to-end (requires OPENAI_API_KEY)
python scripts/create_vector_index.py
python scripts/kg_build.py --source docs/samples/pilot.txt
python scripts/export_to_qdrant.py --collection chunks_main
python scripts/ask_qdrant.py --question "What is the minimal path?"

# Tear down containers and remove data volumes
scripts/check_local_stack.sh --down --destroy-volumes

# Execute the end-to-end smoke test
pytest tests/integration/local_stack/test_minimal_path_smoke.py -k minimal_path_smoke
```

## 2025-10-06 Addendum — FancyRAG Service Hardening (Epic 5)
This addendum captures the architectural targets for Epic 5 (“FancyRAG Service Hardening”) and should be treated as the source of truth while Stories 5.1–5.6 are in flight. Update individual subsections as the implementation lands.

### Orchestrator Decomposition & Typed Settings
- `src/fancyrag/kg/pipeline.py` now leans on helper functions and typed settings for validation, continuing the decomposition started in Story 5.1.
- `src/config/settings.py` exposes the implemented `FancyRAGSettings` aggregate with nested `OpenAISettings`, `Neo4jSettings`, and `QdrantSettings`. The aggregate loads from `.env`/environment via Pydantic, caches per process, and is the canonical surface for scripts and pipeline code.
- All runtime entry points (`scripts/ask_qdrant.py`, `scripts/create_vector_index.py`, `scripts/export_to_qdrant.py`, and `src/fancyrag/kg/pipeline.py`) consume the aggregate instead of direct `os.environ` lookups. `.env.example` remains the onboarding artifact for required keys.
- Backwards compatibility: `fancyrag.utils.env.ensure_env` now delegates to `FancyRAGSettings` first, then falls back to raw environment reads for legacy tests or partial fixtures.

#### `.env` Migration Checklist (Story 5.2)
1. Back up your current `.env` (`cp .env .env.backup-$(date +%Y%m%d)`).
2. Copy `.env.example` updates if new comments are desired; the variable names remain identical. Ensure the backed-up values replace the `YOUR_…` placeholders.
3. Run `PYTHONPATH=src python -m cli.diagnostics openai-probe` or any FancyRAG script; typed settings will fail fast with descriptive errors if a value is missing or malformed.
4. Update CI secrets or `.env.local` variants using the mapping table below to align credential management discussions with the typed surface.
5. Rollback plan: restore `.env.backup-*` if a migration issue occurs; no feature flag is required because the typed loader honours the legacy variables directly.

| Legacy Variable | Typed Settings Location | Notes |
| --------------- | ----------------------- | ----- |
| `OPENAI_API_KEY` | `FancyRAGSettings.openai.api_key` | Same key; now available as a `SecretStr` for clients.
| `OPENAI_MODEL` / `OPENAI_EMBEDDING_MODEL` / overrides | `FancyRAGSettings.openai` fields | Validation errors now highlight unsupported models or invalid overrides.
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | `FancyRAGSettings.neo4j.{uri, auth()}` | URI scheme must be `bolt://`/`neo4j://`; auth tuple sourced via `auth()`.
| `NEO4J_DATABASE` (optional) | `FancyRAGSettings.neo4j.database` | Still optional; returned as `None` when unset.
| `QDRANT_URL` / `QDRANT_API_KEY` | `FancyRAGSettings.qdrant.{url, client_kwargs()}` | HTTP/HTTPS URLs enforced; API key is optional and masked.
| `QDRANT_NEO4J_ID_PROPERTY_*` | `FancyRAGSettings.qdrant.{neo4j_id_property, external_id_property}` | Defaults remain `chunk_id`; override in `.env` for custom schemas.

*(Extend this table when additional settings land in future stories.)*

### Automation Surface & Rollback Workflow
- A single entry point (`make service-run` via `scripts/service.py` → `fancyrag.cli.service_workflow`) orchestrates stack bootstrap → ingestion → export → evaluation → teardown with structured logging. Stage outcomes and artifact locations are persisted to `artifacts/local_stack/service/<timestamp>/service_run.json` for traceability.
- Companion commands `make service-rollback` and `make service-reset` delegate to the same workflow to unwind ingestion artefacts, stop Docker services, and optionally purge bound volumes/Qdrant collections.
- Automation commands accept preset overrides via flags/environment (e.g., `--preset full`, `FANCYRAG_PRESET=qa`). Defaults should reproduce the “smoke” dataset for CI and onboarding.
- Smoke preset baseline: bootstrap → teardown should complete within 35 seconds (measured 2025-10-07 via `artifacts/local_stack/service/20251007T124244/service_run.json`); flag runs exceeding this threshold for investigation.
- Rollback automation is responsible for:
  - Stopping Docker services via `scripts/check_local_stack.sh --down`.
  - Optionally purging bind-mounted volumes when requested.
  - Reverting partially written Neo4j/Qdrant data using ingestion run UUIDs recorded by the pipeline.
- Integration smoke (`tests/integration/local_stack/test_minimal_path_smoke.py`) will call the automation entry point to guarantee parity.

### Configuration Presets & Contracts
- Presets are currently surfaced through typed settings/environment keys (`FANCYRAG_PRESET`, `DATASET_PATH`, `DATASET_DIR`, `FANCYRAG_PROFILE`, `FANCYRAG_TELEMETRY`, `FANCYRAG_ENABLE_SEMANTIC`, `FANCYRAG_ENABLE_EVALUATION`, `FANCYRAG_VECTOR_INDEX`, `FANCYRAG_QDRANT_COLLECTION`). Story 5.4 will migrate these presets into versioned configuration files (`config/presets/<name>.yaml`) while retaining the same override knobs.
- Presets define:
  - Chunking parameters (`chunk_size`, `overlap`, cache toggles).
  - Semantic enrichment/embedding toggles.
  - Evaluation coverage (query sets, thresholds).
  - Telemetry defaults (OTEL exporters, sampling ratios).
- Typed settings layer resolves presets and validates compatibility (e.g., ensuring embedding dim matches vector index). Validation errors must halt runs with actionable guidance.
- Document each preset in this addendum (table to be populated as presets land) and link to relevant stories.

| Preset | Purpose | Key Overrides | Status |
| ------ | ------- | ------------- | ------ |
| smoke  | CI / onboarding minimal dataset | Small chunk size, evaluation OFF, telemetry console exporter | Planned |
| full   | Full corpus ingest | Default chunk size, evaluation ON (baseline), telemetry OTLP optional | Planned |
| enrich | Semantic enrichment heavy | Enrichment ON, cache aggressive, evaluation ON (expanded queries) | Planned |

### RAG Evaluation Harness
- Evaluation will leverage RAGAS (or alternative defined in Story 5.5) with configurable query sets stored alongside presets (e.g., `config/evaluation/smoke.yaml`).
- Pipeline stage `run_evaluation` produces:
  - Per-run JSON scorecard under `artifacts/local_stack/evaluation/<timestamp>/metrics.json`.
  - Markdown summary appended to the ingestion QA report.
- CI thresholds (defined in `pyproject.toml` or dedicated config) fail runs when metrics fall below preset floors. Log messages should include links to the scorecard artifacts.
- QA gates (`docs/qa/gates/5.x-*.yml`) must import the new evaluation evidence.

### Observability & Telemetry
- Embed OpenTelemetry spans around major pipeline stages. Default sampler: parent-based AlwaysOn for debugging; allow configuration via settings.
- Support exporters:
  - Console (default for local runs).
  - OTLP/gRPC (configurable endpoint, headers via settings).
- Emit metrics (latency, throughput, cache hit rate, evaluation durations) through OpenTelemetry metrics API. Provide example dashboards/queries in this document once available.
- Troubleshooting guidance:
  - Include snippets for launching Jaeger locally (`docker run --rm -p 16686:16686 jaegertracing/all-in-one`).
  - Document environment variables required to enable OTLP exports (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`).

### Documentation & Compatibility Commitments
- Update the onboarding section below once automation commands are implemented to showcase the new workflow.
- Maintain compatibility with existing CLI scripts during transition. Deprecate only after presubmit automation and documentation reflect the new entry points.
- Create a change log entry referencing Epic 5 once the first story merges (keep this addendum pinned to the current date, append new dated sections for future waves).

---
*Document owner:* Architecture step of the brownfield-service workflow. Update whenever new stories materially change the local stack, ingestion pipeline, or QA gating logic.
