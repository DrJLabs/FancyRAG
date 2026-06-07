# 0. Purpose & How To Use This Runbook
This runbook is the single, end-to-end execution plan for migrating FancyRAG to align with neo4j-graphrag-python, executed strictly PR-by-PR in the DR_06 sequence. Implement one PR at a time, run the required verification gates (DR_08), and do not proceed to the next PR until the current PR meets the acceptance criteria and “stop-the-line” constraints in Section 1.

# 1. Inputs & Canonical Baseline
## 1.1 Required inputs (this runbook is derived exclusively from these files)
- CANONICAL_BASELINE_00-04.md
- DR_05_symbol_mapping_matrix.md
- DR_06 Phased Refactor Plan.md
- DR_07_risk_register.md
- DR_08_verification_checklist.md
- DR_09_agent_task_list.md
- DR_10_machine_mapping_payload.json (**NOTE:** provided in the attachment named `10 — Machine Mapping Payload.md`, which contains JSON content)

## 1.2 Canonical SHAs (source of truth)
### FancyRAG baseline (canonical)
- Repo: DrJLabs/FancyRAG
- Default branch: `main`
- Canonical HEAD SHA: `0ff7b46717da4310b1626b0756551b7bd175d81b`
- Commit date: January 10, 2026
- Repo-root permalink (@ SHA): https://github.com/DrJLabs/FancyRAG/tree/0ff7b46717da4310b1626b0756551b7bd175d81b/

### Upstream baseline (canonical)
- Repo: neo4j/neo4j-graphrag-python
- Default branch: `main`
- Canonical HEAD SHA: `9e6b58827caa642539c8cb15dcde36c3e4d6d2c6`
- Commit date: December 17, 2025
- Repo-root permalink (@ SHA): https://github.com/neo4j/neo4j-graphrag-python/tree/9e6b58827caa642539c8cb15dcde36c3e4d6d2c6/

### Baseline caveat (must be tracked as a migration risk)
- DR_05 mapping matrix declares upstream baseline commit `62bbea3fe3dca4c6441ec037bfdcb9df6591a798` (circa v1.10) while the canonical upstream baseline is `9e6b58827caa642539c8cb15dcde36c3e4d6d2c6`. This mismatch is treated as a managed risk and must be called out in PRs that rely on DR_05 symbol decisions.

## 1.3 Non-negotiable contracts (must not drift)
### Retrieval/MCP contracts (external API is frozen)
- Endpoints remain stable:
  - `POST /mcp/search`
  - `POST /mcp/fetch`
  - `GET /mcp/health`
- `/mcp/search` response schema remains stable:
  - Top-level: `{query, top_k, effective_search_ratio, results}`
  - Per result: `{text, metadata, score, score_vector, score_fulltext, semantic_nodes, semantic_relationships}`
  - `metadata` must include `element_id` and `labels` and must not include `embedding`
- `/mcp/fetch` response schema remains stable:
  - `{found, element_id, metadata?, text?, error?}`
  - `metadata` must not include `embedding`

### Ingestion and QA contracts
- QA remains run-scoped and safe:
  - QA gates ingestion outcomes.
  - Rollback must only affect the current run and must not delete prior runs’ data.
  - QA artifacts remain scrubbed (no secrets, no embeddings, no credentials).
- Schema invariants remain true:
  - Every ingested `Chunk` remains reachable from a `Document` via `(:Document)-[:HAS_CHUNK]->(:Chunk)`.
- Indexing remains safe and predictable:
  - Prefer idempotent Python index scripts over destructive Makefile index recreation.
  - Detect configuration mismatches and fail fast rather than silently recreating indexes.

## 1.4 “Stop-the-line” criteria (from DR_07; apply to every PR)
- Contract breach (endpoint/schema drift).
- Data integrity issue (graph corruption, missing expected relationships such as Document→Chunk links).
- Security regression (sensitive values appear in logs/artifacts).
- Test suite failure (smoke tests and critical integration tests fail).
- Unstable operations (crashes, severe performance degradation, inability to ingest/search previously working data).

# 2. Execution Model (PR Workflow)
## 2.1 Branch naming convention
- Use one branch per DR_06 PR:
  - `migration/pr-01-baseline-alignment`
  - `migration/pr-02-qa-decouple`
  - …
  - `migration/pr-08-docs-ops`
- Within the PR description, explicitly list the DR_09 Task IDs implemented (traceability), and keep commits grouped by task.

## 2.2 Definition of Done (DoD) for a PR
A PR is “Done” only when all of the following are true:
- All PR acceptance criteria in Section 5 are met.
- DR_08 “minimum gate” passes (Section 4.1).
- No “stop-the-line” criteria are triggered (Section 1.4).
- PR includes:
  - A short “Intent” and “Scope” section.
  - A “Verification Evidence” section showing commands run and outcomes.
  - Explicit rollback instructions (how to revert and restore service).

## 2.3 Required checks before merge (tests + verification checklist)
### Always required (every PR)
- DR_08 A.1 (Record state) and A.2 (Tooling checks)
- DR_08 B.6.1 (`make smoke`)
- DR_08 B.10.1 (`make scan-image`) **when feasible** in the current environment/CI
- DR_08 B.10.2 (artifact/log scrubbing scan) **if** the PR touches artifacts/logging/QA

### Conditionally required (based on files changed)
- If PR touches MCP/retrieval:
  - DR_08 B.8.1–B.8.3 contract checks (`/mcp/health`, `/mcp/search`, `/mcp/fetch`)
- If PR touches index scripts/config:
  - DR_08 B.3.1–B.4 (ensure indexes and verify with `SHOW INDEXES`)
- If PR touches ingestion or QA/rollback:
  - DR_08 B.5.2 (canonical ingestion path; flags may be UNVERIFIED — see Section 8)
  - DR_08 B.6.2 (QA artifact presence + scrubbing scan)
  - DR_08 B.6.3 (rollback negative test) for high-risk ingestion changes

## 2.4 Evidence discipline (keeping permalinks current)
- Use commit-pinned GitHub permalinks for any “evidence” links in PR descriptions and long-lived docs.
- When citing code, prefer permalinks at the canonical SHAs listed in Section 1.2 (or at the PR head commit).
- Do not invent line ranges. If line ranges are needed, generate them by creating the link directly from the target repository UI at the relevant commit.

# 3. Global Test Plan
## 3.1 Unit tests (fast; must pass pre-merge)
**Why:** Protects critical pure-Python behaviors that are easy to regress (scrubbing, config parsing, helper logic, schema checks).
**Must pass:** pre-merge for every PR that touches the relevant modules.
**Add/maintain tests for:**
- Secret scrubbing (`scrub_object`) behavior (fake keys/patterns).
- Configuration parsing/unification (single source of truth; backward-compatible env var names).
- QA evaluator DB-reconstruction logic (run-scoped queries, empty-key behavior).
- Splitter caching determinism (cached vs non-cached chunk outputs).

## 3.2 Integration tests (Neo4j; must pass pre-merge for ingestion/retrieval PRs)
**Why:** Ingestion, indexing, retrieval, and QA all depend on the database behaving as expected under the real stack.
**Must pass:** pre-merge for PRs 2–7; nightly for PRs that only touch docs/formatting.
**Add/maintain tests for:**
- Ingestion populates `Document`, `Chunk`, `HAS_CHUNK` relationships, and embeddings.
- Index scripts (vector + fulltext) detect and/or fail fast on mismatches.
- QA rollback scoping (negative test: failed run deletes only run-tagged data).

## 3.3 Contract tests (MCP/search/fetch/health; must pass pre-merge for MCP PRs)
**Why:** The MCP API is a frozen client contract; schema drift is a stop-the-line issue.
**Must pass:** pre-merge for PRs 1, 6, 7 (and any PR that touches MCP).
**Add/maintain tests for:**
- `/mcp/health` returns `{"status":"ok"}` when DB is reachable.
- `/mcp/search` response contains required fields; `metadata` excludes `embedding`.
- `/mcp/fetch` round-trip works using `element_id` returned by search.
- Invalid parameter handling (>=400; and after PR7, explicit 400 with stable body).

## 3.4 Smoke tests (end-to-end; must pass pre-merge for every PR)
**Why:** `make smoke` is the authoritative regression gate across ingestion/retrieval/indexing interactions.
**Must pass:** pre-merge for every PR.
**Run cadence:** pre-merge always; nightly job can run additional longer scenarios (full ingest on fixture corpus, extended contract suite).

# 4. Global Verification Playbook (Reference)
## 4.1 Minimum gate (run on every PR; DR_08)
Run the following DR_08 sections (copy/paste from DR_08):
1) A.1 Record state
2) A.2 Tooling checks
3) B.2 Bring up services (`make up`)
4) B.3 Create indexes (prefer scripts)
5) B.8 Contract checks (health/search/fetch) — when MCP is in scope
6) B.6.1 `make smoke`
7) B.10 Secret/log safety checks (as applicable)
8) B.11 Teardown (`make down`)

## 4.2 Copy/paste: “minimum gate” quick sequence (wrapper)
> This wrapper intentionally references DR_08 sections; use the exact blocks from DR_08 for details.

```bash
# DR_08 A.1 + A.2
git status --porcelain
git rev-parse HEAD
docker --version
docker compose version
uv --version
curl --version
python --version

# DR_08 B.2 + B.3 (services + indexes)
make up
uv run python scripts/create_vector_index.py
uv run python scripts/create_fulltext_index.py

# DR_08 B.6.1 (authoritative gate)
make smoke

# DR_08 B.11
make down
```

## 4.3 Full gate (run before merges for risky PRs: PR5/PR6/PR7)
Use DR_08 “Full gate” guidance:
- Minimum gate, plus:
  - Real ingestion on a representative fixture (`make ingest`) and Neo4j verification queries (DR_08 B.5.3)
  - Index verification via `SHOW INDEXES` (DR_08 B.4)
  - QA gating verification including rollback behavior (DR_08 B.6.3, currently marked UNVERIFIED; see Section 8)

# 5. PR-by-PR Execution Queue (Core Deliverable)
## PR 1 — Baseline Alignment & Stabilization
- **PR ID + Title:** PR-01 — Baseline Alignment & Stabilization
- **Intent:** Establish consistent baseline for refactor (resolve conflicts, unify settings, and eliminate namespace drift).
- **Files touched (explicit paths):**
  - `pyproject.toml`
  - `uv.lock`
  - `src/fancryrag/** (move/rename) → src/fancyrag/**`
  - `src/fancryrag/__init__.py (temporary compat alias)`
  - `servers/mcp_hybrid_google.py`
  - `scripts/** (as needed for import fixes)`
  - `.env.example`
  - `Makefile`
  - `scripts/create_vector_index.py`
  - `docker-compose.yml`
  - `docker-compose.smoke.yml`
- **Atomic tasks included (DR_09 Task IDs):** DR09-T01, DR09-T02, DR09-T03
- **Acceptance criteria (testable):**
  - GraphRAG dependency is pinned to canonical upstream SHA `9e6b58827caa642539c8cb15dcde36c3e4d6d2c6`; `uv run python -c "import neo4j_graphrag"` succeeds.
  - `fancyrag` import works; `fancryrag` import continues to work via shim during transition.
  - Embedding dimension defaults are consistent across `.env.example`, `Makefile`, and `scripts/create_vector_index.py` and smoke stack remains deterministic.
  - `make smoke` exits 0.
- **Tests required/changed (from DR_05 + Section 3):**
  - Smoke: `make smoke` (required).
  - Unit: import sanity tests (optional), config parsing tests (recommended).
  - Integration: index ensure scripts succeed (`scripts/create_vector_index.py`, `scripts/create_fulltext_index.py`).
- **Verification steps (DR_08):**
  - A.1, A.2, B.1.1, B.2, B.3.1–B.3.2, B.6.1, B.11
- **Risk checks (DR_07) + mitigations:**
  - Upstream Baseline Mismatch → mitigate by pinning dependency (DR09-T01).
  - Module Name Inconsistency → mitigate via rename + compat shim + smoke (DR09-T02).
  - Embedding Dimension Mismatch → mitigate via unified defaults and smoke override (DR09-T03).
- **Rollback plan (from DR_06):** Revert PR; changes are isolated (dependency/naming/config defaults) and have no data migrations.
- **Merge Gate (all must be checked):**
  - [ ] DR_08 minimum gate completed with evidence in PR description
  - [ ] `make smoke` passes
  - [ ] `fancyrag` import works; `fancryrag` compat import works (temporary)
  - [ ] Index scripts run successfully (or fail fast only on intentional mismatch)
  - [ ] No stop-the-line criteria triggered

## PR 2 — Decouple QA from Pipeline
- **PR ID + Title:** PR-02 — Decouple QA from Pipeline
- **Intent:** Ensure QA evaluation and rollback work independently of ingestion implementation by reconstructing QA scope from Neo4j instead of pipeline in-memory state.
- **Files touched (explicit paths):**
  - `src/fancyrag/db/neo4j_queries.py`
  - `src/fancyrag/qa/evaluator.py`
  - `src/fancyrag/qa/report.py`
  - `src/fancyrag/kg/** (run-key tagging normalization)`
  - `tests/**`
- **Atomic tasks included (DR_09 Task IDs):** DR09-T04, DR09-T05, DR09-T06
- **Acceptance criteria (testable):**
  - QA can run using only `ingest_run_key` (or equivalent run identifier) and produces reports with the same top-level schema as baseline.
  - QA artifacts are present and scrubbed (no obvious secret patterns).
  - Rollback remains run-scoped (no collateral deletes).
  - `make smoke` exits 0.
- **Tests required/changed (from DR_05 + Section 3):**
  - Unit: DB reconstruction path tests; scrub tests for QA artifacts.
  - Integration: run a small ingest + QA; validate Neo4j counts and ensure rollback scoping in a negative test (DR_08 B.6.3; UNVERIFIED harness).
  - Smoke: `make smoke` (required).
- **Verification steps (DR_08):**
  - Minimum gate, plus: B.6.2 (QA artifact presence + scrubbing scan), B.5.3 (Neo4j verification) when possible, B.6.3 rollback negative test when rollback logic changes (UNVERIFIED harness).
- **Risk checks (DR_07) + mitigations:**
  - QA Coupled to Pipeline → mitigated by DB-backed reconstruction (DR09-T05).
  - Implicit Schema Invariants → mitigated by adding invariant checks to QA/test suite (begins here; expanded in PR7).
  - Security regression → mitigated via scrub tests and DR_08 B.6.2/B.10 scans.
- **Rollback plan (from DR_06):** Revert PR (QA refactor only); no persistent data changes required.
- **Merge Gate (all must be checked):**
  - [ ] DR_08 minimum gate completed with evidence in PR description
  - [ ] DR_08 B.6.2 confirms QA artifacts exist and are scrubbed
  - [ ] If rollback code touched: negative rollback scenario documented (or UNVERIFIED harness steps executed)
  - [ ] `make smoke` passes
  - [ ] No stop-the-line criteria triggered

## PR 3 — Consolidate Ingestion Path
- **PR ID + Title:** PR-03 — Consolidate Ingestion Path
- **Intent:** Remove dual ingestion pathways and enforce a single, unified entry point for KG building (script-based pipeline as the canonical path).
- **Files touched (explicit paths):**
  - `Makefile`
  - `scripts/kg_build.py`
  - `README.md`
  - `tools/run_pipeline.py (deprecate then remove)`
  - `pipelines/kg_ingest.yaml (deprecate then remove)`
  - `docs/** (as needed)`
- **Atomic tasks included (DR_09 Task IDs):** DR09-T07, DR09-T08, DR09-T09, DR09-T10
- **Acceptance criteria (testable):**
  - `make ingest` uses `scripts/kg_build.py` (no YAML runner invocation).
  - Legacy YAML ingestion artifacts are removed (or at least deprecated and no longer referenced).
  - Any YAML-only behaviors (e.g., resolvers) are either implemented in code or explicitly documented as dropped.
  - `make smoke` exits 0.
- **Tests required/changed (from DR_05 + Section 3):**
  - Smoke: `make smoke` (required).
  - Integration: run `make ingest` against a small fixture; then DR_08 B.5.3 Neo4j verification queries.
  - Unit: CLI help/flag parsing for `scripts/kg_build.py` if Makefile requires stable flags.
- **Verification steps (DR_08):**
  - B.5.2 discovery (`uv run python scripts/kg_build.py --help`) to lock CLI flags; B.5.1/B.5.2 ingestion run; B.6.1 smoke gate.
- **Risk checks (DR_07) + mitigations:**
  - Dual Ingestion Path Drift → mitigated by Makefile routing to script and removing YAML.
  - Dropped Pipeline Feature → mitigated by explicit parity decision (DR09-T10) and QA/invariant checks.
  - Pipeline Config Mismatch → mitigated by moving off YAML config file as the primary mechanism.
- **Rollback plan (from DR_06):** Restore removed files and revert Makefile changes (version control); re-enable prior ingest target if required.
- **Merge Gate (all must be checked):**
  - [ ] DR_08 minimum gate completed with evidence in PR description
  - [ ] `make ingest` runs the script-based pipeline
  - [ ] `git grep -n "run_pipeline.py\|kg_ingest.yaml"` returns no live references (or only in deprecation notes)
  - [ ] `make smoke` passes
  - [ ] No stop-the-line criteria triggered

## PR 4 — Integrate Upstream LLM & Embeddings
- **PR ID + Title:** PR-04 — Integrate Upstream LLM & Embeddings
- **Intent:** Replace custom OpenAI client/embeddings implementations with upstream `OpenAILLM` + `OpenAIEmbeddings`, remove redundant telemetry and retry layers, and preserve scrubbing.
- **Files touched (explicit paths):**
  - `src/cli/openai_client.py`
  - `src/cli/telemetry.py (remove)`
  - `src/cli/sanitizer.py`
  - `src/fancyrag/embeddings.py (NOTE: may have been src/fancryrag/embeddings.py pre-PR1 rename)`
  - `src/fancyrag/mcp/runtime.py (embedder construction if used in retrieval runtime)`
  - `src/fancyrag/config.py and/or src/config/settings.py`
  - `tests/**`
- **Atomic tasks included (DR_09 Task IDs):** DR09-T11, DR09-T12, DR09-T13, DR09-T14
- **Acceptance criteria (testable):**
  - Ingestion and retrieval still function with OpenAI keys (or smoke stub mode), and embeddings are stored with the expected dimension.
  - Custom telemetry is removed without breaking runtime or tests.
  - Secret scrubbing remains effective (QA artifacts and any structured logs).
  - `make smoke` exits 0.
- **Tests required/changed (from DR_05 + Section 3):**
  - Unit: scrubbing tests; config mapping tests; ensure removal of telemetry does not break imports.
  - Integration: ingest → verify `Chunk.embedding` exists and has expected length; run retrieval query (via MCP contract checks).
  - Smoke: `make smoke` (required).
- **Verification steps (DR_08):**
  - B.8.1–B.8.3 (health/search/fetch) if MCP embedder construction changed; B.5.3 Neo4j checks for missing embeddings; B.6.2 scrubbing scan; B.6.1 smoke gate.
- **Risk checks (DR_07) + mitigations:**
  - Double Rate-Limiting Delay → mitigate by ensuring only one rate/limit/retry layer is active.
  - LLM Integration Variance → mitigate with functional tests and adapter layer as needed.
  - Security regression → mitigate via scrub tests + DR_08 scrubbing scans.
- **Rollback plan (from DR_06):** Keep a backup branch containing the old OpenAI client/embeddings logic; revert PR to restore old behavior if upstream integration breaks.
- **Merge Gate (all must be checked):**
  - [ ] DR_08 minimum gate completed with evidence in PR description
  - [ ] `make smoke` passes
  - [ ] If MCP touched: contract checks (DR_08 B.8.1–B.8.3) pass
  - [ ] No secret patterns in artifacts/logs (DR_08 B.6.2/B.10.2)
  - [ ] No stop-the-line criteria triggered

## PR 5 — Adopt Upstream Pipeline Orchestration
- **PR ID + Title:** PR-05 — Adopt Upstream Pipeline Orchestration
- **Intent:** Replace FancyRAG’s custom KG pipeline orchestration with upstream experimental pipeline APIs behind a stable FancyRAG interface, ensuring schema invariants and QA gating remain intact.
- **Files touched (explicit paths):**
  - `scripts/kg_build.py`
  - `src/fancyrag/cli/kg_build_main.py`
  - `src/fancyrag/kg/upstream_backend.py (new)`
  - `src/fancyrag/kg/pipeline.py (deprecated then removed)`
  - `src/fancyrag/kg/phases.py (deprecated then removed)`
  - `src/fancyrag/db/neo4j_queries.py (schema invariant helpers, rollback support)`
  - `src/fancyrag/qa/evaluator.py (optional invariant checks)`
  - `tests/**`
- **Atomic tasks included (DR_09 Task IDs):** DR09-T15, DR09-T16, DR09-T17, DR09-T18
- **Acceptance criteria (testable):**
  - Upstream backend is implemented and can run on a fixture corpus.
  - Default ingestion backend switches to upstream (while preserving a fallback to the prior backend during transition).
  - Schema invariants hold after ingestion (Document/Chunk/HAS_CHUNK links; missing embeddings = 0 for real ingestion).
  - QA runs post-ingestion and remains run-scoped and safe.
  - `make smoke` exits 0.
- **Tests required/changed (from DR_05 + Section 3):**
  - Integration: ingest using upstream backend + Neo4j verification queries + QA report generation.
  - Contract: retrieval contract checks to ensure ingestion changes did not break retrieval expectations.
  - Smoke: `make smoke` (required).
  - Nightly: extended ingest on representative corpus and golden queries (recommended).
- **Verification steps (DR_08):**
  - Full gate required: B.5.2/B.5.3 ingestion + Neo4j verification; B.6.2 QA artifact + scrub scan; B.8.1–B.8.3 contract checks; B.6.3 rollback negative test (UNVERIFIED harness; document discovery outputs).
- **Risk checks (DR_07) + mitigations:**
  - Missing Graph Links → mitigate by enforcing links via post-step (retain or reintroduce ensure_document_relationships if upstream writer does not link).
  - Implicit Schema Invariants → mitigate by hard invariant checks and tests.
  - Dropped Pipeline Feature → mitigate via explicit parity decisions and documentation.
  - Upstream Version Drift → mitigate by pinned SHA + isolating upstream experimental usage behind one module.
- **Rollback plan (from DR_06):** Keep the old pipeline accessible behind a fallback flag during transition; if severe issues arise, revert PR to last stable.
- **Merge Gate (all must be checked):**
  - [ ] Full gate executed (DR_08 minimum + ingestion + QA + contract checks)
  - [ ] Schema invariants verified (Document/Chunk/HAS_CHUNK; missing embeddings)
  - [ ] QA and rollback remain run-scoped (no collateral deletes)
  - [ ] `make smoke` passes
  - [ ] No stop-the-line criteria triggered

## PR 6 — Align Retrieval with Upstream API
- **PR ID + Title:** PR-06 — Align Retrieval with Upstream API
- **Intent:** Integrate upstream retrieval semantics while preserving FancyRAG’s MCP external API and response schema (including score breakdown and semantic context fields).
- **Files touched (explicit paths):**
  - `src/fancyrag/mcp/runtime.py`
  - `queries/hybrid_retrieval.cypher (remove or treat as locked contract artifact; decision required)`
  - `tests/smoke/test_stack.py and/or tests/contract/test_mcp_contract.py`
- **Atomic tasks included (DR_09 Task IDs):** DR09-T19, DR09-T20, DR09-T21
- **Acceptance criteria (testable):**
  - `/mcp/search` returns identical JSON structure (field presence and types) as baseline, including `score`, `score_vector`, `score_fulltext`, `semantic_nodes`, `semantic_relationships`.
  - `/mcp/fetch` works for an `element_id` returned by search and never returns embeddings in metadata.
  - `make smoke` exits 0.
- **Tests required/changed (from DR_05 + Section 3):**
  - Contract tests: automated MCP contract tests (schema-only, not exact-score) in smoke stack.
  - Integration: run contract checks against a known dataset and diff field-by-field (allowing minor numeric differences).
  - Smoke: `make smoke` (required).
- **Verification steps (DR_08):**
  - B.8.1–B.8.3 contract checks (mandatory); B.6.1 smoke gate; verify semantic payload behavior remains stable if query-template strategy changes.
- **Risk checks (DR_07) + mitigations:**
  - Result Format Regression → mitigate by contract tests that assert presence of all fields and by translation layers that compute missing fields.
  - Undefined Search Error Handling → mitigate later in PR7 by explicit 400 handling; for PR6, ensure at least >=400 for invalid inputs.
  - OAuth Misconfiguration → if auth behavior touched, validate in staging per DR_07.
- **Rollback plan (from DR_06):** Reinstate the previous search method (restore original Cypher query and scoring logic) if upstream approach cannot meet contract needs.
- **Merge Gate (all must be checked):**
  - [ ] Contract tests added and passing (DR_08 B.8.1–B.8.3 and automated tests)
  - [ ] `make smoke` passes
  - [ ] No schema drift in `/mcp/search` or `/mcp/fetch`
  - [ ] No stop-the-line criteria triggered

## PR 7 — Hardening & Policy Compliance
- **PR ID + Title:** PR-07 — Hardening & Policy Compliance
- **Intent:** Close remaining contract ambiguities and operational risks: explicit HTTP error semantics, fulltext index drift detection, config consolidation, and automated schema invariant checks.
- **Files touched (explicit paths):**
  - `src/fancyrag/mcp/runtime.py`
  - `scripts/create_fulltext_index.py`
  - `src/fancyrag/config.py`
  - `src/config/settings.py`
  - `src/config/utils.py`
  - `tests/**`
- **Atomic tasks included (DR_09 Task IDs):** DR09-T22, DR09-T23, DR09-T24, DR09-T25
- **Acceptance criteria (testable):**
  - Invalid MCP inputs return explicit HTTP 400 with a stable, non-secret-leaking error body.
  - Fulltext index script detects mismatches and fails fast (and optionally supports an explicit recreate flag).
  - Config definitions are consolidated to a single source of truth while preserving env var compatibility.
  - Schema invariants are enforced by automated checks (QA and/or tests).
  - `make smoke` exits 0.
- **Tests required/changed (from DR_05 + Section 3):**
  - Contract: invalid input tests for `/mcp/search` and `/mcp/fetch`.
  - Integration: fulltext index mismatch detection scenario; schema invariant checks against Neo4j.
  - Unit: config parsing/compatibility and invariant helper functions.
  - Smoke: `make smoke` (required).
- **Verification steps (DR_08):**
  - B.8.4 invalid parameter behavior check (mandatory); B.3.2 + B.4 fulltext index ensure and verification; B.6.1 smoke gate.
- **Risk checks (DR_07) + mitigations:**
  - Undefined Search Error Handling → mitigate by explicit 400s.
  - Index Drift (Fulltext) → mitigate by mismatch detection.
  - Duplicate Config Definitions → mitigate by consolidation.
  - Implicit Schema Invariants → mitigate by enforcement tests.
- **Rollback plan (from DR_06):** Localized changes; revert PR or selectively revert specific hardening commits if needed.
- **Merge Gate (all must be checked):**
  - [ ] Contract error semantics validated (`top_k=0` yields 400; no secrets leaked)
  - [ ] Fulltext index mismatch detection verified
  - [ ] Config consolidation verified (no duplicate `Neo4jSettings` definitions)
  - [ ] `make smoke` passes
  - [ ] No stop-the-line criteria triggered

## PR 8 — Final Cleanup & Documentation
- **PR ID + Title:** PR-08 — Final Cleanup & Documentation
- **Intent:** Remove remaining dead code and finalize documentation, upgrade notes, and CI verification gates.
- **Files touched (explicit paths):**
  - `Deprecated code paths/files identified during PRs 1–7 (repo-wide sweep)`
  - `README.md`
  - `CHANGELOG.md or docs/upgrade.md`
  - `.github/workflows/verify.yml (new)`
  - `docs/**`
- **Atomic tasks included (DR_09 Task IDs):** DR09-T26, DR09-T27, DR09-T28
- **Acceptance criteria (testable):**
  - No references remain to removed legacy components (e.g., YAML runner, removed telemetry classes) outside changelog/history.
  - Docs reflect the current operational reality (ingest, index, smoke, MCP contract).
  - CI gate exists that runs at least the DR_08 minimum verification set per PR.
  - `make smoke` exits 0.
- **Tests required/changed (from DR_05 + Section 3):**
  - Smoke: `make smoke` (required).
  - CI: verify workflow executes smoke and contract checks in smoke mode.
- **Verification steps (DR_08):**
  - B.6.1 smoke gate; documentation walkthrough in a clean environment is recommended (manual), consistent with DR_08 guidance.
- **Risk checks (DR_07) + mitigations:**
  - Legacy Feature Drift / Doc drift → mitigate via CI gate and doc hygiene checks.
  - Upstream Version Drift → mitigate by updating upgrade protocol documentation and mapping artifacts.
- **Rollback plan (from DR_06):** N/A for docs-only changes; for cleanup, revert if an unexpected reference removal breaks runtime.
- **Merge Gate (all must be checked):**
  - [ ] Repo-wide grep sweep for removed symbols shows no live references
  - [ ] Docs updated; README follows Makefile targets and DR_08 commands
  - [ ] CI workflow added and runs smoke/contract checks
  - [ ] `make smoke` passes
  - [ ] No stop-the-line criteria triggered

# 6. Post-Merge Validation & Release Protocol
- After merging **each** PR:
  - Re-run DR_08 minimum gate on the merge commit (or rely on CI evidence if identical).
  - If the PR touched ingestion schema, run DR_08 B.5.3 Neo4j verification queries on a representative dataset.
  - If the PR touched MCP, run DR_08 B.8.1–B.8.3 contract checks and confirm clients see no schema drift.
- Before any release cut (or before PR5/PR6 merge if treated as release-quality):
  - Run DR_08 full gate.
  - Confirm no contract drift by running the automated contract tests (PR6) and validating the same required fields exist in responses.
  - Ensure index scripts are idempotent and mismatch-safe in the target deployment environment.

# 7. Maintenance: Upstream Upgrade Protocol
When bumping `neo4j-graphrag-python`, treat it as its own PR and follow this protocol (adapted from DR_06 “Upgrade Protocol”):
- Pin and monitor upstream versions (commit SHA or exact release version).
- Upgrade incrementally (avoid multi-version leaps).
- Run regression suite before merge:
  - `make smoke`
  - MCP contract tests (`/mcp/health`, `/mcp/search`, `/mcp/fetch`)
  - Ingestion + QA (fixture corpus) and schema invariants
  - Index scripts ensure + `SHOW INDEXES` verification
- Review upstream changes for breaking API/behavior changes affecting:
  - `HybridCypherRetriever`, `OpenAIEmbeddings`, `OpenAILLM`, pipeline APIs, and index helpers.
- Required artifact updates when upgrading upstream:
  - DR_05_symbol_mapping_matrix.md (update targets and decisions if upstream moved)
  - DR_10_machine_mapping_payload.json (regenerate)
  - DR_06 phased plan and this runbook (if PR ordering or acceptance criteria change)
  - Any contract tests / golden queries (if upstream behavior changes require adapters)

# 8. Known Gaps / UNVERIFIED Items
1) **DR_10 filename mismatch** — The attachment is named `10 — Machine Mapping Payload.md` but contains JSON and references `artifact.file_name = "DR_10_machine_mapping_payload.json"`.
   - Fix needed in: packaging/attachments (rename the file consistently) or add a note in DR_10 metadata.
2) **Canonical ingestion CLI flags are not standardized** — DR_08 B.5.2 explicitly marks `scripts/kg_build.py` flags as UNVERIFIED.
   - Fix needed in: DR_08 Verification Checklist, Section B.5.2 (replace discovery-only guidance with the exact, canonical flags).
3) **Rollback negative test harness is UNVERIFIED** — DR_08 B.6.3 describes discovery steps but not a deterministic, automated rollback test.
   - Fix needed in: DR_08 Section B.6.3 and DR_09 tasks (add a deterministic harness and codify into CI).
4) **Decision on `queries/hybrid_retrieval.cypher` lifecycle** — DR_09-T21 requires a decision to remove vs lock this file; DR_06 PR6 implies removal is optional.
   - Fix needed in: DR_06 PR6 description (explicitly choose A or B) and DR_08 contract validation notes (to reflect the chosen strategy).
5) **DR_05 upstream baseline differs from canonical upstream baseline** — DR_05 uses `62bbea3fe3dca4c6441ec037bfdcb9df6591a798` while canonical upstream baseline is `9e6b58827caa642539c8cb15dcde36c3e4d6d2c6`.
   - Fix needed in: DR_05 metadata section (add a clear note that mapping was against an older upstream snapshot and must be revalidated against canonical upstream head, or regenerate DR_05).
6) **Citations in DR_06/DR_07 use `contentReference[oaicite:...]` placeholders** — These are not stable permalinks and are not actionable evidence for coding agents.
   - Fix needed in: DR_06 and DR_07 source docs (replace placeholders with commit-pinned GitHub permalinks where feasible).
