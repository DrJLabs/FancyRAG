# Epic 5: FancyRAG Service Hardening

## Status
In Progress — 2025-10-06. Story 5.1 complete; stories 5.2–5.6 queued for drafting by the Scrum Master.

## Epic Goal
Operationalise the refactored FancyRAG ingestion pipeline into a reliable service workflow that any operator can run end-to-end with built-in telemetry, configuration safeguards, and rollback procedures.

## Epic Description
**Existing System Context:**
- Epic 4 shipped the `src/fancyrag/` package, breaking the former `kg_build.py` monolith into orchestrator, splitter, QA, config, and Neo4j helper modules.
- Local Docker Compose brings up Neo4j 5.26.12 (APOC) and Qdrant 1.15.4, but service bootstrapping still relies on ad-hoc shell scripts and manual environment preparation.
- Telemetry, QA evidence, and ingestion settings remain scattered across `.env`, bespoke CLI flags, and loosely structured artefacts, limiting repeatability and observability.

**Enhancement Highlights:**
- Refactor the pipeline coordinator into composable helpers so each ingestion phase (settings resolution, source discovery, semantic enrichment, QA, evaluation) can be tested and instrumented independently.
- Introduce typed settings classes (`FancyRAGSettings` with nested `OpenAISettings`, `Neo4jSettings`, and `QdrantSettings`) that centralise configuration with validation and sensible defaults.
- Provide automation hooks for stack bootstrap, ingestion execution, rollback, and smoke validation so operators have a single command surface.
- Expand ingestion presets (chunking, cache, semantic enrichment toggles) with documented configuration contracts and stable defaults.
- Package QA, evaluation, and telemetry outputs into versioned artefacts that downstream reviewers can trust without spelunking logs.
- Instrument pipeline stages with OpenTelemetry spans/metrics to expose latency, throughput, and cache health for both local and future managed deployments.

**Outcome:**
- FancyRAG becomes a service-quality stack: bootstrap, ingest, evaluate, and observe using scripted workflows guarded by typed configuration.
- Operators gain deterministic telemetry bundles (QA reports, evaluation scorecards, tracing) aligned with CI gates.
- Future managed-environment rollouts inherit the same configuration surface, reducing rework when migrating away from the local stack.

## Stories
1. ☑ **Story 5.1:** Decompose the pipeline orchestrator into helper functions/classes with explicit inputs and outputs (`docs/stories/5.1.pipeline-orchestrator.md` — done).
2. ☑ **Story 5.2:** Centralise typed settings covering FancyRAG, OpenAI, Neo4j, and Qdrant configuration (`docs/stories/5.2.centralise-typed-settings.md` — done).
3. ☐ **Story 5.3:** Automate stack lifecycle (bootstrap, ingest, teardown, rollback) with reproducible scripts and smoke validation (`docs/stories/5.3.stack-automation.md` — pending).
4. ☐ **Story 5.4:** Harden configuration surface and defaults for new pipeline presets (chunking/cache/enrichment) and document the contracts (`docs/stories/5.4.pipeline-configuration.md` — pending).
5. ☐ **Story 5.5:** Integrate a retrieval QA harness (e.g., RAGAS) that records scorecards alongside ingestion artefacts and CI gates (`docs/stories/5.5.rag-evaluation.md` — pending).
6. ☐ **Story 5.6:** Instrument pipeline stages with OpenTelemetry spans/metrics and provide local/remote exporter guidance (`docs/stories/5.6.observability.md` — pending).

## Compatibility Requirements
- [ ] Existing CLI entry points (`scripts/kg_build.py`, Make targets, smoke scripts) continue to function without flag changes.
- [ ] `.env` contract and environment discovery remain backwards compatible while exposing new optional settings.
- [ ] Generated QA, evaluation, and telemetry artefacts are versioned and stored under `artifacts/local_stack/` without breaking consumers.
- [ ] OpenTelemetry integration can be disabled with zero impact on default logging or performance envelopes.

## Risk Mitigation
- **Primary Risk:** Service orchestration changes regress ingestion behaviour or QA guarantees.  
  **Mitigation:** Maintain smoke comparisons against pre-epic baselines, add unit coverage for each new helper, and gate merges on parity artefacts.
- **Secondary Risk:** Typed settings add configuration friction or misaligned defaults.  
  **Mitigation:** Document defaults, ship migration notes, and provide lint/validation commands that catch misconfiguration early.
- **Tertiary Risk:** Telemetry additions introduce runtime overhead.  
  **Mitigation:** Benchmark spans/metrics in CI, allow opt-out to isolate issues, and document performance budgets.
- **Rollback Plan:** Retain pre-epic scripts and configuration in git history; automation scripts include a rollback target restoring the prior pipeline orchestration.

## Definition of Done
- [ ] All six stories complete with acceptance criteria satisfied and QA sign-off filed.
- [ ] Single-command bootstrap/ingestion/evaluation workflow documented and exercised in CI.
- [ ] Typed settings classes adopted across FancyRAG modules with tests covering validation paths.
- [ ] QA dashboards, evaluation scorecards, and telemetry artefacts generated automatically and linked in the PRD.
- [ ] Observability guidance (Jaeger/Grafana/console) documented in `docs/brownfield-architecture.md` (and, if warranted, a new `docs/architecture/projects/fancyrag-service-hardening.md` shard) so operators have a canonical reference.

## Validation Checklist
**Scope Validation:**
- [ ] Epic contained within six interdependent but manageable stories.
- [ ] Focus remains on service hardening (no net-new product features).
- [ ] Integration points leverage existing FancyRAG modules and Docker stack.

**Risk Assessment:**
- [ ] Regression testing plan documented (unit, smoke, telemetry diff).
- [ ] Configuration migration guidance prepared for operators.
- [ ] Telemetry and QA outputs reviewed with QA/Test stakeholders.

**Completeness Check:**
- [ ] Success metrics: smoke parity, evaluation thresholds, telemetry coverage.
- [ ] Dependencies captured (OpenAI creds, Docker stack, Observability backend).
- [ ] Story sequencing ensures typed settings/automation land before telemetry hooks.

## Story Manager Handoff
"Please draft and sequence the six stories above with the following guardrails in mind:

- Respect the typed settings contract so CLI, automation scripts, and FancyRAG modules consume a single source of configuration truth.
- Preserve existing ingestion outputs while layering on QA scorecards and telemetry; operators must be able to diff artefacts before/after the epic.
- Ensure bootstrap/rollback automation covers Docker lifecycle, env validation, and smoke execution so failures surface early.
- Capture observability usage guides (console, Jaeger/Grafana) and embed thresholds into CI to prevent silent regressions.
- Coordinate with QA to baseline evaluation metrics and document how new telemetry feeds their dashboards."
