# Epic 3: Ingestion Quality Hardening — Adaptive Chunking & QA

## Status
Done — 2025-10-02. All stories (3.1–3.4) delivered; minimal-path smoke validates native retriever and documentation lint guard.

## Epic Goal
Raise the fidelity and observability of the GraphRAG ingestion pipeline so operators can ingest varied corpora (docs, source code, knowledge bases) with tunable chunking, richer metadata, and automated quality checkpoints before export or retrieval.

## Epic Description
**Existing System Context:**
- Stories 3.1–3.3 delivered adaptive chunking, ingestion QA gating, and native retrieval via `QdrantNeo4jRetriever` running on the local stack.
- Minimal-path smoke now exercises the retriever end-to-end using `.env` OpenAI credentials; telemetry and sanitized artifacts remain stable for CI consumers.

**Enhancement Highlights:**
- Chunking presets (text/markdown/code) with deterministic directory ingestion propagate metadata (relative path, git commit, checksum) into Neo4j/Qdrant payloads.
- QA reporting emits gating artifacts under `artifacts/ingestion/` and blocks anomalies before writes; telemetry logged for CI analysis.
- Retrieval now uses the native GraphRAG `QdrantNeo4jRetriever`, ensuring end-to-end alignment with official APIs and simplifying future semantic enrichment.
- Documentation lint guard runs in automation/CI to keep architecture runbooks aligned with retriever workflows and minimal-path commands.

**Outcome:**
- Minimal path executes start-to-finish with adaptive chunking, QA gating, native retrieval, and documentation lint automation. All acceptance criteria satisfied with lint guard keeping runbooks aligned to shipped scripts.

## Stories
1. ✅ **Story 3.1:** Adaptive chunking profiles & source-aware ingestion (`docs/stories/3.1.adaptive-chunking.md`).
2. ✅ **Story 3.2:** Ingestion QA telemetry & quality gate (`docs/stories/3.2.ingestion-qa-telemetry.md`).
3. ✅ **Story 3.3:** Native Qdrant retriever integration (`docs/stories/3.3.qdrant-native-retriever.md`).
4. ✅ **Story 3.4:** Documentation lint automation (`docs/stories/3.4.documentation-lint-automation.md`).

## Compatibility Requirements
- [x] CLI defaults must continue to succeed on the existing sample (`docs/samples/pilot.txt`) without additional flags.
- [x] `.env` contract unchanged; new settings exposed via CLI/config files.
- [x] Unit + integration tests remain runnable without managed services.

## Risk Mitigation
- **Primary Risk:** Complex chunkers increase runtime/cost on large corpora.
  - **Mitigation:** Provide dry-run mode with token estimates, documented defaults per corpus type, and configurable rate-limit guards.
- **Secondary Risk:** Metadata expansion bloats Neo4j/Qdrant payloads.
  - **Mitigation:** Store hashes/checksums separately from text payload; document pruning strategies.
- **Rollback Plan:** Feature-flag new splitters and semantic extraction; reverting to the fixed-size profile restores prior behaviour.

## Definition of Done
- [x] All stories delivered with acceptance criteria satisfied.
- [x] Integration smoke covers at least one mixed-content corpus exercising new profiles.
- [x] QA reports archived for two consecutive successful runs.
- [x] Documentation updated (architecture overview, coding standards, QA assessments) with new workflow details.

## Validation Checklist
**Scope Validation:**
- [x] Epic delivered via four-story plan with Story 3.4 addressing QA follow-up while keeping hand-offs cohesive.
- [x] Enhancements build on existing scripts without architectural rewrite.

**Risk Assessment:**
- [x] Token cost impact measured and documented via Story 3.2 QA telemetry outputs (`docs/stories/3.2.ingestion-qa-telemetry.md`).
- [x] Telemetry verifies Neo4j/Qdrant schema consistency post-ingestion (see `docs/qa/assessments/3.2-trace-20251002.md`).

**Completeness Check:**
- [x] Success metrics quantifiable (QA gates, telemetry thresholds, lint guard exit codes).
- [x] Dependencies identified (OpenAI quotas, documentation lint tooling, Docker compose services).

## Story Manager Handoff
"Please draft detailed stories referencing:

- Chunking presets and CLI ergonomics should follow GraphRAG docs: see `SimpleKGPipeline` text_splitter overrides and LangChain splitter integration guidance.
- Metadata schema updates must align with `docs/architecture/overview.md#minimal-path-workflow` and `docs/architecture.md#database--collection-schema`; include git commit SHA and relative path when available.
- QA reporting should live under `artifacts/ingestion/` with JSON summaries for chunk/token stats, orphan checks, and embedding validation; integrate into CI via a new pytest module or CLI command.
- Semantic enrichment should reuse GraphRAG's `LLMEntityRelationExtractor` (JSON response format) and update documentation on how retrieval scripts tap semantic nodes.
- Ensure automation (unit + integration) covers default vs. advanced profiles and that new dependencies/rate limits are documented.

Successful delivery means ingestion can flex between documentation dumps and codebases while surfacing quality metrics before export or retrieval."

### Story Override Workflow
- Default rule: do not create the next story while the previous one is not marked `Done`.
- When an override is unavoidable (e.g., downstream scheduling pressure), run `python -m cli.stories --override-incomplete --stories-dir docs/stories --new-story <new_story_path> --reason "<why>"` to:
  - Log actor, timestamp, prior story id, and reason into `docs/bmad/story-overrides.md`.
  - Inject a risk acknowledgement note under `## Dev Notes` in the new story document.
- Overrides require Product Owner awareness and QA follow-up; ensure the `reason` captures mitigating actions and link back to the prior story in status meetings.
