# Epic 3: Ingestion Quality Hardening — Adaptive Chunking & QA

## Status
Draft — 2025-10-01. Kickoff after confirming current minimal-path smoke stays green and OpenAI rate limits are stable.

## Epic Goal
Raise the fidelity and observability of the GraphRAG ingestion pipeline so operators can ingest varied corpora (docs, source code, knowledge bases) with tunable chunking, richer metadata, and automated quality checkpoints before export or retrieval.

## Epic Description
**Existing System Context:**
- Epics 1–2 deliver a fixed-size chunking pipeline (`FixedSizeSplitter`, 600/100 overlap), metadata limited to file path + index, and smoke coverage centered on the sample pilot text.
- QA assessments have flagged pending work for managed telemetry and ingestion coverage but local workflows remain the baseline.
- Scripts emit structured logs yet lack per-ingestion health summaries, token histograms, or semantic graph enrichments.

**Enhancement Details:**
- Add configurable splitter presets (text prose, markdown, code) with semantic/recursive options and directory-aware ingestion so larger repositories chunk cleanly.
- Persist metadata and validation artefacts (source path, git commit, checksum, token counts) alongside Neo4j nodes to unlock smarter retrieval filters and regression diffing.
- Build a quality gate that analyses ingestion runs (chunk statistics, orphan checks, embedding validation) and surfaces actionable telemetry in artifacts.
- Optionally enrich the graph with entity/relation extraction using SimpleKGPipeline hooks so downstream agents can navigate both lexical and semantic layers.

**Success criteria:**
- Operators can select chunking profiles via CLI flags or config, run ingestion on heterogeneous content (Markdown + code), and see metadata captured in Neo4j/Qdrant payloads with no manual edits.
- Each ingestion emits a QA report (JSON + human-readable summary) covering chunk/token stats, schema checks, and retry telemetry; pipeline fails fast on severe anomalies.
- Lexical + semantic graph nodes are written in a single run, enabling `ask_qdrant.py` (or future agents) to leverage entity relationships without additional scripts.

## Stories
1. **Story 1:** Introduce adaptive chunking and source-aware ingestion by wiring configurable splitters (FixedSize, RecursiveCharacter, code-aware) into `kg_build.py`, adding CLI presets and directory walkers that capture source metadata and digests per chunk.
2. **Story 2:** Instrument ingestion QA and telemetry by generating post-run reports (token histograms, orphan/node integrity, embedding dimension checks), failing the pipeline on thresholds, and storing sanitized artifacts under `artifacts/ingestion/` with CI hooks.
3. **Story 3:** Layer semantic enrichment and retrieval filters by integrating entity/relation extraction into the pipeline, persisting metadata to Neo4j/Qdrant payloads, and extending `ask_qdrant.py` (or a new CLI flag) to exploit semantic facets during queries.

## Compatibility Requirements
- [ ] CLI defaults must continue to succeed on the existing sample (`docs/samples/pilot.txt`) without additional flags.
- [ ] `.env` contract unchanged; new settings exposed via CLI/config files.
- [ ] Unit + integration tests remain runnable without managed services.

## Risk Mitigation
- **Primary Risk:** Complex chunkers increase runtime/cost on large corpora.
  - **Mitigation:** Provide dry-run mode with token estimates, documented defaults per corpus type, and configurable rate-limit guards.
- **Secondary Risk:** Metadata expansion bloats Neo4j/Qdrant payloads.
  - **Mitigation:** Store hashes/checksums separately from text payload; document pruning strategies.
- **Rollback Plan:** Feature-flag new splitters and semantic extraction; reverting to the fixed-size profile restores prior behaviour.

## Definition of Done
- [ ] All stories delivered with acceptance criteria satisfied.
- [ ] Integration smoke covers at least one mixed-content corpus exercising new profiles.
- [ ] QA reports archived for two consecutive successful runs.
- [ ] Documentation updated (architecture overview, coding standards, QA assessments) with new workflow details.

## Validation Checklist
**Scope Validation:**
- [ ] Epic fits in ≤3 stories with clear, cohesive hand-offs.
- [ ] Enhancements build on existing scripts without architectural rewrite.

**Risk Assessment:**
- [ ] Token cost impact measured and documented.
- [ ] Telemetry verifies Neo4j/Qdrant schema consistency post-ingestion.

**Completeness Check:**
- [ ] Success metrics quantifiable (e.g., QA report thresholds, metadata fields present).
- [ ] Dependencies identified (OpenAI quotas, optional code parsing libs).

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
