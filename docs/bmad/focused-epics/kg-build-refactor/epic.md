# Epic 4: FancyRAG `kg_build.py` Monolith Decomposition

## Status
In Progress — 2025-10-04. Stories 4.1 and 4.2 delivered; downstream refactor slices queued next.

## Epic Goal
Break the `scripts/kg_build.py` monolith into composable modules under `src/fancyrag/`, leaving a thin CLI wrapper and establishing guardrails that improve maintainability, testability, and future extensibility.

## Epic Description
**Existing System Context:**
- The current KG build flow lives in a single script (`scripts/kg_build.py`) that blends CLI parsing, pipeline orchestration, QA reporting, and database utilities.
- Repository layout lacks a dedicated `src/fancyrag/` package, making reuse, testing, and dependency isolation cumbersome.
- QA logic, splitter implementation, and Neo4j helpers are tightly coupled, slowing defect isolation and inflation of blast radius for changes.

**Enhancement Highlights:**
- Introduce a `src/fancyrag/` package with focused submodules (`cli`, `kg`, `qa`, `splitters`, `db`, `config`, `utils`) to encapsulate responsibilities.
- Carve out CLI argument parsing into `cli/kg_build_main.py`, centralize pipeline coordination in `kg/pipeline.py`, and relocate splitter, QA, Neo4j, and schema helpers into their own modules.
- Establish coding guardrails: per-module single responsibility, 200-400 LOC targets, strongly typed interfaces, and centralized environment utilities (`utils/env.py`).
- Seed unit tests per module plus an end-to-end CLI smoke to ensure parity with the pre-refactor behaviour.

**Outcome:**
- FancyRAG contributors gain a maintainable, well-scoped package structure with clear extension points and higher test coverage.
- The CLI entry point remains familiar while delegating business logic to reusable modules, enabling faster iteration on pipeline stages.
- QA and reporting components become individually testable, supporting future automation and telemetry enhancements.

## Stories
1. ☑ **Story 4.1:** Extract CLI wiring into `src/fancyrag/cli/kg_build_main.py` and update `scripts/kg_build.py` to call the packaged entry point (`docs/stories/4.1.cli-kg-build-main.md`).
2. ☑ **Story 4.2:** Move pipeline orchestration and dependency validation into `src/fancyrag/kg/pipeline.py` with cohesive interfaces (`docs/stories/4.2.kg-pipeline.md`).
3. ☐ **Story 4.3:** Relocate the caching splitter into `src/fancyrag/splitters/caching_fixed_size.py` with reusable configuration hooks (`docs/stories/4.3.caching-fixed-size-splitter.md`).
4. ☐ **Story 4.4:** Separate QA evaluation logic into `src/fancyrag/qa/evaluator.py` and establish thresholds/totals utilities (`docs/stories/4.4.qa-evaluator.md`).
5. ☑ **Story 4.5:** Create `src/fancyrag/qa/report.py` for JSON/Markdown reporting and wire it into the pipeline (`docs/stories/4.5.qa-report.md`).
6. ☐ **Story 4.6:** Extract Neo4j query helpers into `src/fancyrag/db/neo4j_queries.py` and update call sites (`docs/stories/4.6.neo4j-queries.md`).
7. ☐ **Story 4.7:** Stand up schema utilities under `src/fancyrag/config/schema.py` plus shared environment helpers in `src/fancyrag/utils/env.py` (`docs/stories/4.7.schema-and-env.md`).
8. ☐ **Story 4.8:** Author module-level unit tests and an end-to-end CLI smoke validating parity with the legacy script (`docs/stories/4.8.tests-and-smoke.md`).

## Compatibility Requirements
- [ ] CLI invocation (`python scripts/kg_build.py`) must preserve flags, defaults, and output structure.
- [ ] Existing `.env` contract and config discovery continue to function without changes for operators.
- [ ] Neo4j schema alignment maintained; no breaking changes to node/relationship definitions or QA metrics output schema.

## Risk Mitigation
- **Primary Risk:** Refactor introduces regressions in ingestion QA or pipeline orchestration.
  - **Mitigation:** Develop module-focused unit tests before removing legacy logic, then run smoke test comparing outputs.
- **Secondary Risk:** Module boundaries create circular dependencies or ambiguous ownership.
  - **Mitigation:** Document module contracts in `src/fancyrag/__init__.py` and enforce import direction (CLI → pipeline → QA/DB helpers).
- **Rollback Plan:** Retain the original `scripts/kg_build.py` implementation in git history; feature branch can revert to monolith if smoke test fails.

## Definition of Done
- [ ] All eight stories completed with documented acceptance criteria.
- [ ] `scripts/kg_build.py` reduced to CLI bridge invoking packaged entry point.
- [ ] Unit tests cover CLI wrapper, pipeline orchestration seams, and QA/report utilities.
- [ ] CLI smoke test passes against sample corpus and matches legacy metrics.
- [ ] Developer documentation updated (project brief, PRD shards, architecture diagram) with new module layout.

## Validation Checklist
**Scope Validation:**
- [ ] Epic deliverable fits within 6–8 tightly scoped stories.
- [ ] Changes focus on refactor/restructure without adding new product features.
- [ ] Module boundaries align with coding standards and guardrails outlined in the project brief.

**Risk Assessment:**
- [ ] Regression risk tracked via expanded tests and telemetry sanity checks.
- [ ] Rollback plan validated with branch strategy and git tags before merge.
- [ ] Team has buy-in on coding guardrails and review checklist updates.

**Completeness Check:**
- [ ] Success metrics measurable via test coverage and maintainability indicators (module size, cyclomatic complexity).
- [ ] Dependencies captured (Neo4j, Qdrant, OpenAI credentials, new package path updates).
- [ ] Story sequencing avoids blocking dependencies (CLI wiring first, tests last).

## Story Manager Handoff
"Please draft the detailed stories referenced above, grounding them in the new project brief, PRD, and architecture doc:

- Validate every module extraction against the guardrails (single responsibility, typed interfaces, LOC targets).
- Ensure CLI behaviour remains unchanged for operators while delegating logic to `src/fancyrag` modules.
- Update documentation and tests as part of the refactor—unit tests per module plus a CLI smoke that exercises the full pipeline via Dockerized services.
- Coordinate with QA to capture baseline metrics before refactor and compare after module split.
- Track environment utility centralization to avoid drift between `.env`, pipeline config, and tests.

Successful delivery means the FancyRAG KG build flow becomes modular, easier to test, and ready for future enhancements without reworking a monolithic script."

### Story Override Workflow
- Default rule: do not create the next story while the previous one is not marked `Done`.
- When an override is unavoidable (e.g., downstream scheduling pressure), run `python -m cli.stories --override-incomplete --stories-dir docs/stories --new-story <new_story_path> --reason "<why>"` to:
  - Log actor, timestamp, prior story id, and reason into `docs/bmad/story-overrides.md`.
  - Inject a risk acknowledgement note under `## Dev Notes` in the new story document.
- Overrides require Product Owner awareness and QA follow-up; ensure the `reason` captures mitigating actions and link back to the prior story in status meetings.
