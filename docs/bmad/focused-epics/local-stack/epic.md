# Epic 2: Local GraphRAG Minimal Path — Dockerized Stack

## Status
Done — 2025-10-01. All stories (2.1–2.5) delivered local Docker stack, automation scripts, documentation, and CI smoke coverage. Follow-up: keep compose smoke enabled with managed OpenAI credentials.

## Epic Goal
Stand up a fully local GraphRAG evaluation stack that operators can provision end-to-end in under one hour using Docker Compose, the official `neo4j-graphrag` tooling, and scripted hand-offs between Neo4j, Qdrant, and OpenAI.

## Epic Description
**Existing System Context:**
- Current relevant functionality: Environment bootstrap scripts target external Neo4j and Qdrant clusters with manual configuration steps; no project-owned services exist.
- Technology stack: Python 3.12, `neo4j-graphrag[experimental,openai,qdrant]`, Docker Compose, Neo4j ≥ 5.26 with APOC Core, Qdrant ≥ 1.8, OpenAI GPT-4.1-mini (fallback `gpt-4o-mini`) and `text-embedding-3-small`.
- Integration points: Scripts must align with the official Neo4j GraphRAG package APIs (`SimpleKGPipeline`, `create_vector_index`, `QdrantNeo4jRetriever`) and reuse the `.env` surface established by the workspace epic.

**Enhancement Details:**
- What's being added/changed: Introduce a project-scoped Docker Compose stack for Neo4j and Qdrant, plus automation scripts that create the Neo4j vector index, run the GraphRAG KG builder, mirror embeddings into Qdrant, and execute a retrieval smoke test.
- How it integrates: New scripts live under `scripts/` and reuse the shared configuration module so they remain compatible with future managed deployments. Docker Compose defaults point to local volumes but honour environment overrides to enable CI use.
- Success criteria: A fresh clone can run `docker compose up`, execute the provided Python scripts, and receive a grounded answer from `GraphRAG.search()` without touching external infrastructure.

## Stories
1. **Story 1:** Author `docker-compose.neo4j-qdrant.yml` and supporting documentation that bootstraps Neo4j (with APOC enabled) and Qdrant using project-managed volumes and `.env` variables.
2. **Story 2:** Implement scripts to create the Neo4j vector index and drive `SimpleKGPipeline` end-to-end against sample content, emitting structured logs and retries consistent with coding standards.
3. **Story 3:** Build the Qdrant export + retrieval smoke pipeline that batches embeddings to Qdrant, validates join keys, and exercises `GraphRAG` to confirm a grounded answer path.

## Compatibility Requirements
- [x] Existing APIs remain unchanged
- [x] Database schema changes are backward compatible
- [x] UI changes follow existing patterns
- [x] Performance impact is minimal

## Risk Mitigation
- **Primary Risk:** Local containers drift from production configuration, leading to incompatibility when promoting to managed services.
- **Mitigation:** Mirror image tags, APOC configuration, and index dimensions used in production; gate scripts behind configuration files with documented overrides.
- **Rollback Plan:** Stop and remove the local Docker stack, reverting to external service configuration documented in the workspace epic if validation fails.

## Definition of Done
- [x] All stories completed with acceptance criteria met
- [x] Existing functionality verified through testing
- [x] Integration points working correctly
- [x] Documentation updated appropriately
- [x] No regression in existing features

## Validation Checklist
**Scope Validation:**
- [x] Epic can be completed in 1-3 stories maximum
- [x] No architectural documentation is required
- [x] Enhancement follows existing patterns
- [x] Integration complexity is manageable

**Risk Assessment:**
- [x] Risk to existing system is low
- [x] Rollback plan is feasible
- [x] Testing approach covers existing functionality
- [x] Team has sufficient knowledge of integration points

**Completeness Check:**
- [x] Epic goal is clear and achievable
- [x] Stories are properly scoped
- [x] Success criteria are measurable
- [x] Dependencies are identified

## Story Manager Handoff
"Please draft focused user stories for this epic with the following anchors:

- We now own a Docker Compose definition (`docker-compose.neo4j-qdrant.yml`) that launches Neo4j 5.26 with APOC Core and Qdrant latest, exposing `7474/7687` and `6333` respectively.
- Scripts to deliver: `scripts/create_vector_index.py`, `scripts/kg_build.py`, and `scripts/export_to_qdrant.py` plus `scripts/ask_qdrant.py` for retrieval validation. All scripts must load shared settings and honour `.env` values (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `QDRANT_URL`, `OPENAI_API_KEY`).
- Acceptance hinges on successfully generating embeddings via `SimpleKGPipeline`, replicating them into Qdrant with payload join keys, and returning a grounded answer using `GraphRAG` with `QdrantNeo4jRetriever`.
- Preserve compatibility with the official `neo4j-graphrag` public APIs—no forks or custom patches. Align logging, retries, and error handling with `docs/architecture/coding-standards.md`.

We only succeed when `docker compose up` + the scripted workflow produce a complete KG ingestion and retrieval loop locally without touching managed services."

### Story Override Workflow
- Default rule: do not create the next story while the previous one is not marked `Done`.
- When an override is unavoidable (e.g., downstream scheduling pressure), run `python -m cli.stories --override-incomplete --stories-dir docs/stories --new-story <new_story_path> --reason "<why>"` to:
  - Log actor, timestamp, prior story id, and reason into `docs/bmad/story-overrides.md`.
  - Inject a risk acknowledgement note under `## Dev Notes` in the new story document.
- Overrides require Product Owner awareness and QA follow-up; ensure the `reason` captures mitigating actions and link back to the prior story in status meetings.
