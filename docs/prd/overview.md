# PRD Overview

## Goals
- Deliver a self-contained GraphRAG evaluation stack that relies on local Docker containers for Neo4j (graph + vectors) plus the official `neo4j-graphrag` package.
- Provide scripted ingestion, vector synchronization, and retrieval commands that operators can run end-to-end on a single host.
- Maintain parity with production-grade settings so the same scripts can later point at managed services.

## Background Context
Teams previously depended on pre-provisioned databases, which slowed experimentation and onboarding. Version 1 shifts to a project-owned stack: `docker compose` launches Neo4j 5.26.12 (APOC Core) and the MCP service from `docker-compose.yml`, while Python scripts powered by `neo4j-graphrag[experimental,openai]` take care of the KG build and retrieval smoke tests. Qdrant export/retrieval remains a legacy optional workflow for teams still using `docker-compose.neo4j-qdrant.yml`. Configuration lives in `.env`, letting operators swap endpoints without rewriting code.

## Upcoming Work
- **Epic 4 — FancyRAG `kg_build.py` Monolith Decomposition:** Planning underway to split the monolithic script into a `src/fancyrag/` package. Track progress via the [project brief](projects/fancyrag-kg-build-refactor/project-brief.md), [PRD shard](projects/fancyrag-kg-build-refactor/prd.md), and [Epic 4 handoff](../bmad/focused-epics/kg-build-refactor/epic.md).
- **Epic 5 — FancyRAG Service Hardening:** Upcoming effort to operationalise the refactored modules with typed configuration, automation, QA telemetry, and observability. See the [Epic 5 brief](../bmad/focused-epics/fancyrag-service-hardening/epic.md) for scope and story sequencing.
- **Testing Alignment:** Epic 3 hardening complete with QA telemetry and chunking presets; refactor work must preserve these guarantees.

## Change Log
| Date       | Version | Description                                       | Author    |
|------------|---------|---------------------------------------------------|-----------|
| 2025-09-24 | 0.1     | Seeded overview shard aligned to PRD baseline    | Codex CLI |
| 2025-09-28 | 0.2     | Narrowed scope to local Docker + scripted path   | Codex CLI |
| 2025-10-02 | 0.3     | Linked upcoming `kg_build.py` refactor planning artifacts | Codex CLI |
