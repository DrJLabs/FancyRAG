# PRD Overview

## Goals
- Deliver a self-contained GraphRAG evaluation stack that relies on local Docker containers for Neo4j and Qdrant plus the official `neo4j-graphrag` package.
- Provide scripted ingestion, vector synchronization, and retrieval commands that operators can run end-to-end on a single host.
- Maintain parity with production-grade settings so the same scripts can later point at managed services.

## Background Context
Teams previously depended on pre-provisioned databases, which slowed experimentation and onboarding. Version 1 shifts to a project-owned stack: `docker compose` launches Neo4j 5.26 (APOC Core) and Qdrant latest, while Python scripts powered by `neo4j-graphrag[experimental,openai,qdrant]` take care of the KG build, vector export, and retrieval smoke tests. Configuration lives in `.env`, letting operators swap endpoints without rewriting code.

## Change Log
| Date       | Version | Description                                       | Author    |
|------------|---------|---------------------------------------------------|-----------|
| 2025-09-24 | 0.1     | Seeded overview shard aligned to PRD baseline    | Codex CLI |
| 2025-09-28 | 0.2     | Narrowed scope to local Docker + scripted path   | Codex CLI |
