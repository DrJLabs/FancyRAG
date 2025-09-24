# PRD Overview

## Goals
- Deliver a CLI-driven GraphRAG workflow that reuses existing Neo4j and Qdrant deployments while maintaining isolation.
- Provide operators with ingestion, vector sync, and retrieval pipelines that are reliable and repeatable.
- Ensure retrieval joins vector results with graph context to support grounded answers.
- Document operational guardrails so the team can adopt GraphRAG without ad-hoc experimentation.

## Background Context
Neo4j GraphRAG now offers first-party tooling for graph-aware retrieval. Our organization already maintains Neo4j and Qdrant services; this initiative standardizes how we ingest data, form the knowledge graph, and perform retrievals strictly through CLI workflows with no HTTP exposure in the initial release.

## Change Log
| Date       | Version | Description                                    | Author    |
|------------|---------|------------------------------------------------|-----------|
| 2025-09-24 | 0.1     | Seeded overview shard aligned to PRD baseline | Codex CLI |
