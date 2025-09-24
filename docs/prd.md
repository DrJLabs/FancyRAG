# Neo4j GraphRAG Product Requirements Document (PRD)

## Goals and Background Context
### Goals
- Deliver a CLI-driven GraphRAG workflow that reuses existing Neo4j and Qdrant services while ensuring strict data isolation.
- Provide operators with a reliable pipeline to ingest a pilot corpus, build a knowledge graph, and populate Qdrant vectors with consistent join keys.
- Enable retrieval that combines Qdrant vector hits with Neo4j graph context to answer questions accurately.
- Document operational guardrails (backups, snapshots, scheduling) so adoption can occur without bespoke research.

### Background Context
Neo4j GraphRAG (Python package) now ships first-party utilities for knowledge-graph aware retrieval. The organization already operates Neo4j and Qdrant clusters and wants to leverage them without exposing new HTTP APIs in v1. Prior attempts relied on ad-hoc scripts with uneven privilege separation. This PRD targets a disciplined CLI workflow that standardizes installation, ingestion, retrieval, and operations while keeping the footprint minimal.

### Change Log
| Date       | Version | Description                                                    | Author     |
|------------|---------|----------------------------------------------------------------|------------|
| 2025-09-24 | 0.1     | Rebuilt PRD in BMAD format; aligned goals, requirements, epics | Codex CLI  |

## Requirements
### Functional
- **FR1:** Provide a reproducible virtual environment install of `neo4j-graphrag[openai,qdrant]` with dependency verification.
- **FR2:** Configure OpenAI generation defaults to `gpt-4o-mini` (128K context) with optional override to `gpt-4.1-mini`; embeddings default to `text-embedding-3-small` (1536 dimensions).
- **FR3:** Provision Neo4j database `graphrag` with a least-privilege user scoped to that database and required APOC procedures.
- **FR4:** Provision Qdrant collection `grag_main_v1` sized to 1536 dimensions, cosine distance, and matching payload schema.
- **FR5:** Use `SimpleKGPipeline` (or successor KG builder) to write pilot entities and relations into Neo4j with validation queries.
- **FR6:** Batch embed text units and upsert to Qdrant with payload fields `{neo4j_id, doc_id, chunk, source}` to guarantee graph join fidelity.
- **FR7:** Expose a CLI workflow that executes `QdrantNeo4jRetriever.search()` and produces grounded LLM answers with retrieved context records.

### Non Functional
- **NFR1:** Retrieval P95 latency ≤ 3 seconds for top-k ≤ 10 on the pilot dataset.
- **NFR2:** Batch reindex of 100k chunks completes within 4 hours using retry/backoff strategy.
- **NFR3:** Secrets remain in `.env` files or secret managers; no credentials committed to VCS.
- **NFR4:** Logging/monitoring enables failure triage through persisted CLI logs and periodic backups.
- **NFR5:** Solution runs on Python 3.12 with pinned dependencies to ensure reproducibility.

## User Interface Design Goals
- **Overall UX Vision:** Operate entirely via documented CLI commands and scripts; outputs must be copy-paste friendly.
- **Key Interaction Paradigms:** Command-line prompts, environment variables, and scripted pipelines; no GUI.
- **Core Screens and Views:** N/A (CLI only); focus on terminal output sections for install, ingestion, retrieval verification, and operational status.
- **Accessibility:** N/A — CLI inherits terminal accessibility features.
- **Branding:** Adopt standard corporate CLI color palette/logging format if available; otherwise plain-text output.
- **Target Device and Platforms:** Web-responsive UI not required; Linux/macOS shell environments.

## Technical Assumptions
- **Repository Structure:** Monorepo containing CLI scripts, infrastructure automation, and documentation.
- **Service Architecture:** Single Python CLI application integrating with external Neo4j and Qdrant services (monolithic control plane).
- **Testing Requirements:** Unit tests for helper functions; integration smoke tests for retriever and ingestion scripts using sandbox data.
- **Additional Technical Assumptions and Requests:**
  - Python 3.12 interpreter with `pip-tools` or `uv` optional for lockfile generation.
  - Qdrant reachable via internal network with API key authentication; TLS termination managed upstream.
  - Neo4j reachable via Bolt protocol; `neo4j` Python driver pinned to latest 5.x compatible version.
  - Logging via `structlog` or standard library logging with JSON formatter for ingestion/retrieval scripts.

## Epic List
| Epic | Title                        | Goal Statement                                                                 |
|------|------------------------------|--------------------------------------------------------------------------------|
| 1    | Environment & Workspace      | Establish reproducible Python environment, install GraphRAG package, verify imports. |
| 2    | Models & Vectors             | Configure OpenAI models and embeddings with guardrails and probes.             |
| 3    | Neo4j Isolation              | Provision dedicated Neo4j database/user with constraints ready for KG ingest. |
| 4    | Qdrant Collection            | Stand up versioned Qdrant collection with security and retention configured.  |
| 5    | Knowledge Graph Build        | Ingest pilot corpus into Neo4j using KG pipeline and validate graph quality.  |
| 6    | Vector Upsert                | Embed text units and upsert vectors to Qdrant with graph join metadata.       |
| 7    | Retrieval Validation         | Execute retriever searches and confirm grounded answers across scenarios.     |
| 8    | Security & Backups           | Implement credential rotation, snapshots, and restore drills.                 |
| 9    | Scheduling & Cost Control    | Optional automation for periodic refreshes with spend and latency tracking.   |
| 10   | Documentation & Change Mgmt  | Maintain runbooks, configs, and change history for onboarding & audits.       |

## Epic Details
### Epic 1 — Environment & Workspace
**Goal:** Provide a dependable development shell that standardizes Python versioning, dependencies, and environment configuration.
- Create and document virtual environment bootstrap (`python3 -m venv`, `pip install "neo4j-graphrag[openai,qdrant]"`).
- Record package versions and validate import via `python -c "import neo4j_graphrag"`.
- Publish sample `.env` template listing required variables.

### Epic 2 — Models & Vectors
**Goal:** Ensure LLM and embedding models are configured, tested, and guarded for production-like reliability.
- Set `OPENAI_MODEL=gpt-4o-mini`; allow override to `gpt-4.1-mini` for extended context scenarios.
- Validate embeddings return 1536-length vectors; document guardrails for batch sizes and retries.
- Capture cost/latency expectations for each model variant.

### Epic 3 — Neo4j Isolation
**Goal:** Guarantee least-privilege access and graph readiness in Neo4j.
- Automate creation of database `graphrag` and user `graphrag` with editor role.
- Apply uniqueness constraints/indexing for node identifiers used by the KG pipeline.
- Document required APOC features and health checks.

### Epic 4 — Qdrant Collection
**Goal:** Prepare Qdrant for production-grade vector storage with audit-ready configuration.
- Provision `grag_main_v1` (size 1536, cosine distance) via API or CLI.
- Secure API key management and network policies; define snapshot cadence.
- Validate readiness via API introspection and test inserts.

### Epic 5 — Knowledge Graph Build
**Goal:** Produce a representative knowledge graph aligned with PRD goals.
- Run `SimpleKGPipeline` (or successor) on curated pilot corpus.
- Tune schema mappings and pruning thresholds; verify node/edge counts exceed baseline.
- Provide Cypher queries for sampling graph quality.

### Epic 6 — Vector Upsert
**Goal:** Synchronize text embeddings into Qdrant with traceable join metadata.
- Batch process text units with backpressure handling; log throughput metrics.
- Ensure payload contains `neo4j_id`, `doc_id`, `chunk`, `source`; enforce idempotent upserts.
- Cross-check Qdrant counts against source documents.

### Epic 7 — Retrieval Validation
**Goal:** Deliver trustworthy question-answering via combined vector + graph retrieval.
- Configure `QdrantNeo4jRetriever` with correct ID properties and database selectors.
- Validate retrieval quality across at least five representative questions (top-k ≤ 10).
- Capture transcripts showing grounded context and final LLM response.

### Epic 8 — Security & Backups
**Goal:** Harden operations and support disaster recovery.
- Rotate Neo4j and Qdrant credentials; document storage locations and rotation cadence.
- Script Qdrant snapshots and Neo4j backups; perform a restore drill in sandbox.
- Document incident escalation paths.

### Epic 9 — Scheduling & Cost Control
**Goal:** Automate refresh workflows while managing spend.
- Optional systemd timer or cron to re-run ingestion/upsert pipelines nightly.
- Track token usage, latency, and storage growth; define thresholds for alerting.
- Provide remediation steps when metrics exceed budgets.

### Epic 10 — Documentation & Change Management
**Goal:** Create durable knowledge assets and governance.
- Maintain versioned configs, prompt libraries, and onboarding checklists.
- Update this PRD change log and supporting architecture/prd shards upon material changes.
- Ensure onboarding time ≤ 1 hour via curated runbook and repository README updates.
