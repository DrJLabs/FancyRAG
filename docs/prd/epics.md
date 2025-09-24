# PRD Epic Catalog

| Epic | Title                       | Goal Statement                                                                 |
|------|-----------------------------|--------------------------------------------------------------------------------|
| 1    | Environment & Workspace     | Establish reproducible Python environment, install GraphRAG package, verify imports. |
| 2    | Models & Vectors            | Configure OpenAI models and embeddings with guardrails and probes.             |
| 3    | Neo4j Isolation             | Provision dedicated Neo4j database/user with constraints ready for KG ingest. |
| 4    | Qdrant Collection           | Stand up versioned Qdrant collection with security and retention configured.  |
| 5    | Knowledge Graph Build       | Ingest pilot corpus into Neo4j using KG pipeline and validate graph quality.  |
| 6    | Vector Upsert               | Embed text units and upsert vectors to Qdrant with graph join metadata.       |
| 7    | Retrieval Validation        | Execute retriever searches and confirm grounded answers across scenarios.     |
| 8    | Security & Backups          | Implement credential rotation, snapshots, and restore drills.                 |
| 9    | Scheduling & Cost Control   | Optional automation for periodic refreshes with spend and latency tracking.   |
| 10   | Documentation & Change Mgmt | Maintain runbooks, configs, and change history for onboarding & audits.       |
