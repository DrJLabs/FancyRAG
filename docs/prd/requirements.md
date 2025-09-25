# PRD Requirements

## Functional Requirements
- **FR1:** Provide a reproducible virtual environment install of `neo4j-graphrag[openai,qdrant]` with dependency verification.
- **FR2:** Configure OpenAI generation defaults to `gpt-4.1-mini` with documented fallback to `gpt-4o-mini`; embeddings default to `text-embedding-3-small` (1536 dimensions).
- **FR3:** Provision Neo4j database `graphrag` with a least-privilege user scoped to that database and required APOC procedures.
- **FR4:** Provision Qdrant collection `grag_main_v1` sized to 1536 dimensions, cosine distance, and matching payload schema.
- **FR5:** Use `SimpleKGPipeline` (or successor KG builder) to write pilot entities and relations into Neo4j with validation queries.
- **FR6:** Batch embed text units and upsert to Qdrant with payload fields `{neo4j_id, doc_id, chunk, source}` to guarantee graph join fidelity.
- **FR7:** Expose a CLI workflow that executes `QdrantNeo4jRetriever.search()` and produces grounded LLM answers with retrieved context records.

## Non-Functional Requirements
- **NFR1:** Retrieval P95 latency ≤ 3 seconds for top-k ≤ 10 on the pilot dataset.
- **NFR2:** Batch reindex of 100k chunks completes within 4 hours using retry/backoff strategy.
- **NFR3:** Secrets remain in `.env` files or secret managers; no credentials committed to VCS.
- **NFR4:** Logging/monitoring enables failure triage through persisted CLI logs and periodic backups.
- **NFR5:** Solution runs on Python 3.12 with pinned dependencies to ensure reproducibility.
