# PRD Requirements

## Functional Requirements
- **FR1:** Ship `docker-compose.neo4j-qdrant.yml` that brings up Neo4j 5.26 with APOC Core and Qdrant latest using persistent named volumes and configurable credentials.
- **FR2:** Provide a bootstrap workflow that installs `neo4j-graphrag[experimental,openai,qdrant]`, validates imports, and documents `.env` variables pointing at the local stack.
- **FR3:** Supply scripts for Neo4j vector index creation and `SimpleKGPipeline` execution targeting sample source material.
- **FR4:** Supply scripts that export chunk embeddings to Qdrant with payload join keys and run a retrieval smoke test via `GraphRAG.search()`.

## Non-Functional Requirements
- **NFR1:** End-to-end local workflow completes within 60 minutes on an 8-core developer laptop.
- **NFR2:** Scripts honour `.env` configuration and never log secrets; failures raise actionable errors with retry guidance.
- **NFR3:** Logging is structured and includes durations for each major operation.
- **NFR4:** Workflows are idempotent—rerunning scripts without cleanup results in consistent graph/vector state.
