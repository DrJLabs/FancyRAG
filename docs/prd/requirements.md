# PRD Requirements

## Functional Requirements
- **FR1:** Ship `docker-compose.yml` that brings up Neo4j 5.26.12 with APOC Core (graph + vector indexing) and the MCP service using persistent named volumes and configurable credentials.
- **FR2:** Provide a bootstrap workflow that installs `neo4j-graphrag[experimental,openai]`, validates imports, and documents `.env` variables pointing at the local stack. Qdrant extras remain optional for legacy export workflows.
- **FR3:** Supply scripts for Neo4j vector index creation and `SimpleKGPipeline` execution targeting sample source material.
- **FR4:** Provide a retrieval smoke test that uses Neo4j-backed vectors end-to-end. Legacy Qdrant export + `QdrantNeo4jRetriever` validation is optional for teams still on the Qdrant stack.

## Non-Functional Requirements
- **NFR1:** End-to-end local workflow completes within 60 minutes on an 8-core developer laptop.
- **NFR2:** Scripts honour `.env` configuration and never log secrets; failures raise actionable errors with retry guidance.
- **NFR3:** Logging is structured and includes durations for each major operation.
- **NFR4:** Workflows are idempotent—rerunning scripts without cleanup results in consistent graph/vector state.
