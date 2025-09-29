# Technology Stack

| Category            | Technology                | Version          | Purpose                                        | Rationale                                                     |
|---------------------|---------------------------|------------------|------------------------------------------------|---------------------------------------------------------------|
| Language            | Python                    | 3.12             | Primary implementation language                | Modern LTS, compatible with neo4j-graphrag dependencies       |
| Runtime             | CPython                   | 3.12             | Execution environment                          | Stable, widely supported                                     |
| Package             | neo4j-graphrag            | 0.10.x           | Graph-aware retrieval utilities                | Provides `SimpleKGPipeline`, vector index helpers, retrievers |
| Package             | neo4j                     | 5.28.x driver    | Bolt connectivity to Neo4j                     | Official driver with async session/transaction helpers        |
| Package             | qdrant-client             | ≥ 1.8            | Vector DB client                               | Native Qdrant client with payload helpers                     |
| Package             | openai                    | ≥ 1.31           | LLM and embeddings access                       | Supports GPT-4o family and `text-embedding-3-small`           |
| Container Runtime   | Docker Compose            | 2.x              | Orchestrate local Neo4j + Qdrant stack          | Simplifies repeatable developer environment                   |
| Observability       | structlog / logging (JSON)| 24.x             | Structured logging                              | Produces JSON logs suitable for pipeline telemetry            |
| Testing             | pytest                    | 8.x              | Unit and integration testing                    | Simple CLI workflow, strong plugin ecosystem                  |
| Packaging           | pip-tools / uv (optional) | latest           | Dependency locking                              | Enables deterministic installs                                |
