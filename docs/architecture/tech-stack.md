# Technology Stack

| Category            | Technology         | Version      | Purpose                                        | Rationale                                                     |
|---------------------|--------------------|--------------|------------------------------------------------|---------------------------------------------------------------|
| Language            | Python             | 3.12         | Primary implementation language                | Modern LTS, compatible with neo4j-graphrag dependencies       |
| Runtime             | CPython            | 3.12         | Execution environment                          | Stable, widely supported                                     |
| Package             | neo4j-graphrag     | 0.9.x        | Graph-aware retrieval utilities                | First-party Neo4j integration with Qdrant support             |
| Package             | neo4j              | 5.x driver   | Bolt connectivity to Neo4j                     | Official driver with session/transaction helpers              |
| Package             | qdrant-client      | 1.10+        | Vector DB client                               | Native Qdrant client with payload helpers                     |
| Package             | openai             | 1.x          | LLM and embeddings access                       | Supports GPT-4o-mini and text-embedding-3-small               |
| Observability       | structlog/logging  | 24.x         | Structured logging                              | Produces JSON logs suitable for pipeline telemetry            |
| Testing             | pytest             | 8.x          | Unit and integration testing                    | Simple CLI workflow, strong plugin ecosystem                  |
| Packaging           | pip-tools (opt.)   | 7.x          | Dependency locking                              | Enables deterministic installs                                |
| IaC                 | Terraform (opt.)   | 1.8          | Manage secrets and scheduled jobs               | Aligns with existing operational tooling                      |
