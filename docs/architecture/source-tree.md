# Source Tree Blueprint

```text
neo4j-graphrag/
├── docker-compose.neo4j-qdrant.yml
├── docs/
│   ├── architecture.md
│   ├── prd.md
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── tech-stack.md
│   │   ├── source-tree.md
│   │   └── coding-standards.md
│   └── prd/
│       ├── overview.md
│       ├── requirements.md
│       ├── epics.md
│       └── technical-assumptions.md
├── scripts/
│   ├── bootstrap.sh
│   ├── create_vector_index.py
│   ├── kg_build.py
│   ├── export_to_qdrant.py
│   ├── ask_qdrant.py
│   └── backup-qdrant.sh
├── src/
│   ├── cli/
│   │   ├── ingest.py
│   │   ├── vectors.py
│   │   ├── search.py
│   │   └── openai_client.py
│   ├── pipelines/
│   │   ├── kg_builder.py
│   │   └── vector_upsert.py
│   └── config/
│       ├── settings.py
│       └── logging.py
├── tests/
│   ├── unit/
│   └── integration/
└── pyproject.toml
```
