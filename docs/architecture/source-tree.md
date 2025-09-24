# Source Tree Blueprint

```text
neo4j-graphrag/
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
├── src/
│   ├── cli/
│   │   ├── ingest.py
│   │   ├── vectors.py
│   │   └── search.py
│   ├── pipelines/
│   │   ├── kg_builder.py
│   │   └── vector_upsert.py
│   └── config/
│       ├── settings.py
│       └── logging.py
├── scripts/
│   ├── bootstrap.sh
│   └── backup-qdrant.sh
├── tests/
│   ├── unit/
│   └── integration/
└── pyproject.toml
```
