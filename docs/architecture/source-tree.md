# Fancryrag Source Tree Blueprint

This layout describes the expected structure of the repository. New files should follow these conventions unless a story specifies otherwise.

```
/
├── AGENTS.md                     # Project-wide agent instructions
├── docker-compose.yml            # Container orchestration for Neo4j and MCP server
├── Makefile                      # Operational targets (indexing, ingest, counts)
├── pyproject.toml                # Python package metadata and dependencies
├── uv.lock                       # Resolved dependency lockfile
├── README.md                     # Quick start guide for the baseline
├── .env.example                  # Sample environment configuration
├── docs/
│   ├── prd.md                    # Product Requirements Document
│   ├── architecture.md           # System architecture blueprint
│   └── architecture/
│       ├── coding-standards.md   # Code quality and style conventions
│       ├── tech-stack.md         # Approved technologies and versions
│       └── source-tree.md        # (This file) repository layout guidance
├── pipelines/
│   └── kg_ingest.yaml            # GraphRAG pipeline configuration for ingestion
├── scripts/                      # Helper scripts (documentation lint, index creation)
│   └── check_docs.py             # Documentation lint guard (scripts/check_docs.py)
├── servers/                      # (Planned) FastMCP server implementations
├── src/
│   └── fancryrag/                # Python package root (add modules under here)
├── tools/
│   ├── run_pipeline.py           # CLI entrypoint for running ingestion
│   └── discover_classes.py       # Utility helpers invoked by the pipeline
├── tests/                        # (Planned) Pytest suites mirroring src structure
├── .github/                      # (Recommended) Issue/PR templates and workflows
└── .ai/
    └── debug-log.md              # Dev agent operational log
```

## Directory Guidelines
- Place application logic in `src/fancryrag/`. Subpackages should reflect bounded contexts (e.g., `retrieval`, `ingest`, `config`).
- Scripts intended for direct execution belong under `tools/` or `scripts/` with CLI argument parsing and docstrings.
- Infrastructure automation (Terraform, Helm, etc.) should live under `infra/` or `.bmad-infrastructure-devops/` when added.
- All tests mirror the `src/` hierarchy within `tests/`, using identical module names with `test_` prefixes.
- Static assets (diagrams, fixture data) belong under `docs/assets/` or `tests/fixtures/` depending on usage.

Keep this document updated as new directories or patterns emerge.
