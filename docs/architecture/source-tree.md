# Source Tree Blueprint

```text
neo4j-graphrag/
├── AGENTS.md
├── README.md
├── docs/
│   ├── architecture.md
│   ├── prd.md
│   ├── architecture/
│   │   ├── coding-standards.md
│   │   ├── overview.md
│   │   ├── source-tree.md
│   │   └── tech-stack.md
│   ├── bmad/
│   │   └── focused-epics/
│   └── prd/
│       ├── epics.md
│       ├── overview.md
│       ├── requirements.md
│       └── technical-assumptions.md
├── requirements.lock
├── scripts/
│   ├── audit_openai_allowlist.py
│   ├── bootstrap.sh
│   ├── check_docs.py
│   └── kg_build.py
├── src/
│   ├── __init__.py
│   ├── _compat/
│   │   ├── __init__.py
│   │   ├── structlog.py
│   │   └── structlog_shim.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── diagnostics.py
│   │   ├── openai_client.py
│   │   ├── sanitizer.py
│   │   ├── stories.py
│   │   ├── telemetry.py
│   │   └── utils.py
│   ├── fancyrag/
│   │   ├── __init__.py
│   │   ├── cli/
│   │   │   ├── __init__.py
│   │   │   └── kg_build_main.py
│   │   ├── kg/
│   │   │   ├── __init__.py
│   │   │   └── pipeline.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── env.py
│   └── config/
│       ├── __init__.py
│       └── settings.py
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   └── openai_probe/
│   ├── integration/
│   │   └── cli/
│   └── unit/
│       ├── cli/
│       └── config/
└── .github/workflows/
    └── openai-allowlist-audit.yml

`scripts/check_docs.py` provides the documentation lint guard referenced in the architecture overview and CI workflows.
```
## Upcoming Module Layout
The FancyRAG `kg_build.py` refactor is gradually introducing a structured package under `src/fancyrag/`:
- ✅ `cli/kg_build_main.py` — CLI wiring for arguments and entrypoint (`scripts/kg_build.py` now delegates here).
- ✅ `kg/pipeline.py` — Orchestration entrypoint exposing `PipelineOptions` and `run_pipeline()` for reuse.
- ✅ `splitters/caching_fixed_size.py` — Standalone splitter implementation exposing caching factories.
- ✅ `qa/evaluator.py` — QA thresholds, metrics aggregation, and report generation helpers extracted from the pipeline.
- ✅ `qa/report.py` — Markdown/JSON formatting helpers shared by evaluator and pipeline.
- ✅ `utils/paths.py` — Shared repository path helpers consumed by pipeline and QA modules.
- `db/neo4j_queries.py` — Cypher query catalog and wrappers.
- `config/schema.py` plus `utils/env.py` — Schema loading and environment utilities.

Keep this source tree file in sync as modules land; the authoritative breakdown lives in [projects/fancyrag-kg-build-refactor.md](projects/fancyrag-kg-build-refactor.md).
