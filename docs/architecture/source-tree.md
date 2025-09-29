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
│   └── bootstrap.sh
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
```
