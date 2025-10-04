# QA Backlog

| ID | Description | Owner | Notes |
|----|-------------|-------|-------|
| QA-UID-20251002 | Neo4j QA metrics still reference legacy `uid` property; adjust ingestion QA Cypher queries to use active chunk identifiers. | QA / Ingestion Maintainers | Raised during Story 3.4 smoke run (warnings in `scripts/kg_build.py`). Track in upcoming maintenance story. |
| QA-ENV-20251004 | Ensure local QA/diagnostics flows load OpenAI credentials from `.env` or document skip-live stubs to stop false "creds missing" signals. | QA / Platform | Follow-up from Story 2.6 hardening; need helper loader or doc changes once quota restored. |
