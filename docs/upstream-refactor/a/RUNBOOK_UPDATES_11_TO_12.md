# RUNBOOK_UPDATES_11_TO_12.md

## What changed (based on verified recommendations)

### 1) Makefile `make index` safety (PR-01)
- **Verified evidence in inputs:** Makefile index path is described as drop/recreate and defaults to `${EMBEDDING_DIMENSIONS:-1024}`【23†L16-L24】 and is treated as destructive in DR_08 B.3.3.
- **Runbook update:** PR-01 now explicitly requires refactoring `make index` to delegate to `uv run python scripts/create_vector_index.py` (idempotent, mismatch-safe), and moving any destructive drop/recreate behind an explicit opt-in target (e.g., `make index-recreate`).

### 2) Clarify `src/cli/**` vs `src/fancyrag/cli/**` ownership (PR-08)
- **Verified evidence in inputs:** Baseline inventory references both root `cli/utils.py` and package `fancyrag/cli/kg_build_main.py`【10†L11-L19】【31†L25-L34】, and DR_05/DR_06 refer to both layouts.
- **Runbook update:** PR-08 acceptance criteria now requires documentation to explicitly clarify whether `cli` remains an app-entrypoint package or is relocated under `src/fancyrag/`.

### 3) Embedding dimension defaults (“golden default”) (PR-01)
- **Verified evidence in inputs:** Script defaults cite “default 1024” and Makefile defaults cite 1024【28†L52-L60】【28†L69-L77】【23†L16-L24】.
- **Runbook update:** PR-01 continues to require a single consistent default across `.env.example`, `Makefile`, and `scripts/create_vector_index.py`, while keeping smoke overrides deterministic.

## Files updated
- `EXECUTION_RUNBOOK_12.md` (updated runbook)
- `PR_QUEUE_12.json` (updated machine-readable queue)
