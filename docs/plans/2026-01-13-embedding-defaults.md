# Embedding Default Dimension Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the project’s default embedding dimension to 1024 for the local embedding model and update all “default dimension” references to avoid mismatches.

**Architecture:** Update the default embedding dimension constant and propagate it to env examples and Make targets that use defaults. Adjust tests and fixtures so default-path behavior expects 1024, while OpenAI-specific tests explicitly override to 1536. Update documentation that currently describes the default as 1536 to reflect the new 1024 default.

**Tech Stack:** Python, pytest, Makefile, markdown docs.

---

### Task 1: Update default-focused tests/fixtures to 1024 (intentional failing changes first)

**Files:**
- Modify: `tests/unit/scripts/test_create_vector_index.py`
- Modify: `tests/integration/local_stack/test_minimal_path_smoke.py`
- Modify: `tests/unit/fancyrag/kg/test_pipeline.py`
- Modify: `tests/fixtures/minimal_path/kg_build_success.json`

**Step 1: Write the failing test updates**

- In `tests/unit/scripts/test_create_vector_index.py`, change the default `dimensions` in `_match_record` from `1536` to `1024`, update assertions expecting `1536` to `1024`, and adjust the mismatch test to use a non-default (e.g., `1536`) so it still fails appropriately.
- In `tests/integration/local_stack/test_minimal_path_smoke.py`, change the fallback `"1536"` to `"1024"`.
- In `tests/unit/fancyrag/kg/test_pipeline.py`, replace default `embedding_dimensions=1536` with `1024`.
- In `tests/fixtures/minimal_path/kg_build_success.json`, update `embedding_dimensions` to `1024`.

**Step 2: Run a focused test to confirm failures (expected before code change)**

Run: `uv run pytest tests/unit/scripts/test_create_vector_index.py -v`
Expected: FAIL due to default dimension mismatch until code defaults are updated.

**Step 3: Commit**

```bash
git add tests/unit/scripts/test_create_vector_index.py \
  tests/integration/local_stack/test_minimal_path_smoke.py \
  tests/unit/fancyrag/kg/test_pipeline.py \
  tests/fixtures/minimal_path/kg_build_success.json
git commit -m "test: align default embedding dimension expectations"
```

---

### Task 2: Make OpenAI tests explicit about 1536 overrides (keep OpenAI behavior)

**Files:**
- Modify: `tests/unit/cli/test_openai_client.py`
- Modify: `tests/unit/cli/test_openai_probe.py`
- Modify: `tests/integration/cli/test_openai_probe_cli.py`
- Modify: `tests/fixtures/openai_probe/probe.json`

**Step 1: Update OpenAI tests to set explicit override**

- Add `OPENAI_EMBEDDING_DIMENSIONS=1536` (or `EMBEDDING_DIMENSIONS=1536`) to test environments or monkeypatches so OpenAI stubs returning 1536 are intentional and not default-driven.
- Update `tests/fixtures/openai_probe/probe.json` if the settings section changes due to explicit override.

**Step 2: Run a focused test to validate**

Run: `uv run pytest tests/unit/cli/test_openai_probe.py -v`
Expected: PASS with explicit override set.

**Step 3: Commit**

```bash
git add tests/unit/cli/test_openai_client.py \
  tests/unit/cli/test_openai_probe.py \
  tests/integration/cli/test_openai_probe_cli.py \
  tests/fixtures/openai_probe/probe.json
git commit -m "test: make OpenAI embedding dimension explicit"
```

---

### Task 3: Update default dimension constant and env/Make defaults to 1024

**Files:**
- Modify: `src/config/settings.py`
- Modify: `.env.example`
- Modify: `Makefile`

**Step 1: Update default constant**

- Change `DEFAULT_EMBEDDING_DIMENSIONS = 1536` → `1024` in `src/config/settings.py`.

**Step 2: Update env and Make defaults**

- In `.env.example`, update the default dimension comments to reflect 1024 as the default; keep 1536 only where explicitly noted as OpenAI override.
- In `Makefile`, update the `index-recreate` default to `${EMBEDDING_DIMENSIONS:-1024}`.

**Step 3: Run focused tests**

Run: `uv run pytest tests/unit/scripts/test_create_vector_index.py -v`
Expected: PASS.

**Step 4: Commit**

```bash
git add src/config/settings.py .env.example Makefile
git commit -m "fix: set default embedding dimensions to 1024"
```

---

### Task 4: Update documentation that states the default as 1536

**Files:**
- Modify: `docs/architecture/overview.md`
- Modify: `docs/brownfield-architecture.md`
- Modify: `docs/plans/2026-01-12-pr-01-baseline-alignment.md`
- Modify: `docs/upstream-refactor/DR_01_fancyrag_inventory_ingestion.md`
- Modify: `docs/upstream-refactor/DR_03_fancyrag_inventory_qa_and_ops.md`
- Modify: `docs/upstream-refactor/DR_06 Phased Refactor Plan.md`
- Modify: `docs/upstream-refactor/DR_07_risk_register.md`
- Modify: `docs/upstream-refactor/DR_09_agent_task_list.md`
- Modify: `docs/upstream-refactor/CANONICAL_BASELINE_00-04.md`
- Modify: `docs/upstream-refactor/RUNBOOK_UPDATES_11_TO_12.md`

**Step 1: Update default dimension references to 1024**

- Replace statements describing the default embedding dimension as 1536 with 1024.
- Keep OpenAI-specific mentions of 1536 only if explicitly scoped to OpenAI overrides.

**Step 2: Spot-check docs for consistency**

Run: `rg -n "default.*1536|EMBEDDING_DIMENSIONS.*1536|embedding dimensions.*1536" docs`
Expected: No remaining references that claim 1536 is the default.

**Step 3: Commit**

```bash
git add docs/architecture/overview.md \
  docs/brownfield-architecture.md \
  docs/plans/2026-01-12-pr-01-baseline-alignment.md \
  docs/upstream-refactor/DR_01_fancyrag_inventory_ingestion.md \
  docs/upstream-refactor/DR_03_fancyrag_inventory_qa_and_ops.md \
  docs/upstream-refactor/DR_06\ Phased\ Refactor\ Plan.md \
  docs/upstream-refactor/DR_07_risk_register.md \
  docs/upstream-refactor/DR_09_agent_task_list.md \
  docs/upstream-refactor/CANONICAL_BASELINE_00-04.md \
  docs/upstream-refactor/RUNBOOK_UPDATES_11_TO_12.md
git commit -m "docs: update default embedding dimension to 1024"
```

---

### Task 5: Final verification & PR update

**Step 1: Run targeted tests**

- `uv run pytest tests/unit/scripts/test_create_vector_index.py -v`
- `uv run pytest tests/unit/cli/test_openai_probe.py -v`
- `uv run pytest tests/unit/cli/test_openai_client.py -v`

**Step 2: Optional integration checks (if docker is available)**

- `make smoke`

**Step 3: Update PR**

- Push commits to `migration/pr-01-baseline-alignment`.
- Update PR description with a brief note about the default dimension change and tests run.

---

## Notes
- Worktree usage is disallowed by `AGENTS.md`, so execution should remain on the existing branch.
