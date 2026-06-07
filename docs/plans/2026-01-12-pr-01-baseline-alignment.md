# PR-01 Baseline Alignment & Stabilization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align FancyRAG baseline with upstream canonical SHA, unify package naming, and stabilize index/config defaults without breaking the MCP contract.

**Architecture:** Rename the Python package to `fancyrag` with a temporary `fancryrag` shim, update imports and config entrypoints, and standardize embedding dimensions across env, scripts, and Make targets. Ensure the MCP server continues to satisfy `/mcp/health`, `/mcp/search`, and `/mcp/fetch` contracts after these baseline changes.

**Tech Stack:** Python, uv, Docker Compose, Neo4j, pytest, FastAPI/FastMCP.

### Task 1: Pin upstream GraphRAG dependency to canonical SHA

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Step 1: Write the failing test**

```python
def test_graphrag_import():
    import neo4j_graphrag
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_graphrag_import.py -v`
Expected: FAIL if dependency is not yet pinned/installed.

**Step 3: Write minimal implementation**

Pin `neo4j-graphrag` to `git+https://...@9e6b58827caa642539c8cb15dcde36c3e4d6d2c6` and update lockfile.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_graphrag_import.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/unit/test_graphrag_import.py
git commit -m "chore: pin graphrag upstream baseline"
```

### Task 2: Rename package to `fancyrag` and add compat shim

**Files:**
- Modify/Move: `src/fancryrag/**` → `src/fancyrag/**`
- Create: `src/fancryrag/__init__.py`
- Modify: `servers/mcp_hybrid_google.py`
- Modify: `scripts/**` (imports)
- Modify: `tests/**`

**Step 1: Write the failing test**

```python
def test_namespace_shim():
    import fancyrag
    import fancryrag
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_namespace_shim.py -v`
Expected: FAIL before rename + shim.

**Step 3: Write minimal implementation**

- Rename module folder to `src/fancyrag`.
- Add shim in `src/fancryrag/__init__.py` that re-exports `fancyrag`.
- Update imports/references in code/tests.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_namespace_shim.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/fancyrag src/fancryrag servers/mcp_hybrid_google.py scripts tests
git commit -m "refactor: rename package to fancyrag with shim"
```

### Task 3: Align embedding dimension defaults across env, scripts, and Makefile

**Files:**
- Modify: `.env.example`
- Modify: `scripts/create_vector_index.py`
- Modify: `docker-compose.smoke.yml`
- Modify: `Makefile`

**Step 1: Write the failing test**

```python
def test_embedding_dimension_default():
    from fancyrag.config.app_config import EmbeddingSettings
    assert EmbeddingSettings().dimensions == 1024
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_embedding_defaults.py -v`
Expected: FAIL before defaults are unified.

**Step 3: Write minimal implementation**

- Set `EMBEDDING_DIMENSIONS=1024` in `.env.example`.
- Ensure `scripts/create_vector_index.py` uses the same default.
- Update `Makefile` to avoid conflicting defaults.
- Keep smoke stack deterministic (explicit env in compose).

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_embedding_defaults.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add .env.example scripts/create_vector_index.py docker-compose.smoke.yml Makefile
git commit -m "chore: align embedding dimension defaults"
```

### Task 4: Make `make index` idempotent and add explicit recreate target

**Files:**
- Modify: `Makefile`

**Step 1: Write the failing test**

```python
def test_make_index_safe_target():
    import subprocess
    assert "index-recreate" in subprocess.check_output(["make", "-n", "index"]).decode()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_make_index_target.py -v`
Expected: FAIL before Makefile updated.

**Step 3: Write minimal implementation**

- `make index` should call `uv run python scripts/create_vector_index.py`.
- Add `index-recreate` target to drop/recreate explicitly.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_make_index_target.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add Makefile tests/unit/test_make_index_target.py
git commit -m "chore: make index idempotent with explicit recreate target"
```

### Task 5: MCP contract alignment + tests (if MCP touched)

**Files:**
- Modify: `src/fancyrag/mcp/runtime.py`
- Modify: `tests/servers/test_runtime.py`

**Step 1: Write the failing test**

```python
def test_mcp_contract_endpoints():
    # Use TestClient to hit /mcp/search and /mcp/fetch
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/servers/test_runtime.py -v`
Expected: FAIL if endpoints not exposed.

**Step 3: Write minimal implementation**

Expose `/mcp/search` and `/mcp/fetch` REST routes that delegate to tool logic and return contract fields.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/servers/test_runtime.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/fancyrag/mcp/runtime.py tests/servers/test_runtime.py
git commit -m "fix: expose mcp search/fetch rest endpoints"
```

### Task 6: Verification gates (DR_08 minimum + contract checks)

**Files:**
- Modify: none (verification only)

**Step 1: Run DR_08 A.1 + A.2**

Run: `git status --porcelain && git rev-parse HEAD && docker --version && docker compose version && uv --version && curl --version && python --version`
Expected: commands succeed.

**Step 2: Run DR_08 B.2 + B.3**

Run: `make up && uv run python scripts/create_vector_index.py && uv run python scripts/create_fulltext_index.py`
Expected: services start; index scripts exit 0.

**Step 3: Run DR_08 B.8.1–B.8.3**

Run contract checks (health/search/fetch) against `http://localhost:22012`.
Expected: JSON schema matches DR_08 requirements.

**Step 4: Run DR_08 B.6.1**

Run: `make smoke`
Expected: smoke tests pass.

**Step 5: Run DR_08 B.11**

Run: `make down`
Expected: services stop cleanly.

**Step 6: Commit (optional)**

No code changes; capture verification evidence in PR description.
