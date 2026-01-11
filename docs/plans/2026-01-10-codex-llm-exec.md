# Codex CLI LLM Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route all chat-based LLM calls through Codex CLI (gpt-5.1-mini, medium reasoning) while leaving embeddings on the existing local embedding model/dimensions.

**Architecture:** Add a Codex exec client + LLM adapter that implements neo4j-graphrag's `LLMInterface`, then switch LLM provider selection via config while keeping embeddings on `SharedOpenAIClient`. Persist Codex JSONL logs/last messages under the ingestion QA report directory for debugging.

**Tech Stack:** Python 3.12, neo4j-graphrag `LLMInterface`, Codex CLI (`codex exec`), Ruff, pytest.

### Task 1: Add Codex provider settings and selection

**Files:**
- Modify: `src/config/settings.py`
- Create: `tests/unit/config/test_codex_settings.py`

**Step 1: Write the failing test**

```python
def test_codex_settings_defaults():
    settings = CodexSettings.load(env={})
    assert settings.model == "gpt-5.1-mini"
    assert settings.reasoning_effort == "medium"
    assert settings.max_concurrency == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/config/test_codex_settings.py::test_codex_settings_defaults -v`
Expected: FAIL with "CodexSettings not defined" or import error.

**Step 3: Write minimal implementation**

Add `CodexSettings` + `LLM_PROVIDER` to `src/config/settings.py` with env-backed defaults and validation.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/config/test_codex_settings.py::test_codex_settings_defaults -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config/settings.py tests/unit/config/test_codex_settings.py
git commit -m "feat: add codex llm settings"
```

### Task 2: Codex exec client wrapper

**Files:**
- Create: `src/cli/codex_exec_client.py`
- Create: `tests/unit/cli/test_codex_exec_client.py`

**Step 1: Write the failing test**

```python
def test_codex_exec_builds_command():
    client = CodexExecClient(model="gpt-5.1-mini", reasoning_effort="medium")
    cmd = client._build_command(schema_path=Path("schema.json"), output_path=Path("out.json"))
    assert "codex" in cmd
    assert "--output-schema" in cmd
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/cli/test_codex_exec_client.py::test_codex_exec_builds_command -v`
Expected: FAIL with "CodexExecClient not defined".

**Step 3: Write minimal implementation**

Add a Codex exec client that builds the command, enforces max concurrency via a semaphore, and captures JSONL/last-message outputs.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/cli/test_codex_exec_client.py::test_codex_exec_builds_command -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cli/codex_exec_client.py tests/unit/cli/test_codex_exec_client.py
git commit -m "feat: add codex exec client"
```

### Task 3: Codex LLM adapter + pipeline selection

**Files:**
- Create: `src/fancyrag/kg/codex_llm.py`
- Modify: `src/fancyrag/kg/pipeline.py`
- Modify: `src/fancyrag/kg/phases.py`
- Modify: `tests/unit/fancyrag/kg/test_pipeline.py`
- Modify: `tests/unit/fancyrag/kg/test_phases.py`

**Step 1: Write the failing test**

```python
def test_pipeline_uses_codex_llm_when_enabled():
    settings = FakeSettings(llm_provider="codex")
    bundle = pipeline._build_clients_for_settings(settings, qa_report_dir=Path("artifacts"))
    assert bundle.llm.__class__.__name__ == "CodexExecLLM"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/fancyrag/kg/test_pipeline.py::test_pipeline_uses_codex_llm_when_enabled -v`
Expected: FAIL with "CodexExecLLM not defined" or assertion error.

**Step 3: Write minimal implementation**

Add `CodexExecLLM` implementing `LLMInterface` and wire provider selection in `build_clients`/pipeline using `LLM_PROVIDER`. Ensure embeddings still use `SharedOpenAIClient`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/fancyrag/kg/test_pipeline.py::test_pipeline_uses_codex_llm_when_enabled -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/fancyrag/kg/codex_llm.py src/fancyrag/kg/pipeline.py src/fancyrag/kg/phases.py \
  tests/unit/fancyrag/kg/test_pipeline.py tests/unit/fancyrag/kg/test_phases.py
git commit -m "feat: add codex llm adapter"
```

### Task 4: Diagnostics LLM provider switch

**Files:**
- Modify: `src/cli/diagnostics.py`
- Create: `tests/unit/cli/test_diagnostics_codex.py`

**Step 1: Write the failing test**

```python
def test_diagnostics_uses_codex_llm_when_enabled():
    result = run_probe_with_env({"LLM_PROVIDER": "codex"})
    assert result["llm_provider"] == "codex"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/cli/test_diagnostics_codex.py::test_diagnostics_uses_codex_llm_when_enabled -v`
Expected: FAIL with "llm_provider missing".

**Step 3: Write minimal implementation**

Route diagnostics LLM calls through Codex when enabled and record provider in the probe output.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/cli/test_diagnostics_codex.py::test_diagnostics_uses_codex_llm_when_enabled -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cli/diagnostics.py tests/unit/cli/test_diagnostics_codex.py
git commit -m "feat: route diagnostics llm via codex"
```

### Task 5: Repo-local CODEX_HOME config + docs sync

**Files:**
- Create: `.codex/ingest/config.toml.example`
- Create: `.codex/ingest/.gitignore`
- Modify: `.env.example`
- Modify: `README.md`

**Step 1: Update config/example and docs**

Document `CODEX_HOME`, `LLM_PROVIDER=codex`, model/reasoning settings, logging location, and concurrency defaults.

**Step 2: Commit**

```bash
git add .codex/ingest/config.toml.example .codex/ingest/.gitignore .env.example README.md
git commit -m "docs: add codex llm config example"
```

