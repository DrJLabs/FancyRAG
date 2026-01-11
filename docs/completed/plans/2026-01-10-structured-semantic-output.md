# Structured Semantic Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enforce schema‑valid semantic extraction output via the Responses API, with safe fallback + retry and improved diagnostics for long‑term ingestion stability.

**Architecture:** Introduce a structured‑output LLM adapter for semantic extraction that always sends a Responses API `text.format` JSON schema, uses strict schema validation, and retries once on parse/format failures. Wire this adapter into the KG pipeline via a dedicated semantic LLM in the client bundle. Add optional failure artifacts and configuration flags while keeping strict QA defaults.

**Tech Stack:** Python 3.12, OpenAI Responses API, Pydantic v2 schema generation, neo4j_graphrag `LLMEntityRelationExtractor`, FancyRAG pipeline.

---

## Goal
- Ensure semantic extraction returns valid `Neo4jGraph` JSON via schema enforcement.
- Preserve strict QA gating by default while providing robust retries and optional diagnostics.

## Assumptions / constraints
- No git worktrees (repo rule); work on a normal branch.
- No new third‑party dependencies unless strictly required.
- Keep embeddings/Neo4j schema/chunking untouched.
- Models: default `gpt-5-mini` for semantic extraction; fallback to JSON mode if schema unsupported.

## Research (current state)
- Semantic extraction uses `LLMEntityRelationExtractor` and raises on invalid JSON format.
  - `neo4j_graphrag.experimental.components.entity_relation_extractor.py`
- KG pipeline builds a single `SharedOpenAILLM` for both ingestion and semantic extraction.
  - `src/fancyrag/kg/pipeline.py`
  - `src/fancyrag/kg/phases.py`
- Responses API supports `text.format` with `json_schema` and `strict: true`.
  - Context7 OpenAI docs (Responses Structured Outputs).

## Analysis
### Options
1) **Responses API `text.format` JSON schema + strict** with 1 retry (recommended).
2) JSON mode (`json_object`) + strict validation + retry.
3) Prompt‑only JSON + relaxed QA thresholds.

### Decision
- **Chosen:** Option 1 with Option 2 fallback.
- **Why:** Highest long‑term stability; schema enforcement reduces malformed output while fallback preserves compatibility.

### Risks / edge cases
- Model rejects schema/format → need fallback detection and retry.
- Pydantic schema needs `additionalProperties: false` for strict mode.
- Logging invalid outputs can leak data → make it opt‑in and scrub paths.

### Open questions
- None (user confirmed strict defaults and fallback strategy).

## Q&A (answer before implementation)
- Accepted: strict schema enforcement, fallback to json_object, 1 retry, optional failure artifacts.

---

## Implementation plan

### Task 1: Add semantic output settings and schema helper

**Files:**
- Modify: `src/config/settings.py`
- Create: `src/fancyrag/kg/structured_output.py`
- Test: `tests/unit/config/test_openai_settings.py`

#### Step 1: Write the failing tests
```python
# tests/unit/config/test_openai_settings.py

def test_semantic_output_defaults():
    settings = OpenAISettings.load({}, actor="pytest")
    assert settings.semantic_response_format == "json_schema"
    assert settings.semantic_schema_strict is True
    assert settings.semantic_max_retries == 1
    assert settings.semantic_failure_artifacts is False
```

#### Step 2: Run test to verify it fails
Run: `uv run pytest tests/unit/config/test_openai_settings.py::test_semantic_output_defaults -v`
Expected: FAIL (missing settings).

#### Step 3: Write minimal implementation
Add fields + env parsing:
```python
# src/config/settings.py (OpenAISettings)
semantic_response_format: str = Field(default="json_schema")
semantic_schema_strict: bool = Field(default=True)
semantic_max_retries: int = Field(default=1, ge=0)
semantic_failure_artifacts: bool = Field(default=False)
```
Parse envs:
- `OPENAI_SEMANTIC_RESPONSE_FORMAT`
- `OPENAI_SEMANTIC_SCHEMA_STRICT`
- `OPENAI_SEMANTIC_MAX_RETRIES`
- `OPENAI_SEMANTIC_FAILURE_ARTIFACTS`

Add to `export_environment()`.

Add schema helper:
```python
# src/fancyrag/kg/structured_output.py
from neo4j_graphrag.experimental.components.types import Neo4jGraph

def _strict_schema(schema: dict) -> dict:
    # recursively set additionalProperties=false on object nodes
    ...

def build_neo4j_graph_schema() -> dict:
    schema = Neo4jGraph.model_json_schema()
    return _strict_schema(schema)
```

#### Step 4: Run test to verify it passes
Run: `uv run pytest tests/unit/config/test_openai_settings.py::test_semantic_output_defaults -v`
Expected: PASS.

#### Step 5: Commit
```bash
git add src/config/settings.py src/fancyrag/kg/structured_output.py tests/unit/config/test_openai_settings.py
git commit -m "feat: add semantic output settings and schema helper"
```

---

### Task 2: Implement structured semantic LLM adapter with fallback

**Files:**
- Create: `src/fancyrag/kg/semantic_llm.py`
- Modify: `src/fancyrag/kg/pipeline.py`
- Test: `tests/unit/fancyrag/kg/test_pipeline.py`

#### Step 1: Write the failing test
```python
# tests/unit/fancyrag/kg/test_pipeline.py

def test_semantic_llm_uses_json_schema_format():
    llm = StructuredSemanticLLM(shared_client, settings)
    llm.invoke("prompt")
    assert shared_client.last_params["text"]["format"]["type"] == "json_schema"
```

#### Step 2: Run test to verify it fails
Run: `uv run pytest tests/unit/fancyrag/kg/test_pipeline.py::test_semantic_llm_uses_json_schema_format -v`
Expected: FAIL (class missing).

#### Step 3: Write minimal implementation
Create adapter that always passes Responses API `text.format`:
```python
# src/fancyrag/kg/semantic_llm.py
class StructuredSemanticLLM(LLMInterface):
    def __init__(self, client: SharedOpenAIClient, settings: OpenAISettings, schema: dict):
        self._client = client
        self._settings = settings
        self._schema = schema

    def invoke(self, input: str, message_history=None, system_instruction=None):
        params = {
            "messages": self._build_messages(input, message_history, system_instruction),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "neo4j_graph",
                    "schema": self._schema,
                    "strict": True,
                }
            },
        }
        try:
            return self._client.chat_completion(**params)
        except OpenAIClientError as exc:
            # fallback to json_object
            if _looks_like_format_error(exc):
                params["text"]["format"] = {"type": "json_object"}
                return self._client.chat_completion(**params)
            raise
```
Add `_looks_like_format_error` that checks `str(exc)` for response_format/schema messages.
Store last response content for optional failure artifacts.

#### Step 4: Run test to verify it passes
Run: `uv run pytest tests/unit/fancyrag/kg/test_pipeline.py::test_semantic_llm_uses_json_schema_format -v`
Expected: PASS.

#### Step 5: Commit
```bash
git add src/fancyrag/kg/semantic_llm.py src/fancyrag/kg/pipeline.py tests/unit/fancyrag/kg/test_pipeline.py
git commit -m "feat: add structured semantic LLM adapter"
```

---

### Task 3: Wire semantic LLM into client bundle and pipeline

**Files:**
- Modify: `src/fancyrag/kg/phases.py`
- Modify: `src/fancyrag/kg/pipeline.py`
- Test: `tests/unit/fancyrag/kg/test_pipeline.py`

#### Step 1: Write failing tests
```python
# tests/unit/fancyrag/kg/test_pipeline.py

def test_build_clients_includes_semantic_llm():
    bundle = build_clients(...)
    assert bundle.semantic_llm is not None
```

#### Step 2: Run test to verify it fails
Run: `uv run pytest tests/unit/fancyrag/kg/test_pipeline.py::test_build_clients_includes_semantic_llm -v`
Expected: FAIL.

#### Step 3: Implement
- Extend `ClientBundle` to include `semantic_llm`.
- Update `build_clients` to build semantic LLM using `StructuredSemanticLLM` and the schema helper.
- In `_run_semantic_enrichment`, use `semantic_llm` instead of `llm`.

#### Step 4: Run test to verify it passes
Run: `uv run pytest tests/unit/fancyrag/kg/test_pipeline.py::test_build_clients_includes_semantic_llm -v`
Expected: PASS.

#### Step 5: Commit
```bash
git add src/fancyrag/kg/phases.py src/fancyrag/kg/pipeline.py tests/unit/fancyrag/kg/test_pipeline.py
git commit -m "feat: wire structured semantic LLM into pipeline"
```

---

### Task 4: Add per‑chunk retry and optional failure artifacts

**Files:**
- Modify: `src/fancyrag/kg/pipeline.py`
- Modify: `src/config/settings.py`
- Modify: `src/fancyrag/kg/semantic_llm.py`
- Test: `tests/unit/fancyrag/kg/test_pipeline.py`

#### Step 1: Write failing tests
```python
# tests/unit/fancyrag/kg/test_pipeline.py

def test_semantic_retry_on_format_error(monkeypatch):
    # stub LLM to fail once then succeed
    ...
    stats = _run_semantic_enrichment(...)
    assert stats.chunk_failures == 0
```

#### Step 2: Run test to verify it fails
Run: `uv run pytest tests/unit/fancyrag/kg/test_pipeline.py::test_semantic_retry_on_format_error -v`
Expected: FAIL.

#### Step 3: Implement
- Add retry loop in `_extract_and_process` (max retries from settings).
- If retries exhausted, increment `chunk_failures`.
- If `semantic_failure_artifacts` enabled, write sanitized failure payload to:
  `artifacts/ingestion/<run_id>/semantic_failures/<chunk_uid>.json`.

#### Step 4: Run test to verify it passes
Run: `uv run pytest tests/unit/fancyrag/kg/test_pipeline.py::test_semantic_retry_on_format_error -v`
Expected: PASS.

#### Step 5: Commit
```bash
git add src/fancyrag/kg/pipeline.py src/fancyrag/kg/semantic_llm.py src/config/settings.py tests/unit/fancyrag/kg/test_pipeline.py
git commit -m "feat: add semantic retries and failure artifacts"
```

---

### Task 5: Documentation + env examples

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

#### Step 1: Update docs
Add envs:
- `OPENAI_SEMANTIC_RESPONSE_FORMAT` (`json_schema|json_object|off`)
- `OPENAI_SEMANTIC_SCHEMA_STRICT` (`true|false`)
- `OPENAI_SEMANTIC_MAX_RETRIES` (int)
- `OPENAI_SEMANTIC_FAILURE_ARTIFACTS` (`true|false`)

#### Step 2: Commit
```bash
git add README.md .env.example
git commit -m "docs: document semantic output controls"
```

---

### Task 6: Verification

#### Step 1: Run targeted unit tests
Run:
```bash
uv run pytest \
  tests/unit/config/test_openai_settings.py \
  tests/unit/fancyrag/kg/test_pipeline.py \
  tests/unit/cli/test_openai_client.py -q
```
Expected: PASS.

#### Step 2: Run quick ingest smoke
Run:
```bash
FANCYRAG_DOTENV_PATH=.env.local NEO4J_URI=bolt://localhost:22011 \
  uv run python scripts/kg_build.py --source-dir docs/samples/quick_ingest --profile markdown --enable-semantic
```
Expected: completes; QA report status = pass; no “improper format” warnings.

#### Step 3: Commit verification summary
Update `TESTING_SUMMARY.md` with test outputs.

---

## Tests to run
- `uv run pytest tests/unit/config/test_openai_settings.py -q`
- `uv run pytest tests/unit/fancyrag/kg/test_pipeline.py -q`
- `uv run pytest tests/unit/cli/test_openai_client.py -q`
- Manual: `scripts/kg_build.py` against `docs/samples/quick_ingest` with semantic enabled.
