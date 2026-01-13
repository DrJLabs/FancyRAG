# PR 65 Code Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address actionable review comments in PR #65 by tightening exception handling, fixing OpenAI error usage, aligning MCP HTTP routes with configured base paths, and updating Neo4j index creation.

**Architecture:** Make targeted fixes in `src/fancyrag/embeddings.py` and `src/fancyrag/mcp/runtime.py` with minimal behavior changes, driven by new/updated tests that fail before the fix. Update the Makefile cypher command to use `CREATE VECTOR INDEX` with `vector.dimensions`/`vector.similarity_function` config keys.

**Tech Stack:** Python, pytest, FastMCP/Starlette, Neo4j driver, Makefile.

### Task 1: Embedding retry error handling

**Files:**
- Modify: `tests/test_embeddings.py`
- Modify: `src/fancyrag/embeddings.py`

**Step 1: Write the failing test**

Add a test that asserts empty OpenAI responses raise `OpenAIError` and do not retry:

```python
from openai import APIConnectionError, OpenAIError
import httpx

class EmptyEmbeddingsAPI:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[])


def test_retrying_embeddings_empty_data_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = EmptyEmbeddingsAPI()
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
        max_retries=3,
    )

    with pytest.raises(OpenAIError, match="empty data list"):
        embedder.embed_query("test query")

    assert len(api.calls) == 1
```

Update the transient failure stub to use `APIConnectionError` so retry tests still match the new retryable exception list.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings.py::test_retrying_embeddings_empty_data_no_retry -v`

Expected: FAIL (currently retries and/or raises `NameError` for missing `OpenAIError` import).

**Step 3: Write minimal implementation**

Update `src/fancyrag/embeddings.py` to import OpenAI exceptions and only retry on connection/timeout/rate-limit errors:

```python
try:  # pragma: no cover - exercised when openai is installed
    from openai import APIConnectionError, APITimeoutError, RateLimitError, OpenAIError
except ImportError:  # pragma: no cover - defensive fallback
    class OpenAIError(Exception):
        pass

    class APIConnectionError(OpenAIError):
        def __init__(self, *_args, **_kwargs):
            super().__init__("Connection error")

    class APITimeoutError(OpenAIError):
        def __init__(self, *_args, **_kwargs):
            super().__init__("Timeout error")

    class RateLimitError(OpenAIError):
        def __init__(self, *_args, **_kwargs):
            super().__init__("Rate limit error")

...
                if not getattr(response, "data", None):
                    raise OpenAIError("OpenAI API returned empty data list")
...
            except (APIConnectionError, APITimeoutError, RateLimitError) as error:
                ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embeddings.py::test_retrying_embeddings_empty_data_no_retry -v`

Expected: PASS.

**Step 5: Run the full embeddings test file**

Run: `uv run pytest tests/test_embeddings.py -v`

Expected: PASS.

### Task 2: HTTP route pathing, auth guard, and JSON error handling

**Files:**
- Modify: `tests/servers/test_runtime.py`
- Modify: `src/fancyrag/mcp/runtime.py`

**Step 1: Write failing tests**

Add tests for:

1) Custom base path routing:
```python

def test_http_routes_respect_server_path(base_config):
    config = base_config.model_copy(deep=True)
    config.server.auth_required = False
    config.oauth = None
    config.server.path = "/api/mcp"

    records = [{"node": FakeNode("1", text="Doc"), "score": 1.0, "text": "Doc"}]
    metadata = {"query_vector": [0.1]}
    driver = StubDriver({
        runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 1.0}], None, None),
        runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 1.0}], None, None),
        runtime.FETCH_NODE_QUERY: ([{"node": FakeNode("1", text="Doc"), "labels": ["Chunk"]}], None, None),
    })
    state = _state_with(driver, FakeRetriever(records, metadata), config)
    server = runtime.build_server(state)
    app = server.http_app(path=config.server.path, stateless_http=True, json_response=True)

    with TestClient(app) as client:
        response = client.post("/api/mcp/search", json={"query": "graph"})
        assert response.status_code == 200
```

2) Auth error guard when `_get_resource_url` fails:
```python

class BrokenResourceProvider(StubAuthProvider):
    def _get_resource_url(self, *_args, **_kwargs):
        raise AttributeError("no resource")


def test_http_auth_error_handles_missing_resource_url(base_config):
    records = [{"node": FakeNode("1", text="Doc"), "score": 1.0, "text": "Doc"}]
    metadata = {"query_vector": [0.1]}
    driver = StubDriver({
        runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 1.0}], None, None),
        runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 1.0}], None, None),
    })
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    provider = BrokenResourceProvider(token="valid-token")
    server = runtime.build_server(state, auth_provider=provider)
    app = server.http_app(path=base_config.server.path, stateless_http=True, json_response=True)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/mcp/search", json={"query": "graph"})
        assert response.status_code == 401
```

3) Request.json non-JSON errors are not swallowed:
```python

def test_http_search_json_runtime_error_surfaces(base_config, monkeypatch):
    config = base_config.model_copy(deep=True)
    config.server.auth_required = False
    config.oauth = None

    state = _state_with(StubDriver({}), FakeRetriever([], {}), config)
    server = runtime.build_server(state)
    app = server.http_app(path=config.server.path, stateless_http=True, json_response=True)

    async def boom(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(Request, "json", boom, raising=False)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/mcp/search", data="{}", headers={"content-type": "application/json"})
        assert response.status_code == 500
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/servers/test_runtime.py::test_http_routes_respect_server_path -v`

Expected: FAIL (404 until base path is honored).

**Step 3: Write minimal implementation**

Update `src/fancyrag/mcp/runtime.py` to:
- Build custom route prefixes from `state.config.server.path`.
- Guard `_get_resource_url` with `try/except AttributeError` (or `Exception`) so auth errors still return 401/403.
- Catch `json.JSONDecodeError` only for request parsing.
- Use exact type checks for `top_k`/`effective_search_ratio`.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/servers/test_runtime.py::test_http_routes_respect_server_path -v`

Expected: PASS.

**Step 5: Run the full runtime test file**

Run: `uv run pytest tests/servers/test_runtime.py -v`

Expected: PASS.

### Task 3: Neo4j exception narrowing

**Files:**
- Modify: `tests/servers/test_runtime.py`
- Modify: `src/fancyrag/mcp/runtime.py`

**Step 1: Write failing tests**

Add tests that ensure non-Neo4j errors are not swallowed:

```python

def test_vector_scores_propagates_non_neo4j_error(base_config):
    from fancyrag.mcp.runtime import _vector_scores

    class FailingDriver:
        def execute_query(self, *_, **__):
            raise ValueError("boom")

    state = _state_with(FailingDriver(), FakeRetriever([], {}), base_config)

    with pytest.raises(ValueError, match="boom"):
        _vector_scores(state, [0.1], top_k=1, ratio=1)


def test_fetch_sync_propagates_non_neo4j_error(base_config):
    class FailingDriver:
        def execute_query(self, *_, **__):
            raise ValueError("boom")

    state = _state_with(FailingDriver(), FakeRetriever([], {}), base_config)

    with pytest.raises(ValueError, match="boom"):
        runtime.fetch_sync(state, "42")
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/servers/test_runtime.py::test_vector_scores_propagates_non_neo4j_error -v`

Expected: FAIL (current code catches Exception and returns empty results).

**Step 3: Write minimal implementation**

Update `src/fancyrag/mcp/runtime.py` to import and catch `Neo4jError` in `_vector_scores`, `_fulltext_scores`, and `fetch_sync`:

```python
from neo4j.exceptions import Neo4jError
...
    except Neo4jError as error:
        ...
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/servers/test_runtime.py::test_vector_scores_propagates_non_neo4j_error -v`

Expected: PASS.

**Step 5: Run the full runtime test file**

Run: `uv run pytest tests/servers/test_runtime.py -v`

Expected: PASS.

### Task 4: Update Makefile vector index creation

**Files:**
- Modify: `Makefile`

**Step 1: Update index-recreate Cypher**

Replace the deprecated procedure with `CREATE VECTOR INDEX`:

```make
index-recreate: up
	docker compose exec neo4j cypher-shell -u $${NEO4J_USERNAME:-neo4j} -p $${NEO4J_PASSWORD:-password} \
		"DROP INDEX text_embeddings IF EXISTS; CREATE VECTOR INDEX text_embeddings IF NOT EXISTS FOR (n:Chunk) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: $${EMBEDDING_DIMENSIONS:-1024}, `vector.similarity_function`: 'cosine'}};"
```

**Step 2: Manual verification**

No automated tests. Verify by running `make index-recreate` in a local Neo4j stack.

---

Plan complete and saved to `docs/plans/2026-01-13-code-review-fixes.md`. Two execution options:

1. Subagent-Driven (this session) – run each task with reviews between steps.
2. Parallel Session (separate) – open new session and execute with `superpowers:executing-plans`.

Which approach?
