# MCP Optional Auth Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow local MCP runs to disable OAuth via `MCP_AUTH_REQUIRED=false` while keeping OAuth required by default.

**Architecture:** Add an explicit auth-required flag to server settings, treat OAuth config as optional when auth is disabled, and only attach auth providers when auth is required.

**Tech Stack:** Python (Pydantic), FastMCP, pytest, Docker Compose.

### Task 1: Add failing tests for auth disable behavior

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/servers/test_runtime.py`

**Step 1: Write the failing tests**

```python
# tests/test_config.py

def test_auth_disabled_skips_oauth_requirements(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    monkeypatch.setenv("INDEX_NAME", "text_embeddings")
    monkeypatch.setenv("FULLTEXT_INDEX_NAME", "chunk_text_fulltext")
    monkeypatch.setenv("EMBEDDING_API_BASE_URL", "http://localhost:20010/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "dummy")
    monkeypatch.setenv("MCP_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("HYBRID_RETRIEVAL_QUERY_PATH", str(query_file))
    monkeypatch.setenv("MCP_AUTH_REQUIRED", "false")

    config = load_config()

    assert config.server.auth_required is False
    assert config.oauth is None


def test_auth_required_invalid_value_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("MCP_AUTH_REQUIRED", "maybe")

    with pytest.raises(ConfigurationError, match="MCP_AUTH_REQUIRED"):
        load_config()
```

```python
# tests/servers/test_runtime.py

def test_stateless_http_allows_requests_when_auth_disabled(base_config):
    config = base_config.model_copy(deep=True)
    config.server.auth_required = False
    config.oauth = None

    state = _state_with(StubDriver({}), FakeRetriever([], {}), config)
    server = runtime.build_server(state)
    app = server.http_app(path="/mcp", stateless_http=True, json_response=True)

    with TestClient(app) as client:
        response = client.post("/mcp", headers={"content-type": "application/json"}, json=_ping_payload())
        assert response.status_code == 200
```

**Step 2: Run tests to verify they fail**

Run:
```
PYTHONPATH=src pytest tests/test_config.py::test_auth_disabled_skips_oauth_requirements tests/test_config.py::test_auth_required_invalid_value_raises tests/servers/test_runtime.py::test_stateless_http_allows_requests_when_auth_disabled -q
```
Expected: failures indicating missing fields (`auth_required`/`oauth` handling).

### Task 2: Implement auth-required flag in config

**Files:**
- Modify: `src/fancryrag/config.py`

**Step 1: Write minimal implementation**

- Add a boolean parser for `MCP_AUTH_REQUIRED` (default `true`).
- Extend `ServerSettings` with `auth_required: bool`.
- Make `oauth` optional in `AppConfig` and only require OAuth envs when `auth_required` is true.

```python
class ServerSettings(BaseModel):
    host: str = Field(default="0.0.0.0")  # noqa: S104
    port: int = Field(default=8080, ge=1, le=65535)
    path: str = Field(default="/mcp")
    base_url: str
    auth_required: bool = Field(default=True)

class AppConfig(BaseModel):
    ...
    oauth: OAuthSettings | None
    ...


def _parse_bool(env: Mapping[str, str], key: str, *, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{key} must be a boolean")
```

**Step 2: Run tests to verify they pass**

Run:
```
PYTHONPATH=src pytest tests/test_config.py::test_auth_disabled_skips_oauth_requirements tests/test_config.py::test_auth_required_invalid_value_raises -q
```
Expected: PASS.

### Task 3: Honor auth-required flag in server runtime

**Files:**
- Modify: `src/fancryrag/mcp/runtime.py`
- Modify: `servers/mcp_hybrid_google.py`

**Step 1: Implement conditional auth**

- If `config.server.auth_required` is false, build `FastMCP` without an auth provider.
- If auth is required, keep current behavior (Google OAuth or static token when provided).

**Step 2: Run tests to verify they pass**

Run:
```
PYTHONPATH=src pytest tests/servers/test_runtime.py::test_stateless_http_enforces_authentication tests/servers/test_runtime.py::test_stateless_http_allows_requests_when_auth_disabled -q
```
Expected: PASS.

### Task 4: Update docs and env templates

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

**Step 1: Document `MCP_AUTH_REQUIRED`**

- Add `MCP_AUTH_REQUIRED=true` to `.env.example` with a note on setting `false` for local-only runs.
- Update README to mention the flag and that OAuth vars are optional when auth is disabled.

### Task 5: Local verification

**Files:**
- Modify (local only): `.env.local`

**Step 1: Set flag and restart MCP**

```
MCP_AUTH_REQUIRED=false
```

Run:
```
docker compose -f docker-compose.yml up -d --force-recreate --no-deps mcp
```

Expected: MCP container becomes healthy; no OAuth missing env errors in logs.

