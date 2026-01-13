import time
from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from fancyrag.config import (
    AppConfig,
    EmbeddingSettings,
    IndexSettings,
    Neo4jSettings,
    OAuthSettings,
    QuerySettings,
    ServerSettings,
)
from fancyrag.mcp import runtime
from fastmcp.server.auth.auth import AccessToken, AuthProvider


class FakeNode:
    def __init__(self, element_id: str, **props):
        self.element_id = element_id
        self._props = props
        self.labels = {"Chunk"}

    def get(self, key, default=None):
        return self._props.get(key, default)

    def items(self):
        return self._props.items()

    def __iter__(self):
        return iter(self._props)

    def __getitem__(self, item):
        return self._props[item]


class StubDriver:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.responses.get(query, ([], None, None))

    def close(self):  # pragma: no cover - driver cleanup in entrypoint
        pass


class FakeRetriever:
    def __init__(self, records, metadata):
        self._records = records
        self._metadata = metadata

    def get_search_results(self, **_kwargs):
        return SimpleNamespace(records=self._records, metadata=self._metadata)


class StubAuthProvider(AuthProvider):
    def __init__(self, token: str):
        super().__init__(base_url="http://localhost:8080", required_scopes=["openid"])
        self._token = token

    async def verify_token(self, token: str):
        if token == self._token:
            return AccessToken(
                token=token,
                client_id="stub-client",
                scopes=["openid"],
                expires_at=int(time.time()) + 3600,
                resource=str(self.base_url) if self.base_url else None,
                claims={},
            )
        return None


@pytest.fixture
def base_config(tmp_path):
    query_path = tmp_path / "hybrid.cypher"
    query_path.write_text("RETURN node, score", encoding="utf-8")
    return AppConfig(
        neo4j=Neo4jSettings(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",  # test uses dummy password  # noqa: S106
            database="neo4j",
        ),
        indexes=IndexSettings(index_name="text_embeddings", fulltext_index_name="chunk_text_fulltext"),
        embedding=EmbeddingSettings(
            base_url="http://localhost:20010/v1",
            api_key="dummy",
            model="text-embedding-3-large",
            timeout_seconds=5,
            max_retries=2,
        ),
        oauth=OAuthSettings(
            client_id="client",
            client_secret="secret",  # test uses dummy secret  # noqa: S106
            required_scopes=["openid"],
        ),
        server=ServerSettings(base_url="http://localhost:8080"),
        query=QuerySettings(path=query_path, template="RETURN node, score"),
    )


def _state_with(driver, retriever, config):
    return runtime.ServerState(config=config, driver=driver, retriever=retriever)


def test_search_sync_returns_scores(base_config):
    records = [
        {"node": FakeNode("1", text="Doc 1", embedding=[0.1]), "score": 0.9, "text": "Doc 1"},
        {"node": FakeNode("2", text="Doc 2", embedding=[0.2]), "score": 0.7},
    ]
    metadata = {"query_vector": [0.11, 0.22]}

    driver = StubDriver(
        {
            runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.5}, {"element_id": "2", "score": 1.0}], None, None),
            runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 2.0}], None, None),
        }
    )
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    response = runtime.search_sync(state, "graph", top_k=2, effective_ratio=1)

    assert response["top_k"] == 2
    assert len(response["results"]) == 2

    first = response["results"][0]
    assert first["text"] == "Doc 1"
    assert first["score_vector"] == pytest.approx(0.5)
    assert first["score_fulltext"] == pytest.approx(1.0)
    assert "embedding" not in first["metadata"]

    second = response["results"][1]
    assert second["score_vector"] == pytest.approx(1.0)
    assert second["score_fulltext"] == pytest.approx(0.0)


def test_search_sync_surfaces_semantic_metadata(base_config):
    records = [
        {
            "node": FakeNode("1", text="Doc 1", embedding=[0.1]),
            "score": 0.9,
            "text": "Doc 1",
            "semantic_nodes": [{"id": "n1"}],
            "semantic_relationships": [{"type": "REL"}],
        }
    ]
    metadata = {"query_vector": [0.11, 0.22]}

    driver = StubDriver(
        {
            runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.5}], None, None),
            runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 2.0}], None, None),
        }
    )
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    response = runtime.search_sync(state, "graph", top_k=1, effective_ratio=1)

    result = response["results"][0]
    assert result["semantic_nodes"] == [{"id": "n1"}]
    assert result["semantic_relationships"] == [{"type": "REL"}]


def test_fetch_sync_found(base_config):
    node = FakeNode("42", text="Doc", embedding=[0.2])
    driver = StubDriver({runtime.FETCH_NODE_QUERY: ([{"node": node, "labels": ["Chunk"]}], None, None)})
    state = _state_with(driver, FakeRetriever([], {}), base_config)

    result = runtime.fetch_sync(state, "42")

    assert result["found"] is True
    assert result["metadata"]["element_id"] == "42"
    assert result["metadata"]["labels"] == ["Chunk"]


def test_fetch_sync_not_found(base_config):
    driver = StubDriver({runtime.FETCH_NODE_QUERY: ([], None, None)})
    state = _state_with(driver, FakeRetriever([], {}), base_config)

    result = runtime.fetch_sync(state, "999")

    assert result == {"found": False, "element_id": "999"}


@pytest.mark.asyncio
async def test_build_server_registers_tools_with_shared_state(base_config):
    records = [
        {"node": FakeNode("1", text="Doc 1", embedding=[0.1]), "score": 0.9, "text": "Doc 1"},
        {"node": FakeNode("2", text="Doc 2", embedding=[0.2]), "score": 0.7},
    ]
    metadata = {"query_vector": [0.11, 0.22]}

    driver = StubDriver(
        {
            runtime.VECTOR_SCORE_QUERY: (
                [
                    {"element_id": "1", "score": 0.5},
                    {"element_id": "2", "score": 1.0},
                ],
                None,
                None,
            ),
            runtime.FULLTEXT_SCORE_QUERY: (
                [
                    {"element_id": "1", "score": 2.0},
                ],
                None,
                None,
            ),
            runtime.FETCH_NODE_QUERY: (
                [
                    {"node": FakeNode("1", text="Doc 1", embedding=[0.3]), "labels": ["Chunk"]},
                ],
                None,
                None,
            ),
        }
    )
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    server = runtime.build_server(state)

    search_tool = await server.get_tool("search")
    fetch_tool = await server.get_tool("fetch")

    search_result = await search_tool.run({
        "query": "graph",
        "top_k": 2,
        "effective_search_ratio": 1,
    })

    assert search_result.structured_content["results"][0]["metadata"]["element_id"] == "1"
    assert search_result.structured_content["results"][0]["score_vector"] == pytest.approx(0.5)
    assert search_result.structured_content["results"][0]["score_fulltext"] == pytest.approx(1.0)

    fetch_result = await fetch_tool.run({"element_id": "1"})

    assert fetch_result.structured_content["found"] is True
    assert fetch_result.structured_content["metadata"]["labels"] == ["Chunk"]

    queried_endpoints = [call[0] for call in driver.calls]
    assert runtime.VECTOR_SCORE_QUERY in queried_endpoints
    assert runtime.FULLTEXT_SCORE_QUERY in queried_endpoints
    assert runtime.FETCH_NODE_QUERY in queried_endpoints


def _ping_payload(identifier: str = "ping-1") -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": identifier,
        "method": "ping",
    }


def test_stateless_http_enforces_authentication(base_config):
    records = [
        {"node": FakeNode("1", text="Doc 1", embedding=[0.1]), "score": 0.9, "text": "Doc 1"},
    ]
    metadata = {"query_vector": [0.11, 0.22]}

    driver = StubDriver(
        {
            runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.9}], None, None),
            runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 1.0}], None, None),
        }
    )
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    provider = StubAuthProvider(token="valid-token")  # test uses dummy token  # noqa: S106
    server = runtime.build_server(state, auth_provider=provider)
    app = server.http_app(path="/mcp", stateless_http=True, json_response=True)

    headers = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
    }

    with TestClient(app) as client:
        unauthorized = client.post("/mcp", headers=headers, json=_ping_payload("unauth"))
        assert unauthorized.status_code == 401

        auth_headers = {
            **headers,
            "authorization": "Bearer valid-token",
        }

        authorized = client.post("/mcp", headers=auth_headers, json=_ping_payload("auth"))
        assert authorized.status_code == 200
        body = authorized.json()
        assert body["result"] == {}


def test_stateless_http_allows_requests_when_auth_disabled(base_config):
    config = base_config.model_copy(deep=True)
    config.server.auth_required = False
    config.oauth = None

    state = _state_with(StubDriver({}), FakeRetriever([], {}), config)
    server = runtime.build_server(state)
    app = server.http_app(path="/mcp", stateless_http=True, json_response=True)

    headers = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
    }

    with TestClient(app) as client:
        response = client.post("/mcp", headers=headers, json=_ping_payload("unauth"))
        assert response.status_code == 200


def test_http_search_and_fetch_routes_return_contract(base_config):
    config = base_config.model_copy(deep=True)
    config.server.auth_required = False
    config.oauth = None

    records = [
        {"node": FakeNode("1", text="Doc 1", embedding=[0.1]), "score": 0.9, "text": "Doc 1"},
    ]
    metadata = {"query_vector": [0.11, 0.22]}

    driver = StubDriver(
        {
            runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.9}], None, None),
            runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 1.0}], None, None),
            runtime.FETCH_NODE_QUERY: (
                [{"node": FakeNode("1", text="Doc 1", embedding=[0.1]), "labels": ["Chunk"]}],
                None,
                None,
            ),
        }
    )
    state = _state_with(driver, FakeRetriever(records, metadata), config)
    server = runtime.build_server(state)
    app = server.http_app(path="/mcp", stateless_http=True, json_response=True)

    headers = {"content-type": "application/json"}

    with TestClient(app) as client:
        response = client.post(
            "/mcp/search",
            headers=headers,
            json={"query": "graph", "top_k": 1, "effective_search_ratio": 1},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["query"] == "graph"
        assert payload["top_k"] == 1
        assert payload["effective_search_ratio"] == 1
        assert payload["results"]
        result = payload["results"][0]
        assert "embedding" not in result["metadata"]

        element_id = result["metadata"]["element_id"]
        fetch = client.post("/mcp/fetch", headers=headers, json={"element_id": element_id})
        assert fetch.status_code == 200
        fetched = fetch.json()
        assert fetched["found"] is True
        assert fetched["element_id"] == element_id
        assert "embedding" not in fetched.get("metadata", {})


def test_http_routes_respect_server_path(base_config):
    config = base_config.model_copy(deep=True)
    config.server.auth_required = False
    config.oauth = None
    config.server.path = "/api/mcp"

    records = [
        {"node": FakeNode("1", text="Doc 1", embedding=[0.1]), "score": 0.9, "text": "Doc 1"},
    ]
    metadata = {"query_vector": [0.11, 0.22]}

    driver = StubDriver(
        {
            runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.9}], None, None),
            runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 1.0}], None, None),
            runtime.FETCH_NODE_QUERY: (
                [{"node": FakeNode("1", text="Doc 1", embedding=[0.1]), "labels": ["Chunk"]}],
                None,
                None,
            ),
        }
    )
    state = _state_with(driver, FakeRetriever(records, metadata), config)
    server = runtime.build_server(state)
    app = server.http_app(path=config.server.path, stateless_http=True, json_response=True)

    with TestClient(app) as client:
        response = client.post("/api/mcp/search", json={"query": "graph"})
        assert response.status_code == 200


def test_http_auth_error_handles_missing_resource_url(base_config):
    records = [
        {"node": FakeNode("1", text="Doc 1", embedding=[0.1]), "score": 0.9, "text": "Doc 1"},
    ]
    metadata = {"query_vector": [0.11, 0.22]}

    driver = StubDriver(
        {
            runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.9}], None, None),
            runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 1.0}], None, None),
        }
    )
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    class BrokenResourceProvider(StubAuthProvider):
        def _get_resource_url(self, *_args, **_kwargs):
            raise AttributeError("missing resource url")

    provider = BrokenResourceProvider(token="valid-token")  # test uses dummy token  # noqa: S106
    server = runtime.build_server(state, auth_provider=provider)
    app = server.http_app(path=base_config.server.path, stateless_http=True, json_response=True)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/mcp/search", json={"query": "graph"})
        assert response.status_code == 401


def test_http_search_json_runtime_error_surfaces(base_config, monkeypatch):
    config = base_config.model_copy(deep=True)
    config.server.auth_required = False
    config.oauth = None

    state = _state_with(StubDriver({}), FakeRetriever([], {}), config)
    server = runtime.build_server(state)
    app = server.http_app(path=config.server.path, stateless_http=True, json_response=True)

    async def boom(_self):
        raise RuntimeError("boom")

    monkeypatch.setattr(Request, "json", boom, raising=False)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/mcp/search",
            data="{}",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 500


def test_search_latency_within_budget(base_config):
    records = [
        {"node": FakeNode("1", text="Doc", embedding=[0.1]), "score": 0.9, "text": "Doc"},
    ]
    metadata = {"query_vector": [0.11, 0.22]}

    driver = StubDriver(
        {
            runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.9}], None, None),
            runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 0.9}], None, None),
        }
    )
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    start = time.perf_counter()

    runtime.search_sync(state, "graph", top_k=1, effective_ratio=1)
    duration = time.perf_counter() - start

    assert duration < 1.5, "Hybrid search exceeded the 1.5s latency budget"


def test_normalize_scores_empty_records() -> None:
    from fancyrag.mcp.runtime import _normalize_scores

    result = _normalize_scores([])

    assert result == {}


def test_normalize_scores_single_record() -> None:
    from fancyrag.mcp.runtime import _normalize_scores

    records = [{"element_id": "1", "score": 0.5}]
    result = _normalize_scores(records)

    assert result == {"1": 1.0}


def test_normalize_scores_multiple_records() -> None:
    from fancyrag.mcp.runtime import _normalize_scores

    records = [
        {"element_id": "1", "score": 0.5},
        {"element_id": "2", "score": 1.0},
        {"element_id": "3", "score": 0.25},
    ]
    result = _normalize_scores(records)

    assert result["1"] == pytest.approx(0.5)
    assert result["2"] == pytest.approx(1.0)
    assert result["3"] == pytest.approx(0.25)


def test_normalize_scores_zero_max_score() -> None:
    from fancyrag.mcp.runtime import _normalize_scores

    records = [
        {"element_id": "1", "score": 0.0},
        {"element_id": "2", "score": 0.0},
    ]
    result = _normalize_scores(records)

    assert result["1"] == 0.0
    assert result["2"] == 0.0


def test_node_metadata_extraction(base_config) -> None:
    from fancyrag.mcp.runtime import _node_metadata

    _ = base_config
    node = FakeNode("42", text="Test", embedding=[0.1, 0.2])
    metadata = _node_metadata(node)

    assert metadata["element_id"] == "42"
    assert metadata["text"] == "Test"
    assert "embedding" not in metadata
    assert metadata["labels"] == ["Chunk"]


def test_node_metadata_none_node() -> None:
    from fancyrag.mcp.runtime import _node_metadata

    metadata = _node_metadata(None)

    assert metadata == {}


def test_vector_scores_empty_vector(base_config) -> None:
    from fancyrag.mcp.runtime import _vector_scores

    driver = StubDriver({})
    state = _state_with(driver, FakeRetriever([], {}), base_config)

    result = _vector_scores(state, [], top_k=5, ratio=1)

    assert result == {}
    assert len(driver.calls) == 0


def test_vector_scores_neo4j_error(base_config) -> None:
    from fancyrag.mcp.runtime import _vector_scores
    from neo4j.exceptions import ServiceUnavailable

    class FailingDriver:
        def execute_query(self, *_, **__):
            raise ServiceUnavailable("Database unavailable")  # noqa

    state = _state_with(FailingDriver(), FakeRetriever([], {}), base_config)

    result = _vector_scores(state, [0.1, 0.2], top_k=5, ratio=1)

    assert result == {}


def test_vector_scores_propagates_non_neo4j_error(base_config) -> None:
    from fancyrag.mcp.runtime import _vector_scores

    class FailingDriver:
        def execute_query(self, *_, **__):
            raise ValueError("boom")

    state = _state_with(FailingDriver(), FakeRetriever([], {}), base_config)

    with pytest.raises(ValueError, match="boom"):
        _vector_scores(state, [0.1, 0.2], top_k=5, ratio=1)


def test_fulltext_scores_empty_query(base_config) -> None:
    from fancyrag.mcp.runtime import _fulltext_scores

    driver = StubDriver({})
    state = _state_with(driver, FakeRetriever([], {}), base_config)

    result = _fulltext_scores(state, "", top_k=5)

    assert result == {}
    assert len(driver.calls) == 0


def test_fulltext_scores_neo4j_error(base_config) -> None:
    from fancyrag.mcp.runtime import _fulltext_scores
    from neo4j.exceptions import ClientError

    class FailingDriver:
        def execute_query(self, *_, **__):
            raise ClientError("Query failed")  # noqa

    state = _state_with(FailingDriver(), FakeRetriever([], {}), base_config)

    result = _fulltext_scores(state, "test query", top_k=5)

    assert result == {}


def test_fulltext_scores_propagates_non_neo4j_error(base_config) -> None:
    from fancyrag.mcp.runtime import _fulltext_scores

    class FailingDriver:
        def execute_query(self, *_, **__):
            raise ValueError("boom")

    state = _state_with(FailingDriver(), FakeRetriever([], {}), base_config)

    with pytest.raises(ValueError, match="boom"):
        _fulltext_scores(state, "test query", top_k=5)


def test_search_sync_no_query_vector(base_config) -> None:
    records = [{"node": FakeNode("1", text="Doc"), "score": 0.9}]
    metadata = {}
    driver = StubDriver({})
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    response = runtime.search_sync(state, "test", top_k=1, effective_ratio=1)

    assert len(response["results"]) == 1
    assert response["results"][0]["score_vector"] == 0.0


def test_search_sync_no_results(base_config) -> None:
    driver = StubDriver({})
    state = _state_with(driver, FakeRetriever([], {}), base_config)

    response = runtime.search_sync(state, "no results", top_k=5, effective_ratio=1)

    assert response["results"] == []
    assert response["top_k"] == 5


def test_search_sync_text_from_node_when_not_in_record(base_config) -> None:
    node = FakeNode("1", text="Node text")
    records = [{"node": node, "score": 0.9}]
    metadata = {"query_vector": [0.1]}
    driver = StubDriver({
        runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.5}], None, None),
        runtime.FULLTEXT_SCORE_QUERY: ([], None, None),
    })
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    response = runtime.search_sync(state, "test", top_k=1, effective_ratio=1)
    assert response["results"][0]["text"] == "Node text"


def test_search_sync_effective_ratio_multiplier(base_config) -> None:
    records = [{"node": FakeNode("1", text="Doc"), "score": 0.9, "text": "Doc"}]
    metadata = {"query_vector": [0.1, 0.2]}
    driver = StubDriver({
        runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.9}], None, None),
        runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 1.0}], None, None),
    })
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    runtime.search_sync(state, "test", top_k=10, effective_ratio=3)

    vector_call = next(call for call in driver.calls if call[0] == runtime.VECTOR_SCORE_QUERY)
    assert vector_call[1]["limit"] == 30


def test_fetch_sync_gracefully_handles_neo4j_error(base_config) -> None:
    from neo4j.exceptions import TransientError

    class FailingDriver:
        def execute_query(self, *_, **__):
            raise TransientError("Database error")  # noqa

    state = _state_with(FailingDriver(), FakeRetriever([], {}), base_config)

    result = runtime.fetch_sync(state, "42")

    assert result == {
        "found": False,
        "element_id": "42",
        "error": "Database query failed: TransientError",
    }


def test_fetch_sync_propagates_non_neo4j_error(base_config) -> None:
    class FailingDriver:
        def execute_query(self, *_, **__):
            raise ValueError("boom")

    state = _state_with(FailingDriver(), FakeRetriever([], {}), base_config)

    with pytest.raises(ValueError, match="boom"):
        runtime.fetch_sync(state, "42")


@pytest.mark.asyncio
async def test_search_tool_validates_top_k_positive(base_config) -> None:
    driver = StubDriver({})
    state = _state_with(driver, FakeRetriever([], {}), base_config)
    server = runtime.build_server(state)

    search_tool = await server.get_tool("search")

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        await search_tool.run({"query": "test", "top_k": 0, "effective_search_ratio": 1})


@pytest.mark.asyncio
async def test_search_tool_validates_top_k_negative(base_config) -> None:
    driver = StubDriver({})
    state = _state_with(driver, FakeRetriever([], {}), base_config)
    server = runtime.build_server(state)

    search_tool = await server.get_tool("search")

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        await search_tool.run({"query": "test", "top_k": -1, "effective_search_ratio": 1})


@pytest.mark.asyncio
async def test_search_tool_validates_effective_ratio_positive(base_config) -> None:
    driver = StubDriver({})
    state = _state_with(driver, FakeRetriever([], {}), base_config)
    server = runtime.build_server(state)

    search_tool = await server.get_tool("search")

    with pytest.raises(ValueError, match="effective_search_ratio must be greater than zero"):
        await search_tool.run({"query": "test", "top_k": 5, "effective_search_ratio": 0})


@pytest.mark.asyncio
async def test_fetch_tool_validates_element_id_not_empty(base_config) -> None:
    driver = StubDriver({})
    state = _state_with(driver, FakeRetriever([], {}), base_config)
    server = runtime.build_server(state)

    fetch_tool = await server.get_tool("fetch")

    with pytest.raises(ValueError, match="element_id is required"):
        await fetch_tool.run({"element_id": ""})


@pytest.mark.asyncio
async def test_search_tool_default_parameters(base_config) -> None:
    records = [{"node": FakeNode("1", text="Doc"), "score": 0.9, "text": "Doc"}]
    metadata = {"query_vector": [0.1]}
    driver = StubDriver({
        runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.9}], None, None),
        runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 1.0}], None, None),
    })
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)
    server = runtime.build_server(state)

    search_tool = await server.get_tool("search")
    result = await search_tool.run({"query": "test"})

    assert result.structured_content["top_k"] == 5
    assert result.structured_content["effective_search_ratio"] == 1


def test_create_state_initializes_driver_and_retriever(base_config, monkeypatch) -> None:
    class MockDriver:
        def __init__(self, uri, auth):
            self.uri = uri
            self.auth = auth

        def verify_connectivity(self):
            pass

    class MockRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("fancyrag.mcp.runtime.GraphDatabase.driver", MockDriver)
    monkeypatch.setattr("fancyrag.mcp.runtime.HybridCypherRetriever", MockRetriever)
    monkeypatch.setattr("fancyrag.mcp.runtime.RetryingOpenAIEmbeddings", lambda *_, **__: "embedder")

    state = runtime.create_state(base_config)

    assert state.config == base_config
    assert isinstance(state.driver, MockDriver)
    assert isinstance(state.retriever, MockRetriever)


def test_build_server_with_custom_auth_provider(base_config) -> None:
    driver = StubDriver({})
    state = _state_with(driver, FakeRetriever([], {}), base_config)

    custom_provider = StubAuthProvider(token="custom-token")  # test uses dummy token  # noqa: S106
    server = runtime.build_server(state, auth_provider=custom_provider)

    assert server is not None


def test_node_metadata_handles_node_without_items_method(base_config) -> None:
    from fancyrag.mcp.runtime import _node_metadata

    _ = base_config

    class MinimalNode:
        def __init__(self):
            self.element_id = "123"
            self.labels = {"Label"}

    node = MinimalNode()
    metadata = _node_metadata(node)

    assert metadata["element_id"] == "123"
    assert metadata["labels"] == ["Label"]


def test_search_sync_multiple_nodes_with_mixed_scores(base_config) -> None:
    records = [
        {"node": FakeNode("1", text="First"), "score": 1.0, "text": "First"},
        {"node": FakeNode("2", text="Second"), "score": 0.8, "text": "Second"},
        {"node": FakeNode("3", text="Third"), "score": 0.6, "text": "Third"},
    ]
    metadata = {"query_vector": [0.1, 0.2, 0.3]}

    driver = StubDriver({
        runtime.VECTOR_SCORE_QUERY: ([
            {"element_id": "1", "score": 0.9},
            {"element_id": "2", "score": 0.7},
            {"element_id": "3", "score": 0.5},
        ], None, None),
        runtime.FULLTEXT_SCORE_QUERY: ([
            {"element_id": "1", "score": 2.0},
            {"element_id": "3", "score": 1.0},
        ], None, None),
    })
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    response = runtime.search_sync(state, "test", top_k=3, effective_ratio=1)

    assert len(response["results"]) == 3

    first_result = response["results"][0]
    assert first_result["text"] == "First"
    assert first_result["score"] == 1.0
    assert first_result["score_vector"] == pytest.approx(1.0)
    assert first_result["score_fulltext"] == pytest.approx(1.0)

    second_result = response["results"][1]
    assert second_result["score_vector"] == pytest.approx(7.0 / 9.0)
    assert second_result["score_fulltext"] == pytest.approx(0.0)

    third_result = response["results"][2]
    assert third_result["score_fulltext"] == pytest.approx(0.5)


def test_search_sync_missing_element_id_in_metadata(base_config) -> None:
    node = FakeNode("1", text="Doc")
    node.element_id = None
    records = [{"node": node, "score": 0.9, "text": "Doc"}]
    metadata = {"query_vector": [0.1]}

    driver = StubDriver({
        runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.9}], None, None),
        runtime.FULLTEXT_SCORE_QUERY: ([], None, None),
    })
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    response = runtime.search_sync(state, "test", top_k=1, effective_ratio=1)

    assert len(response["results"]) == 1
    assert response["results"][0]["score_vector"] == 0.0


def test_fetch_sync_preserves_labels_from_query(base_config) -> None:
    node = FakeNode("42", text="Doc")
    node.labels = {"OriginalLabel"}

    driver = StubDriver({
        runtime.FETCH_NODE_QUERY: ([{"node": node, "labels": ["QueryLabel1", "QueryLabel2"]}], None, None)
    })
    state = _state_with(driver, FakeRetriever([], {}), base_config)

    result = runtime.fetch_sync(state, "42")

    assert result["metadata"]["labels"] == ["QueryLabel1", "QueryLabel2"]
