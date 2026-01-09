from __future__ import annotations

from types import SimpleNamespace

import pytest

from fancryrag.embeddings import RetryingOpenAIEmbeddings
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings


class TransientFailure(RuntimeError):
    """Exception for transient failure in stub create calls."""
    def __init__(self):
        super().__init__("transient failure")


class StubEmbeddingsAPI:
    def __init__(self, *, succeed_on: int = 1):
        self.calls: list[dict[str, object]] = []
        self._succeed_on = succeed_on

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) < self._succeed_on:
            raise TransientFailure()
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


def _patch_openai_client(monkeypatch: pytest.MonkeyPatch, client: object, captured_kwargs: dict[str, object]) -> None:
    def fake_initialize(_self, **kwargs):
        captured_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(OpenAIEmbeddings, "_initialize_client", fake_initialize, raising=False)


def test_retrying_embeddings_respects_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = StubEmbeddingsAPI(succeed_on=2)
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    sleep_calls: list[float] = []
    monkeypatch.setattr("fancryrag.embeddings.time.sleep", lambda value: sleep_calls.append(value))

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
        timeout_seconds=7.5,
        max_retries=2,
    )

    result = embedder.embed_query("hello world")

    assert result == [0.1, 0.2, 0.3]
    assert captured_kwargs == {"base_url": "http://embeddings.local", "api_key": "secret"}
    assert len(api.calls) == 2
    assert api.calls[0]["timeout"] == 7.5
    assert api.calls[0]["model"] == "test-model"
    assert sleep_calls == [0.5]


def test_retrying_embeddings_raises_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = StubEmbeddingsAPI(succeed_on=999)
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
        timeout_seconds=1.5,
        max_retries=2,
    )

    with pytest.raises(RuntimeError, match="transient failure"):
        embedder.embed_query("fail")

    assert len(api.calls) == 2
    assert captured_kwargs == {"base_url": "http://embeddings.local", "api_key": "secret"}


def test_retrying_embeddings_first_attempt_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = StubEmbeddingsAPI(succeed_on=1)
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    sleep_calls: list[float] = []
    monkeypatch.setattr("fancryrag.embeddings.time.sleep", lambda value: sleep_calls.append(value))

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
        timeout_seconds=5.0,
        max_retries=3,
    )

    result = embedder.embed_query("test query")

    assert result == [0.1, 0.2, 0.3]
    assert len(api.calls) == 1
    assert len(sleep_calls) == 0


def test_retrying_embeddings_third_attempt_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = StubEmbeddingsAPI(succeed_on=3)
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    sleep_calls: list[float] = []
    monkeypatch.setattr("fancryrag.embeddings.time.sleep", lambda value: sleep_calls.append(value))

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
        timeout_seconds=5.0,
        max_retries=3,
        backoff_seconds=1.0,
    )

    result = embedder.embed_query("test query")

    assert result == [0.1, 0.2, 0.3]
    assert len(api.calls) == 3
    assert len(sleep_calls) == 2
    assert sleep_calls[0] == 1.0
    assert sleep_calls[1] == 2.0


def test_retrying_embeddings_custom_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = StubEmbeddingsAPI(succeed_on=3)
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    sleep_calls: list[float] = []
    monkeypatch.setattr("fancryrag.embeddings.time.sleep", lambda value: sleep_calls.append(value))

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
        timeout_seconds=5.0,
        max_retries=3,
        backoff_seconds=0.25,
    )

    result = embedder.embed_query("test query")

    assert result == [0.1, 0.2, 0.3]
    assert len(sleep_calls) == 2
    assert sleep_calls[0] == 0.25
    assert sleep_calls[1] == 0.5


def test_retrying_embeddings_backoff_caps_at_five_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = StubEmbeddingsAPI(succeed_on=5)
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    sleep_calls: list[float] = []
    monkeypatch.setattr("fancryrag.embeddings.time.sleep", lambda value: sleep_calls.append(value))

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
        timeout_seconds=5.0,
        max_retries=5,
        backoff_seconds=2.0,
    )

    result = embedder.embed_query("test query")

    assert result == [0.1, 0.2, 0.3]
    assert len(sleep_calls) == 4
    assert sleep_calls[0] == 2.0
    assert sleep_calls[1] == 4.0
    assert sleep_calls[2] == 5.0
    assert sleep_calls[3] == 5.0


def test_retrying_embeddings_passes_kwargs_to_api(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = StubEmbeddingsAPI(succeed_on=1)
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
        timeout_seconds=10.0,
        max_retries=3,
    )

    result = embedder.embed_query("test", dimensions=512)

    assert result == [0.1, 0.2, 0.3]
    assert api.calls[0]["dimensions"] == 512


def test_retrying_embeddings_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = StubEmbeddingsAPI(succeed_on=1)
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    embedder = RetryingOpenAIEmbeddings(
        model="custom-embedding-model-v3",
        base_url="http://embeddings.local",
        api_key="secret",
    )

    embedder.embed_query("test")

    assert api.calls[0]["model"] == "custom-embedding-model-v3"


def test_retrying_embeddings_inherits_from_openai_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

    # use monkeypatch to satisfy linter
    _ = monkeypatch

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
    )

    assert isinstance(embedder, OpenAIEmbeddings)


def test_retrying_embeddings_default_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = StubEmbeddingsAPI(succeed_on=2)
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    sleep_calls: list[float] = []
    monkeypatch.setattr("fancryrag.embeddings.time.sleep", lambda value: sleep_calls.append(value))

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
    )

    embedder.embed_query("test")

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == 0.5


def test_retrying_embeddings_raises_original_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = StubEmbeddingsAPI(succeed_on=999)
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
        max_retries=1,
    )

    with pytest.raises(RuntimeError) as exc_info:
        embedder.embed_query("test")

    assert "transient failure" in str(exc_info.value)


def test_retrying_embeddings_multiple_queries_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}
    api = StubEmbeddingsAPI(succeed_on=1)
    client = SimpleNamespace(embeddings=api)
    _patch_openai_client(monkeypatch, client, captured_kwargs)

    embedder = RetryingOpenAIEmbeddings(
        model="test-model",
        base_url="http://embeddings.local",
        api_key="secret",
    )

    result1 = embedder.embed_query("first query")
    result2 = embedder.embed_query("second query")

    assert result1 == [0.1, 0.2, 0.3]
    assert result2 == [0.1, 0.2, 0.3]
    assert len(api.calls) == 2
    assert api.calls[0]["input"] == "first query"
    assert api.calls[1]["input"] == "second query"