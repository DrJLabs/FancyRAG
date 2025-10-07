from __future__ import annotations

import os

import pytest

from config.settings import FancyRAGSettings


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Ensure each test runs with a clean FancyRAGSettings cache."""

    FancyRAGSettings.clear_cache()
    yield
    FancyRAGSettings.clear_cache()


@pytest.fixture
def base_env(monkeypatch: pytest.MonkeyPatch):
    """Populate the minimum environment needed for FancyRAGSettings."""

    values = {
        "GRAPH_RAG_ACTOR": "pytest",
        "OPENAI_API_KEY": "sk-test",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
        "QDRANT_URL": "http://localhost:6333",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    yield values
    for key in values:
        monkeypatch.delenv(key, raising=False)


def test_load_returns_cached_instance(base_env):
    first = FancyRAGSettings.load()
    second = FancyRAGSettings.load()
    assert first is second

    FancyRAGSettings.clear_cache()
    third = FancyRAGSettings.load()
    assert third is not first


def test_load_refresh_updates_values(monkeypatch: pytest.MonkeyPatch, base_env):
    initial = FancyRAGSettings.load()
    assert initial.neo4j.uri == "bolt://localhost:7687"

    monkeypatch.setenv("NEO4J_URI", "bolt://override:7687")
    refreshed = FancyRAGSettings.load(refresh=True)
    assert refreshed.neo4j.uri == "bolt://override:7687"

    cached_after_refresh = FancyRAGSettings.load()
    assert cached_after_refresh is refreshed


def test_missing_required_openai_variable_raises(monkeypatch: pytest.MonkeyPatch, base_env):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        FancyRAGSettings.load(require={"openai"})


def test_openai_optional_when_not_required(monkeypatch: pytest.MonkeyPatch, base_env):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = FancyRAGSettings.load()

    assert settings.openai.api_key is None


def test_invalid_qdrant_url_rejected(monkeypatch: pytest.MonkeyPatch, base_env):
    monkeypatch.setenv("QDRANT_URL", "ftp://example.com")
    with pytest.raises(ValueError, match="Invalid Qdrant configuration"):
        FancyRAGSettings.load()


def test_qdrant_optional_when_not_required(monkeypatch: pytest.MonkeyPatch, base_env):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    settings = FancyRAGSettings.load()
    assert settings.qdrant is None

    with pytest.raises(ValueError, match="QDRANT_URL"):
        FancyRAGSettings.load(require={"qdrant"})


def test_export_environment_round_trip(base_env):
    settings = FancyRAGSettings.load()
    exported = settings.export_environment()

    assert exported["OPENAI_API_KEY"] == "sk-test"
    assert exported["NEO4J_URI"] == "bolt://localhost:7687"
    assert exported["QDRANT_URL"] == "http://localhost:6333"
    assert exported["OPENAI_ENABLE_FALLBACK"] == "true"


def test_clear_cache_handles_empty_cache():
    FancyRAGSettings.clear_cache()
    FancyRAGSettings.clear_cache()
    # If we reach here without an exception the behaviour is acceptable.
