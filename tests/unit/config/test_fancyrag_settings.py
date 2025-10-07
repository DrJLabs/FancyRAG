from __future__ import annotations

import os
from pathlib import Path

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
    assert exported["FANCYRAG_PRESET"] == "smoke"


def test_clear_cache_handles_empty_cache():
    FancyRAGSettings.clear_cache()
    FancyRAGSettings.clear_cache()
    # If we reach here without an exception the behaviour is acceptable.


def test_service_settings_defaults(base_env):
    settings = FancyRAGSettings.load(refresh=True)
    service = settings.service
    assert service.preset == "smoke"
    assert service.dataset_path == "docs/samples/pilot.txt"
    assert service.dataset_dir is None
    assert service.telemetry == "console"
    resolved = service.resolve_dataset_path(repo_root=Path("/repo"))
    assert resolved == Path("/repo/docs/samples/pilot.txt")


def test_service_settings_env_overrides(monkeypatch: pytest.MonkeyPatch, base_env):
    monkeypatch.setenv("FANCYRAG_PRESET", "enrich")
    monkeypatch.setenv("DATASET_DIR", "data/full")
    monkeypatch.setenv("FANCYRAG_ENABLE_SEMANTIC", "1")
    monkeypatch.setenv("FANCYRAG_TELEMETRY", "otlp")
    monkeypatch.setenv("FANCYRAG_VECTOR_INDEX", "custom_vec")
    monkeypatch.setenv("FANCYRAG_QDRANT_COLLECTION", "custom_collection")

    settings = FancyRAGSettings.load(refresh=True)
    service = settings.service

    assert service.preset == "enrich"
    assert service.dataset_dir == "data/full"
    assert service.dataset_path is None
    assert service.semantic_enabled is True
    assert service.telemetry == "otlp"
    assert service.vector_index == "custom_vec"
    assert service.collection == "custom_collection"
