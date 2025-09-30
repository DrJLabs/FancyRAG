from __future__ import annotations

import json
import os
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

stub = sys.modules.get("pandas")
if stub is None:
    stub = types.ModuleType("pandas")
    sys.modules["pandas"] = stub
if not hasattr(stub, "NA"):
    stub.NA = object()
if not hasattr(stub, "Series"):
    stub.Series = type("Series", (), {})
if not hasattr(stub, "DataFrame"):
    stub.DataFrame = type("DataFrame", (), {})
if not hasattr(stub, "Categorical"):
    stub.Categorical = type("Categorical", (), {})
if not hasattr(stub, "core"):
    stub.core = types.SimpleNamespace()
if not hasattr(stub.core, "arrays"):
    stub.core.arrays = types.SimpleNamespace()
if not hasattr(stub.core.arrays, "ExtensionArray"):
    stub.core.arrays.ExtensionArray = type("ExtensionArray", (), {})

import scripts.create_vector_index as civ


class FakeRecord:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def data(self) -> dict[str, object]:
        return self._data


class FakeDriver:
    def __enter__(self) -> "FakeDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def close(self) -> None:  # pragma: no cover - compatibility shim
        return None


@pytest.fixture(autouse=True)
def _ensure_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure neo4j and neo4j_graphrag wheels are importable during tests."""

    wheel_dir = "/tmp"
    for name in os.listdir(wheel_dir):
        if name.startswith("neo4j_graphrag-") and name.endswith(".whl"):
            monkeypatch.syspath_prepend(os.path.join(wheel_dir, name))
        if name.startswith("neo4j-") and name.endswith(".whl"):
            monkeypatch.syspath_prepend(os.path.join(wheel_dir, name))


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://example:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")


def _match_record(name: str, label: str, prop: str, *, dimensions: int = 1536, similarity: str = "cosine") -> FakeRecord:
    return FakeRecord(
        {
            "name": name,
            "labelsOrTypes": [label],
            "properties": [prop],
            "options": {
                "indexConfig": {
                    "vector.dimensions": dimensions,
                    "vector.similarity_function": similarity,
                }
            },
        }
    )


def test_creates_index_when_missing(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    log_file = tmp_path / "log.json"
    calls: dict[str, int] = {"retrieve": 0, "create": 0}

    def fake_retrieve(_driver, **_kwargs):
        calls["retrieve"] += 1
        if calls["retrieve"] == 1:
            return None
        return _match_record("chunks_vec", "Chunk", "embedding")

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        calls["create"] += 1
        assert name == "chunks_vec"
        assert label == "Chunk"
        assert embedding_property == "embedding"
        assert dimensions == 1536
        assert similarity_fn == "cosine"
        assert neo4j_database is None

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file)])

    assert log["status"] == "created"
    assert calls == {"retrieve": 2, "create": 1}
    assert log_file.exists()
    saved = json.loads(log_file.read_text())
    assert saved["status"] == "created"


def test_skips_when_existing_matches(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    log_file = tmp_path / "log.json"
    monkeypatch.setattr(civ, "retrieve_vector_index_info", lambda *_args, **_kwargs: _match_record("chunks_vec", "Chunk", "embedding"))

    create_calls = {"count": 0}

    def fake_create(*_args, **_kwargs):
        create_calls["count"] += 1

    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file)])

    assert log["status"] == "exists"
    assert create_calls["count"] == 0
    saved = json.loads(log_file.read_text())
    assert saved["status"] == "exists"


def test_mismatch_raises(monkeypatch: pytest.MonkeyPatch, env) -> None:
    monkeypatch.setattr(civ, "retrieve_vector_index_info", lambda *_args, **_kwargs: _match_record("chunks_vec", "Chunk", "embedding", dimensions=1024))
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises(civ.VectorIndexMismatchError):
        civ.run([])
