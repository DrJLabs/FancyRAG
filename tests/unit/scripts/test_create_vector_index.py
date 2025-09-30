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
        """
        Initialize the FakeRecord with the provided record dictionary.
        
        Parameters:
            data (dict[str, object]): Mapping representing the record's data to store.
        """
        self._data = data

    def data(self) -> dict[str, object]:
        """
        Return the stored record dictionary.
        
        Returns:
            dict[str, object]: The underlying data mapping provided when the FakeRecord was created.
        """
        return self._data


class FakeDriver:
    def __enter__(self) -> "FakeDriver":
        """
        Return the context-manager instance for use with `with` statements.
        
        Returns:
            FakeDriver: The same FakeDriver instance (`self`).
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        No-op context manager exit method that performs no cleanup and does not suppress exceptions; exceptions propagate normally.
        """
        return None

    def close(self) -> None:  # pragma: no cover - compatibility shim
        """
        No-op close method provided for API compatibility with driver-like objects.
        
        This method intentionally performs no action and exists so instances can be used
        where a close() method is expected.
        """
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
    """
    Set environment variables used by tests to configure a Neo4j connection.
    
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Fixture used to set environment variables for the test process; this function sets
            NEO4J_URI to "bolt://example:7687", NEO4J_USERNAME to "neo4j", and NEO4J_PASSWORD to "test-password".
    """
    monkeypatch.setenv("NEO4J_URI", "bolt://example:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")


def _match_record(name: str, label: str, prop: str, *, dimensions: int = 1536, similarity: str = "cosine") -> FakeRecord:
    """
    Create a FakeRecord containing vector index metadata for tests.
    
    Parameters:
        name (str): Index name.
        label (str): Node label or relationship type associated with the index.
        prop (str): Property name used for the embedding vector.
        dimensions (int): Vector dimensionality (default 1536).
        similarity (str): Vector similarity function name (default "cosine").
    
    Returns:
        FakeRecord: A record with keys `name`, `labelsOrTypes`, `properties`, and `options.indexConfig`
        populated to reflect the provided vector index configuration.
    """
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
        """
        Increments the retrieve call counter and returns a matched FakeRecord after the first invocation.
        
        This helper increments calls["retrieve"] each time it is invoked. On the first call it returns None; on subsequent calls it returns a FakeRecord produced by _match_record("chunks_vec", "Chunk", "embedding").
        
        Returns:
            FakeRecord or None: `None` on the first invocation, otherwise a `FakeRecord` matching the vector index configuration.
        """
        calls["retrieve"] += 1
        if calls["retrieve"] == 1:
            return None
        return _match_record("chunks_vec", "Chunk", "embedding")

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        """
        Increment the test create-call counter and assert the received index creation arguments match expected fixtures.
        
        Parameters:
            driver: Neo4j driver (unused; accepted for interface compatibility).
            name: must be "chunks_vec".
            label: must be "Chunk".
            embedding_property: must be "embedding".
            dimensions: must be 1536.
            similarity_fn: must be "cosine".
            neo4j_database: must be None.
        
        Raises:
            AssertionError: if any argument does not match the expected value.
        """
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
        """
        Increment the shared create_calls counter used to record invocations.
        
        This function is a lightweight stub intended to act as a replacement for the real create function in tests; it increases create_calls["count"] by one each time it's called.
        """
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
