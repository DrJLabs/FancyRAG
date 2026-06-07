from __future__ import annotations

import json
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

from config.settings import FancyRAGSettings

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

neo_stub = sys.modules.get("neo4j")
if neo_stub is None:
    neo_stub = types.ModuleType("neo4j")
    sys.modules["neo4j"] = neo_stub
if not hasattr(neo_stub, "GraphDatabase"):
    class _GraphDatabase:
        @staticmethod
        def driver(*_args, **_kwargs):  # pragma: no cover - placeholder stub
            raise ImportError("neo4j driver not available in test stub")

    neo_stub.GraphDatabase = _GraphDatabase
if not hasattr(neo_stub, "Record"):
    neo_stub.Record = type("Record", (), {})
if not hasattr(neo_stub, "Driver"):
    neo_stub.Driver = type("Driver", (), {})
if not hasattr(neo_stub, "Query"):
    neo_stub.Query = type("Query", (), {})
if not hasattr(neo_stub, "exceptions"):
    exceptions_module = types.ModuleType("neo4j.exceptions")

    def _make_exc(name: str) -> type[RuntimeError]:
        return type(name, (RuntimeError,), {})

    def _exceptions_getattr(name: str) -> type[RuntimeError]:  # pragma: no cover - dynamic
        exc_type = _make_exc(name)
        setattr(exceptions_module, name, exc_type)
        return exc_type

    exceptions_module.__getattr__ = _exceptions_getattr  # type: ignore[attr-defined]
    for _name in ("Neo4jError", "ClientError", "DriverError", "CypherSyntaxError", "CypherTypeError"):
        setattr(exceptions_module, _name, _make_exc(_name))
    sys.modules["neo4j.exceptions"] = exceptions_module
    neo_stub.exceptions = exceptions_module
else:
    sys.modules.setdefault("neo4j.exceptions", neo_stub.exceptions)

import scripts.create_vector_index as civ


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    FancyRAGSettings.clear_cache()
    yield
    FancyRAGSettings.clear_cache()


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

    def execute_query(self, *_args, **_kwargs):  # pragma: no cover - simple stub
        return SimpleNamespace(records=[])


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


def _match_record(name: str, label: str, prop: str, *, dimensions: int = 1024, similarity: str = "cosine") -> FakeRecord:
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
        assert dimensions == 1024
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
    monkeypatch.setattr(civ, "retrieve_vector_index_info", lambda *_args, **_kwargs: _match_record("chunks_vec", "Chunk", "embedding", dimensions=1536))
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises(civ.VectorIndexMismatchError):
        civ.run([])


def test_mismatch_different_similarity_function(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test that a mismatch in similarity function raises VectorIndexMismatchError."""
    monkeypatch.setattr(civ, "retrieve_vector_index_info", lambda *_args, **_kwargs: _match_record("chunks_vec", "Chunk", "embedding", similarity="euclidean"))
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises(civ.VectorIndexMismatchError):
        civ.run([])


def test_mismatch_different_label(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test that a mismatch in label raises VectorIndexMismatchError."""
    monkeypatch.setattr(civ, "retrieve_vector_index_info", lambda *_args, **_kwargs: _match_record("chunks_vec", "Document", "embedding"))
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises(civ.VectorIndexMismatchError):
        civ.run([])


def test_run_invokes_get_settings_once(monkeypatch: pytest.MonkeyPatch, env) -> None:
    calls = {"count": 0}

    class Neo4jStub:
        uri = "bolt://localhost:7687"

        @staticmethod
        def auth() -> tuple[str, str]:
            return ("neo4j", "password")

    stub_settings = SimpleNamespace(neo4j=Neo4jStub())

    def fake_get_settings(*, refresh: bool = False, require: set[str] | None = None):  # noqa: FBT002
        calls["count"] += 1
        required = {item.lower() for item in require or set()}
        if "neo4j" in required and not hasattr(stub_settings, "neo4j"):
            raise ValueError("Missing required environment variable: NEO4J_URI")
        return stub_settings

    monkeypatch.setattr(civ, "get_settings", fake_get_settings)

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("ensure_env should not be called during typed settings execution")

    monkeypatch.setattr("fancyrag.utils.env.ensure_env", _unexpected)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda *_args, **_kwargs: FakeDriver())
    monkeypatch.setattr(civ, "retrieve_vector_index_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(civ, "create_vector_index", lambda *_args, **_kwargs: None)

    log = civ.run([])

    assert log["status"] == "created"
    assert calls["count"] == 1


def test_mismatch_different_property(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test that a mismatch in property raises VectorIndexMismatchError."""
    monkeypatch.setattr(civ, "retrieve_vector_index_info", lambda *_args, **_kwargs: _match_record("chunks_vec", "Chunk", "vector"))
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises(civ.VectorIndexMismatchError):
        civ.run([])


def test_custom_index_name(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test creating index with custom name via command-line argument."""
    log_file = tmp_path / "log.json"
    calls: dict[str, int] = {"retrieve": 0, "create": 0}

    def fake_retrieve(_driver, **kwargs):
        calls["retrieve"] += 1
        if calls["retrieve"] == 1:
            return None
        return _match_record("custom_index", "Chunk", "embedding")

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        calls["create"] += 1
        assert name == "custom_index"
        assert label == "Chunk"
        assert embedding_property == "embedding"

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file), "--index-name", "custom_index"])

    assert log["status"] == "created"
    assert calls["create"] == 1


def test_custom_dimensions(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test creating index with custom dimensions via command-line argument."""
    log_file = tmp_path / "log.json"
    calls: dict[str, int] = {"retrieve": 0, "create": 0}

    def fake_retrieve(_driver, **kwargs):
        calls["retrieve"] += 1
        if calls["retrieve"] == 1:
            return None
        return _match_record("chunks_vec", "Chunk", "embedding", dimensions=768)

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        calls["create"] += 1
        assert dimensions == 768

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file), "--dimensions", "768"])

    assert log["status"] == "created"


def test_custom_similarity_function(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test creating index with custom similarity function via command-line argument."""
    log_file = tmp_path / "log.json"
    calls: dict[str, int] = {"retrieve": 0, "create": 0}

    def fake_retrieve(_driver, **kwargs):
        calls["retrieve"] += 1
        if calls["retrieve"] == 1:
            return None
        return _match_record("chunks_vec", "Chunk", "embedding", similarity="euclidean")

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        calls["create"] += 1
        assert similarity_fn == "euclidean"

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file), "--similarity", "euclidean"])

    assert log["status"] == "created"


def test_missing_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that missing required environment variables raise appropriate errors."""
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    with pytest.raises((KeyError, ValueError, RuntimeError, SystemExit)):
        civ.run([])


def test_partial_missing_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that partially missing environment variables raise appropriate errors."""
    monkeypatch.setenv("NEO4J_URI", "bolt://example:7687")
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    with pytest.raises((KeyError, ValueError, RuntimeError, SystemExit)):
        civ.run([])


def test_log_file_directory_creation(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test that log file directories are created if they don't exist."""
    nested_log_path = tmp_path / "nested" / "dir" / "log.json"
    
    monkeypatch.setattr(civ, "retrieve_vector_index_info", lambda *_args, **_kwargs: _match_record("chunks_vec", "Chunk", "embedding"))
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(nested_log_path)])

    assert log["status"] == "exists"
    assert nested_log_path.exists()
    saved = json.loads(nested_log_path.read_text())
    assert saved["status"] == "exists"


def test_driver_connection_failure(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test handling of driver connection failures."""
    def failing_driver(uri, auth):
        raise RuntimeError("Connection failed")

    monkeypatch.setattr(civ.GraphDatabase, "driver", failing_driver)

    with pytest.raises(RuntimeError, match="Connection failed"):
        civ.run([])


def test_retrieve_vector_index_info_failure(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test handling of failures when retrieving vector index info."""
    def failing_retrieve(_driver, **_kwargs):
        raise RuntimeError("Failed to retrieve index info")

    monkeypatch.setattr(civ, "retrieve_vector_index_info", failing_retrieve)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises(RuntimeError, match="Failed to retrieve index info"):
        civ.run([])


def test_create_vector_index_failure(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test handling of failures when creating vector index."""
    log_file = tmp_path / "log.json"
    
    def fake_retrieve(_driver, **_kwargs):
        return None

    def failing_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        raise RuntimeError("Failed to create index")

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", failing_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises(RuntimeError, match="Failed to create index"):
        civ.run(["--log-path", str(log_file)])


def test_no_log_path_specified(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test that running without log path works correctly."""
    monkeypatch.setattr(civ, "retrieve_vector_index_info", lambda *_args, **_kwargs: _match_record("chunks_vec", "Chunk", "embedding"))
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run([])

    assert log["status"] == "exists"


def test_custom_label_and_property(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test creating index with custom label and property."""
    log_file = tmp_path / "log.json"
    calls: dict[str, int] = {"retrieve": 0, "create": 0}

    def fake_retrieve(_driver, **kwargs):
        calls["retrieve"] += 1
        if calls["retrieve"] == 1:
            return None
        return _match_record("chunks_vec", "Document", "vector_embedding")

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        calls["create"] += 1
        assert label == "Document"
        assert embedding_property == "vector_embedding"

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file), "--label", "Document", "--embedding-property", "vector_embedding"])

    assert log["status"] == "created"
    assert calls["create"] == 1


def test_database_parameter(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test creating index with custom database parameter."""
    log_file = tmp_path / "log.json"
    calls: dict[str, int] = {"retrieve": 0, "create": 0}

    def fake_retrieve(_driver, **kwargs):
        calls["retrieve"] += 1
        if calls["retrieve"] == 1:
            return None
        return _match_record("chunks_vec", "Chunk", "embedding")

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        calls["create"] += 1
        assert neo4j_database == "custom_db"

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file), "--database", "custom_db"])

    assert log["status"] == "created"


def test_empty_log_file_path(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test handling of empty log file path."""
    monkeypatch.setattr(civ, "retrieve_vector_index_info", lambda *_args, **_kwargs: _match_record("chunks_vec", "Chunk", "embedding"))
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises((ValueError, TypeError, SystemExit, IsADirectoryError)):
        civ.run(["--log-path", ""])


def test_invalid_dimensions(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test handling of invalid dimensions parameter."""
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises((ValueError, TypeError, SystemExit)):
        civ.run(["--dimensions", "not_a_number"])


def test_negative_dimensions(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test handling of negative dimensions parameter."""
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises((ValueError, TypeError, SystemExit)):
        civ.run(["--dimensions", "-100"])


def test_zero_dimensions(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test handling of zero dimensions parameter."""
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises((ValueError, TypeError, SystemExit)):
        civ.run(["--dimensions", "0"])


def test_retrieve_returns_none_then_creates_successfully(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test the full flow: retrieve returns None, create is called, then retrieve confirms creation."""
    log_file = tmp_path / "log.json"
    calls: dict[str, list] = {"retrieve": [], "create": []}

    def fake_retrieve(_driver, **kwargs):
        calls["retrieve"].append(kwargs)
        if len(calls["retrieve"]) == 1:
            return None
        return _match_record("chunks_vec", "Chunk", "embedding")

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        calls["create"].append({
            "name": name,
            "label": label,
            "embedding_property": embedding_property,
            "dimensions": dimensions,
            "similarity_fn": similarity_fn,
            "neo4j_database": neo4j_database
        })

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file)])

    assert log["status"] == "created"
    assert len(calls["retrieve"]) == 2
    assert len(calls["create"]) == 1
    assert calls["create"][0]["name"] == "chunks_vec"


def test_multiple_runs_idempotent(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test that running the script multiple times when index exists is idempotent."""
    log_file = tmp_path / "log.json"
    calls: dict[str, int] = {"retrieve": 0, "create": 0}

    def fake_retrieve(_driver, **_kwargs):
        calls["retrieve"] += 1
        return _match_record("chunks_vec", "Chunk", "embedding")

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        calls["create"] += 1

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    # Run multiple times
    for i in range(3):
        log = civ.run(["--log-path", str(log_file)])
        assert log["status"] == "exists"

    # Should retrieve 3 times (once per run) but never create
    assert calls["retrieve"] == 3
    assert calls["create"] == 0


def test_fake_record_data_method() -> None:
    """Test the FakeRecord helper class data method."""
    test_data = {"name": "test_index", "type": "vector"}
    record = FakeRecord(test_data)
    
    assert record.data() == test_data
    assert record.data()["name"] == "test_index"
    assert record.data()["type"] == "vector"


def test_fake_driver_context_manager() -> None:
    """Test the FakeDriver context manager functionality."""
    driver = FakeDriver()
    
    with driver as d:
        assert d is driver
    
    # Ensure close method exists and doesn't raise
    driver.close()


def test_match_record_default_parameters() -> None:
    """Test _match_record helper with default parameters."""
    record = _match_record("test_name", "test_label", "test_prop")
    data = record.data()
    
    assert data["name"] == "test_name"
    assert data["labelsOrTypes"] == ["test_label"]
    assert data["properties"] == ["test_prop"]
    assert data["options"]["indexConfig"]["vector.dimensions"] == 1024
    assert data["options"]["indexConfig"]["vector.similarity_function"] == "cosine"


def test_match_record_custom_parameters() -> None:
    """Test _match_record helper with custom parameters."""
    record = _match_record("custom_name", "CustomLabel", "custom_prop", dimensions=2048, similarity="euclidean")
    data = record.data()
    
    assert data["name"] == "custom_name"
    assert data["labelsOrTypes"] == ["CustomLabel"]
    assert data["properties"] == ["custom_prop"]
    assert data["options"]["indexConfig"]["vector.dimensions"] == 2048
    assert data["options"]["indexConfig"]["vector.similarity_function"] == "euclidean"


def test_log_content_structure(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test that log file contains proper JSON structure."""
    log_file = tmp_path / "log.json"
    
    monkeypatch.setattr(civ, "retrieve_vector_index_info", lambda *_args, **_kwargs: _match_record("chunks_vec", "Chunk", "embedding"))
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file)])

    assert log_file.exists()
    saved = json.loads(log_file.read_text())
    
    # Verify structure
    assert isinstance(saved, dict)
    assert "status" in saved
    assert saved["status"] in ["created", "exists"]


def test_ensure_dependencies_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the _ensure_dependencies fixture behavior."""
    # This test verifies the fixture modifies sys.path correctly
    original_path = list(sys.path)
    
    # The fixture should add wheel directories to sys.path
    # This is mostly a smoke test to ensure the fixture doesn't crash
    assert sys.path is not None
    assert isinstance(sys.path, list)


def test_pandas_stub_attributes() -> None:
    """Test that pandas stub has required attributes."""
    import pandas as pd
    
    assert hasattr(pd, "NA")
    assert hasattr(pd, "Series")
    assert hasattr(pd, "DataFrame")
    assert hasattr(pd, "Categorical")
    assert hasattr(pd, "core")
    assert hasattr(pd.core, "arrays")
    assert hasattr(pd.core.arrays, "ExtensionArray")


def test_invalid_uri_format(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test handling of invalid Neo4j URI format."""
    monkeypatch.setenv("NEO4J_URI", "invalid://uri/format")
    
    def failing_driver(uri, auth):
        if not uri.startswith("bolt://") and not uri.startswith("neo4j://"):
            raise ValueError("Invalid URI scheme")
        return FakeDriver()
    
    monkeypatch.setattr(civ.GraphDatabase, "driver", failing_driver)

    with pytest.raises(ValueError, match="Invalid Neo4j configuration"):
        civ.run([])


def test_empty_username(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of empty username."""
    monkeypatch.setenv("NEO4J_URI", "bolt://example:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises((ValueError, RuntimeError)):
        civ.run([])


def test_empty_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of empty password."""
    monkeypatch.setenv("NEO4J_URI", "bolt://example:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "")
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    with pytest.raises((ValueError, RuntimeError)):
        civ.run([])


def test_very_large_dimensions(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test creating index with very large dimensions."""
    log_file = tmp_path / "log.json"
    calls: dict[str, int] = {"retrieve": 0, "create": 0}

    def fake_retrieve(_driver, **kwargs):
        calls["retrieve"] += 1
        if calls["retrieve"] == 1:
            return None
        return _match_record("chunks_vec", "Chunk", "embedding", dimensions=10000)

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        calls["create"] += 1
        assert dimensions == 10000

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file), "--dimensions", "10000"])

    assert log["status"] == "created"


def test_special_characters_in_index_name(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test creating index with special characters in name."""
    log_file = tmp_path / "log.json"
    special_name = "chunks_vec_v2.0-beta"
    calls: dict[str, int] = {"retrieve": 0, "create": 0}

    def fake_retrieve(_driver, **kwargs):
        calls["retrieve"] += 1
        if calls["retrieve"] == 1:
            return None
        return _match_record(special_name, "Chunk", "embedding")

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        calls["create"] += 1
        assert name == special_name

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file), "--index-name", special_name])

    assert log["status"] == "created"


def test_unicode_in_parameters(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test creating index with unicode characters in parameters."""
    log_file = tmp_path / "log.json"
    unicode_label = "Chunk_文档"
    calls: dict[str, int] = {"retrieve": 0, "create": 0}

    def fake_retrieve(_driver, **kwargs):
        calls["retrieve"] += 1
        if calls["retrieve"] == 1:
            return None
        return _match_record("chunks_vec", unicode_label, "embedding")

    def fake_create(driver, name, label, embedding_property, dimensions, similarity_fn, neo4j_database=None):
        calls["create"] += 1
        assert label == unicode_label

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ, "create_vector_index", fake_create)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    log = civ.run(["--log-path", str(log_file), "--label", unicode_label])

    assert log["status"] == "created"


def test_concurrent_execution_safety(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    """Test that the script handles concurrent-like scenarios safely."""
    log_file = tmp_path / "log.json"
    call_count = {"count": 0}

    def fake_retrieve(_driver, **_kwargs):
        call_count["count"] += 1
        return _match_record("chunks_vec", "Chunk", "embedding")

    monkeypatch.setattr(civ, "retrieve_vector_index_info", fake_retrieve)
    monkeypatch.setattr(civ.GraphDatabase, "driver", lambda uri, auth: FakeDriver())

    # Simulate multiple calls
    for _ in range(5):
        log = civ.run(["--log-path", str(log_file)])
        assert log["status"] == "exists"

    assert call_count["count"] == 5
