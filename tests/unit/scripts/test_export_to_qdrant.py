#\!/usr/bin/env python
"""Comprehensive unit tests for export_to_qdrant.py script."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, Mock, call, patch

import pytest

pytest.importorskip(
    "qdrant_client",
    reason="qdrant-client is required to exercise Qdrant export integration",
)
pytest.importorskip(
    "qdrant_client.models",
    reason="qdrant-client models module is required to exercise Qdrant export integration",
)

from neo4j.exceptions import Neo4jError
import qdrant_client.models as qmodels

import scripts.export_to_qdrant as export_script

# Import functions to test
from scripts.export_to_qdrant import (
    _batched,
    _coerce_point_id,
    _fetch_chunks,
    main,
)

from config.settings import FancyRAGSettings


@pytest.fixture(autouse=True)
def _reset_fancyrag_settings_cache():
    FancyRAGSettings.clear_cache()
    yield
    FancyRAGSettings.clear_cache()


class TestFetchChunks:
    """Test suite for _fetch_chunks function."""

    def test_fetch_chunks_returns_list_of_dicts(self):
        """Test that _fetch_chunks returns a list of dictionaries."""
        mock_driver = Mock()
        mock_records = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": [0.1, 0.2, 0.3],
                "source_path": "/path/to/file.txt",
            },
            {
                "chunk_id": "2",
                "chunk_index": 1,
                "text": "Another text",
                "embedding": [0.4, 0.5, 0.6],
                "source_path": "/path/to/file2.txt",
            },
        ]
        mock_driver.execute_query.return_value = (mock_records, None, None)

        result = _fetch_chunks(mock_driver, database="test_db")

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(record, dict) for record in result)
        mock_driver.execute_query.assert_called_once()

    def test_fetch_chunks_with_none_database(self):
        """Test _fetch_chunks when database parameter is None."""
        mock_driver = Mock()
        mock_records = []
        mock_driver.execute_query.return_value = (mock_records, None, None)

        result = _fetch_chunks(mock_driver, database=None)

        assert result == []
        mock_driver.execute_query.assert_called_once()
        call_args = mock_driver.execute_query.call_args
        assert call_args[1]["database_"] is None

    def test_fetch_chunks_empty_result(self):
        """Test _fetch_chunks when no chunks are found."""
        mock_driver = Mock()
        mock_driver.execute_query.return_value = ([], None, None)

        result = _fetch_chunks(mock_driver, database="test_db")

        assert result == []

    def test_fetch_chunks_with_various_data_types(self):
        """Test _fetch_chunks with different data types in records."""
        mock_driver = Mock()
        mock_records = [
            {
                "chunk_id": 123,
                "chunk_index": 0,
                "text": "Text with integer ID",
                "embedding": [0.1] * 384,
                "source_path": "path/to/source.txt",
            },
            {
                "chunk_id": None,
                "chunk_index": 1,
                "text": "Text with None ID",
                "embedding": [0.2] * 384,
                "source_path": None,
            },
        ]
        mock_driver.execute_query.return_value = (mock_records, None, None)

        result = _fetch_chunks(mock_driver, database="neo4j")

        assert len(result) == 2
        assert result[0]["chunk_id"] == 123
        assert result[1]["chunk_id"] is None

    def test_fetch_chunks_preserves_all_fields(self):
        """Test that _fetch_chunks preserves all record fields."""
        mock_driver = Mock()
        expected_record = {
            "chunk_id": "test-id",
            "chunk_index": 42,
            "text": "Test text content",
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "source_path": "/test/path/file.md",
            "relative_path": "docs/file.md",
            "git_commit": "abc123",
            "checksum": "deadbeef",
        }
        mock_driver.execute_query.return_value = ([expected_record], None, None)

        result = _fetch_chunks(mock_driver, database="test")

        assert len(result) == 1
        assert result[0] == expected_record


class TestBatched:
    """Test suite for _batched function."""

    def test_batched_even_division(self):
        """Test _batched with items that divide evenly into batches."""
        items = [{"id": i} for i in range(10)]
        batches = list(_batched(items, 5))

        assert len(batches) == 2
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5
        assert batches[0][0]["id"] == 0
        assert batches[1][0]["id"] == 5

    def test_batched_uneven_division(self):
        """Test _batched with items that don't divide evenly."""
        items = [{"id": i} for i in range(7)]

        batches = list(_batched(items, 3))

        assert len(batches) == 3
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 1

    def test_batched_single_item(self):
        """Test _batched with a single item."""
        items = [{"id": 1}]
        batches = list(_batched(items, 10))

        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_batched_empty_iterable(self):
        """Test _batched with an empty iterable."""
        items = []
        batches = list(_batched(items, 5))

        assert len(batches) == 0

    def test_batched_size_one(self):
        """Test _batched with batch size of 1."""
        items = [{"id": i} for i in range(3)]
        batches = list(_batched(items, 1))

        assert len(batches) == 3
        assert all(len(batch) == 1 for batch in batches)

    def test_batched_size_larger_than_items(self):
        """Test _batched when batch size is larger than number of items."""
        items = [{"id": i} for i in range(3)]
        batches = list(_batched(items, 10))

        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_batched_preserves_order(self):
        """Test that _batched preserves the order of items."""
        items = [{"id": i, "value": f"item_{i}"} for i in range(15)]
        batches = list(_batched(items, 4))

        flat_items = [item for batch in batches for item in batch]
        assert flat_items == items

    def test_batched_with_generator(self):
        """Test _batched with a generator as input."""
        def item_generator():
            for i in range(5):
                yield {"id": i}

        batches = list(_batched(item_generator(), 2))

        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1


class TestCoercePointId:
    """Test suite for _coerce_point_id function."""

    def test_coerce_point_id_with_none(self):
        """Test _coerce_point_id returns fallback when value is None."""
        result = _coerce_point_id(None, fallback=42)
        assert result == 42

    def test_coerce_point_id_with_integer(self):
        """Test _coerce_point_id returns integer as-is."""
        result = _coerce_point_id(123, fallback=0)
        assert result == 123
        assert isinstance(result, int)

    def test_coerce_point_id_with_numeric_string(self):
        """Test _coerce_point_id converts numeric string to int."""
        result = _coerce_point_id("456", fallback=0)
        assert result == 456
        assert isinstance(result, int)

    def test_coerce_point_id_with_numeric_string_whitespace(self):
        """Test _coerce_point_id strips whitespace from numeric string."""
        result = _coerce_point_id("  789  ", fallback=0)
        assert result == 789
        assert isinstance(result, int)

    def test_coerce_point_id_with_non_numeric_string(self):
        """Test _coerce_point_id returns string for non-numeric values."""
        result = _coerce_point_id("chunk-abc-123", fallback=0)
        assert result == "chunk-abc-123"
        assert isinstance(result, str)

    def test_coerce_point_id_with_empty_string(self):
        """Test _coerce_point_id handles empty string."""
        result = _coerce_point_id("", fallback=100)
        assert result == ""
        assert isinstance(result, str)

    def test_coerce_point_id_with_whitespace_only_string(self):
        """Test _coerce_point_id handles whitespace-only string."""
        result = _coerce_point_id("   ", fallback=100)
        assert result == ""
        assert isinstance(result, str)

    def test_coerce_point_id_with_float(self):
        """Test _coerce_point_id converts float to string."""
        result = _coerce_point_id(12.34, fallback=0)
        assert result == "12.34"
        assert isinstance(result, str)

    def test_coerce_point_id_with_boolean(self):
        """Test _coerce_point_id handles boolean values."""
        result = _coerce_point_id(True, fallback=0)
        assert result == "True"
        assert isinstance(result, str)

    def test_coerce_point_id_with_list(self):
        """Test _coerce_point_id converts list to string."""
        result = _coerce_point_id([1, 2, 3], fallback=0)
        assert result == "[1, 2, 3]"
        assert isinstance(result, str)

    def test_coerce_point_id_with_dict(self):
        """Test _coerce_point_id converts dict to string."""
        result = _coerce_point_id({"key": "value"}, fallback=0)
        assert isinstance(result, str)
        assert "key" in result

    def test_coerce_point_id_with_zero(self):
        """Test _coerce_point_id handles zero integer."""
        result = _coerce_point_id(0, fallback=999)
        assert result == 0
        assert isinstance(result, int)

    def test_coerce_point_id_with_negative_integer(self):
        """Test _coerce_point_id handles negative integers."""
        result = _coerce_point_id(-42, fallback=0)
        assert result == -42
        assert isinstance(result, int)

    def test_coerce_point_id_with_uuid_string(self):
        """Test _coerce_point_id handles UUID-like strings."""
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        result = _coerce_point_id(uuid_str, fallback=0)
        assert result == uuid_str
        assert isinstance(result, str)


@pytest.fixture
def settings_stub(monkeypatch):
    """Provide a stubbed FancyRAG settings aggregate for exporter tests."""

    class Neo4jStub:
        @property
        def uri(self) -> str:
            return os.environ.get("NEO4J_URI", "bolt://localhost:7687")

        @property
        def database(self) -> str | None:
            return os.environ.get("NEO4J_DATABASE")

        def auth(self) -> tuple[str, str]:
            return (
                os.environ.get("NEO4J_USERNAME", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", "password"),
            )

    class QdrantStub:
        @property
        def neo4j_id_property(self) -> str:
            return os.environ.get("QDRANT_NEO4J_ID_PROPERTY_NEO4J", "chunk_id")

        @property
        def external_id_property(self) -> str:
            return os.environ.get("QDRANT_NEO4J_ID_PROPERTY_EXTERNAL", "chunk_id")

        def client_kwargs(self) -> dict[str, str]:
            kwargs = {"url": os.environ.get("QDRANT_URL", "http://localhost:6333")}
            api_key = os.environ.get("QDRANT_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key
            return kwargs

    stub = SimpleNamespace(neo4j=Neo4jStub(), qdrant=QdrantStub())

    monkeypatch.setattr("scripts.export_to_qdrant.get_settings", lambda: stub)
    return stub


@pytest.mark.usefixtures("settings_stub")
class TestMain:
    """Test suite for main function."""

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_success_with_chunks(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test main function successfully exports chunks."""
        # Setup mocks
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": [0.1, 0.2, 0.3],
                "source_path": "/path/to/file.txt",
                "relative_path": "docs/file.txt",
                "git_commit": "commit1",
                "checksum": "aaa",
            },
            {
                "chunk_id": "2",
                "chunk_index": 1,
                "text": "Another text",
                "embedding": [0.4, 0.5, 0.6],
                "source_path": "/path/to/file2.txt",
                "relative_path": "src/file2.txt",
                "git_commit": "commit1",
                "checksum": "bbb",
            },
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_artifacts_dir.__truediv__ = Mock(return_value=Mock())

        mock_scrub.return_value = {"status": "success"}

        # Run main
        with patch("sys.argv", ["export_to_qdrant.py", "--collection", "test_collection"]):
            main()

        # Assertions
        mock_graph_db.driver.assert_called_once()
        mock_client.create_collection.assert_called_once()
        mock_client.upsert.assert_called_once()
        payload_batch = mock_client.upsert.call_args.kwargs["points"]
        assert payload_batch.payloads[0]["relative_path"] == "docs/file.txt"
        assert payload_batch.payloads[0]["checksum"] == "aaa"
        assert payload_batch.payloads[1]["relative_path"] == "src/file2.txt"

    def test_main_calls_get_settings_once(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        calls = {"count": 0}

        class Neo4jStub:
            uri = "bolt://localhost:7687"
            database = None

            @staticmethod
            def auth() -> tuple[str, str]:
                return ("neo4j", "password")

        class QdrantStub:
            def client_kwargs(self) -> dict[str, str]:
                return {"url": "http://localhost:6333"}

        class OpenAIStub:
            def for_actor(self, actor: str):
                return SimpleNamespace(actor=actor)

        stub_settings = SimpleNamespace(
            neo4j=Neo4jStub(),
            qdrant=QdrantStub(),
            openai=OpenAIStub(),
        )

        def fake_get_settings(*, refresh: bool = False, require: set[str] | None = None):  # noqa: FBT002
            calls["count"] += 1
            required = {item.lower() for item in require or set()}
            if "qdrant" in required and stub_settings.qdrant is None:
                raise ValueError("Missing required environment variable: QDRANT_URL")
            return stub_settings

        class FakeDriver:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def execute_query(self, *_args, **_kwargs):
                record = {
                    "chunk_id": "1",
                    "chunk_uid": "chunk-1",
                    "chunk_index": 0,
                    "text": "example",
                    "embedding": [0.1, 0.2, 0.3],
                    "source_path": "doc.txt",
                    "relative_path": "docs/doc.txt",
                    "git_commit": "deadbeef",
                    "checksum": "aaa",
                }
                return ([record], None, None)

        class FakeQdrantClient:
            def __init__(self, **_kwargs):
                self.created = False

            def collection_exists(self, _name):
                return False

            def create_collection(self, **_kwargs):
                self.created = True

            def get_collection(self, _name):
                return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=3))))

            def upsert(self, **_kwargs):
                return SimpleNamespace()

        monkeypatch.setattr(export_script, "get_settings", fake_get_settings)

        def _unexpected(*_args, **_kwargs):
            raise AssertionError("ensure_env should not be invoked during typed settings path")

        monkeypatch.setattr("fancyrag.utils.env.ensure_env", _unexpected)
        monkeypatch.setattr(export_script, "GraphDatabase", SimpleNamespace(driver=lambda *_args, **_kwargs: FakeDriver()))
        monkeypatch.setattr(export_script, "QdrantClient", FakeQdrantClient)
        monkeypatch.setattr(export_script.qmodels, "VectorParams", lambda size, distance: SimpleNamespace(size=size, distance=distance))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PYTHONPATH", "src")

        monkeypatch.setattr(sys, "argv", ["export_to_qdrant.py", "--collection", "chunks_main", "--batch-size", "1"])

        export_script.main()

        assert calls["count"] == 1

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_skipped_no_chunks(
        self, mock_graph_db, _mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test main function when no chunks are available."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver
        mock_driver.execute_query.return_value = ([], None, None)

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "skipped", "message": "No chunk nodes available to export"}

        with patch("sys.argv", ["export_to_qdrant.py"]):
            main()

        mock_scrub.assert_called_once()
        scrub_arg = mock_scrub.call_args[0][0]
        assert scrub_arg["status"] == "skipped"
        assert "No chunk nodes available" in scrub_arg["message"]
        assert scrub_arg["skipped_chunks"] == 0

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_skips_empty_embeddings(
        self, mock_graph_db, _mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test main function skips chunks that have empty embeddings."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": [],
                "source_path": "/path/to/file.txt",
            }
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "skipped"}

        with patch("sys.argv", ["export_to_qdrant.py"]):
            main()

        scrub_arg = mock_scrub.call_args[0][0]
        assert scrub_arg["status"] == "skipped"
        assert "Skipped 1 chunk" in scrub_arg["message"]
        assert scrub_arg["skipped_chunks"] == 1

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_with_batch_size_argument(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test main function with custom batch size argument."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        # Create enough chunks to test batching
        mock_chunks = [
            {
                "chunk_id": str(i),
                "chunk_index": i,
                "text": f"Text {i}",
                "embedding": [0.1] * 384,
                "source_path": f"/path/file{i}.txt",
            }
            for i in range(10)
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "success"}

        with patch("sys.argv", ["export_to_qdrant.py", "--batch-size", "3"]):
            main()

        # Should have been called 4 times (10 items / 3 batch size = 3 full + 1 partial)
        assert mock_client.upsert.call_count == 4

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_deletes_existing_collection(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test main function deletes existing collection before creating new one."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": [0.1, 0.2, 0.3],
                "source_path": "/path/to/file.txt",
            }
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = True

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "success"}

        with patch(
            "sys.argv",
            ["export_to_qdrant.py", "--collection", "existing_collection", "--recreate-collection"],
        ):
            main()

        mock_client.delete_collection.assert_called_once_with("existing_collection")
        mock_client.create_collection.assert_called_once()

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_preserves_existing_collection_when_not_recreating(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": [0.1, 0.2, 0.3],
                "source_path": "/path/to/file.txt",
            }
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        existing_vectors = SimpleNamespace(size=3)
        collection_info = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=existing_vectors))
        )

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = collection_info

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "success"}

        with patch("sys.argv", ["export_to_qdrant.py"]):
            main()

        mock_client.delete_collection.assert_not_called()
        mock_client.create_collection.assert_not_called()

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_existing_collection_dimension_mismatch_requires_recreate(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": [0.1, 0.2, 0.3],
                "source_path": "/path/to/file.txt",
            }
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        remote_vectors = SimpleNamespace(size=8)
        collection_info = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=remote_vectors))
        )

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = collection_info

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.side_effect = lambda payload: payload

        with patch("sys.argv", ["export_to_qdrant.py"]):
            with pytest.raises(SystemExit) as excinfo:
                main()

        assert excinfo.value.code == 1
        error_message = mock_scrub.call_args[0][0]["message"]
        assert "--recreate-collection" in error_message

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "QDRANT_API_KEY": "test-api-key",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_with_qdrant_api_key(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test main function uses Qdrant API key when provided."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": [0.1, 0.2, 0.3],
                "source_path": "/path/to/file.txt",
            }
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "success"}

        with patch("sys.argv", ["export_to_qdrant.py"]):
            main()

        mock_qdrant_client.assert_called_once_with(
            url="http://localhost:6333", api_key="test-api-key"
        )

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_neo4j_error_handling(
        self, mock_graph_db, mock_scrub, mock_path
    ):
        """Test main function handles Neo4j errors gracefully."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver
        mock_driver.execute_query.side_effect = Neo4jError("Connection failed")

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "error", "message": "Connection failed"}

        with patch("sys.argv", ["export_to_qdrant.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1

        scrub_arg = mock_scrub.call_args[0][0]
        assert scrub_arg["status"] == "error"
        assert scrub_arg["message"] == "Connection failed"

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
            "NEO4J_DATABASE": "custom_db",
        },
    )
    def test_main_with_neo4j_database_env(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test main function uses NEO4J_DATABASE environment variable."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": [0.1, 0.2, 0.3],
                "source_path": "/path/to/file.txt",
            }
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "success"}

        with patch("sys.argv", ["export_to_qdrant.py"]):
            main()

        # Verify execute_query was called with the custom database
        call_kwargs = mock_driver.execute_query.call_args[1]
        assert call_kwargs["database_"] == "custom_db"

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_creates_artifacts_directory(
        self, mock_graph_db, _mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test main function creates artifacts directory."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver
        mock_driver.execute_query.return_value = ([], None, None)

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "skipped"}

        with patch("sys.argv", ["export_to_qdrant.py"]):
            main()

        mock_path.assert_called_once_with("artifacts/local_stack")
        mock_artifacts_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_writes_json_artifact(
        self, mock_graph_db, _mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test main function writes JSON artifact file."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_driver.execute_query.return_value = ([], None, None)

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        sanitized_log = {
            "timestamp": "2024-01-01T00:00:00Z",
            "operation": "export_to_qdrant",
            "status": "skipped",
            "message": "No chunk nodes available to export",
        }
        mock_scrub.return_value = sanitized_log

        with patch("sys.argv", ["export_to_qdrant.py"]):
            main()

        mock_file.write_text.assert_called_once()
        call_args = mock_file.write_text.call_args
        assert "encoding" in call_args[1]
        assert call_args[1]["encoding"] == "utf-8"

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_payload_structure(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test that main function creates correct payload structure for Qdrant."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "test-id",
                "chunk_index": 5,
                "text": "Test content",
                "embedding": [0.1, 0.2, 0.3],
                "source_path": "/test/path.txt",
            }
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "success"}

        with patch("sys.argv", ["export_to_qdrant.py"]):
            main()

        # Verify upsert was called with correct structure
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args[1]
        batch = call_args["points"]

        assert len(batch.ids) == 1
        assert batch.ids[0] == "test-id"
        assert len(batch.vectors) == 1
        assert batch.vectors[0] == [0.1, 0.2, 0.3]
        assert len(batch.payloads) == 1
        assert batch.payloads[0]["chunk_id"] == "test-id"
        assert batch.payloads[0]["chunk_index"] == 5
        assert batch.payloads[0]["text"] == "Test content"
        assert batch.payloads[0]["source_path"] == "/test/path.txt"

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_vector_config_cosine_distance(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test that main function configures Qdrant collection with COSINE distance."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": [0.1] * 384,
                "source_path": "/path/to/file.txt",
            }
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "success"}

        with patch("sys.argv", ["export_to_qdrant.py"]):
            main()

        # Verify create_collection was called with COSINE distance
        call_args = mock_client.create_collection.call_args[1]
        vectors_config = call_args["vectors_config"]
        assert vectors_config.size == 384
        assert vectors_config.distance == qmodels.Distance.COSINE

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_fallback_id_generation(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test that main function generates fallback IDs correctly."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        # Create chunks with None chunk_id to test fallback
        mock_chunks = [
            {
                "chunk_id": None,
                "chunk_index": i,
                "text": f"Text {i}",
                "embedding": [0.1] * 384,
                "source_path": f"/path/file{i}.txt",
            }
            for i in range(3)
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "success"}

        with patch("sys.argv", ["export_to_qdrant.py"]):
            main()

        # Verify upsert was called and check IDs
        call_args = mock_client.upsert.call_args[1]
        batch = call_args["points"]

        # Fallback IDs should be 1, 2, 3 (exported + idx + 1)
        assert batch.ids == [1, 2, 3]

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_log_structure(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test that main function creates proper log structure."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": [0.1, 0.2, 0.3],
                "source_path": "/path/to/file.txt",
            }
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        def scrub_passthrough(log):
            return log

        mock_scrub.side_effect = scrub_passthrough

        with patch("sys.argv", ["export_to_qdrant.py", "--collection", "test_col"]):
            main()

        # Verify log structure passed to scrub_object
        scrub_arg = mock_scrub.call_args[0][0]
        assert "timestamp" in scrub_arg
        assert scrub_arg["operation"] == "export_to_qdrant"
        assert scrub_arg["collection"] == "test_col"
        assert scrub_arg["status"] in ["success", "error", "skipped"]
        assert "message" in scrub_arg
        assert "count" in scrub_arg
        assert "skipped_chunks" in scrub_arg
        assert "duration_ms" in scrub_arg
        assert isinstance(scrub_arg["duration_ms"], int)

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_handles_none_embeddings(
        self, mock_graph_db, _mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test main function when embeddings are None."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": None,
                "source_path": "/path/to/file.txt",
            }
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "error"}

        with patch("sys.argv", ["export_to_qdrant.py"]):
            main()

        log_arg = mock_scrub.call_args[0][0]
        assert log_arg["status"] == "skipped"
        assert log_arg["skipped_chunks"] == 1

    @patch("scripts.export_to_qdrant.Path")
    @patch("scripts.export_to_qdrant.scrub_object")
    @patch("scripts.export_to_qdrant.QdrantClient")
    @patch("scripts.export_to_qdrant.GraphDatabase")
    @patch.dict(
        os.environ,
        {
            "QDRANT_URL": "http://localhost:6333",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "password",
        },
    )
    def test_main_batch_size_minimum_one(
        self, mock_graph_db, mock_qdrant_client, mock_scrub, mock_path
    ):
        """Test that main function enforces minimum batch size of 1."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value.__enter__.return_value = mock_driver

        mock_chunks = [
            {
                "chunk_id": "1",
                "chunk_index": 0,
                "text": "Sample text",
                "embedding": [0.1, 0.2, 0.3],
                "source_path": "/path/to/file.txt",
            }
        ]
        mock_driver.execute_query.return_value = (mock_chunks, None, None)

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.collection_exists.return_value = False

        mock_artifacts_dir = Mock()
        mock_path.return_value = mock_artifacts_dir
        mock_file = Mock()
        mock_artifacts_dir.__truediv__ = Mock(return_value=mock_file)

        mock_scrub.return_value = {"status": "success"}

        # Test with batch size 0 (should be coerced to 1)
        with patch("sys.argv", ["export_to_qdrant.py", "--batch-size", "0"]):
            main()

        # Should still process the chunk
        assert mock_client.upsert.call_count == 1


class TestEdgeCasesAndIntegration:
    """Additional edge case and integration tests."""

    def test_batched_with_complex_dictionaries(self):
        """Test _batched with complex nested dictionaries."""
        items = [
            {
                "id": i,
                "nested": {"values": [1, 2, 3]},
                "metadata": {"key": f"value_{i}"},
            }
            for i in range(5)
        ]
        batches = list(_batched(items, 2))

        assert len(batches) == 3
        assert batches[0][0]["nested"]["values"] == [1, 2, 3]

    def test_coerce_point_id_with_special_characters(self):
        """Test _coerce_point_id with special characters in strings."""
        test_cases = [
            ("chunk-id-with-dashes", "chunk-id-with-dashes"),
            ("chunk_id_with_underscores", "chunk_id_with_underscores"),
            ("chunk:id:with:colons", "chunk:id:with:colons"),
            ("chunk/id/with/slashes", "chunk/id/with/slashes"),
            ("chunk.id.with.dots", "chunk.id.with.dots"),
        ]

        for input_val, expected in test_cases:
            result = _coerce_point_id(input_val, fallback=0)
            assert result == expected
            assert isinstance(result, str)

    def test_coerce_point_id_with_large_numbers(self):
        """Test _coerce_point_id with very large numbers."""
        large_int = 9999999999999999
        result = _coerce_point_id(large_int, fallback=0)
        assert result == large_int
        assert isinstance(result, int)

        large_str = "9999999999999999"
        result = _coerce_point_id(large_str, fallback=0)
        assert result == 9999999999999999
        assert isinstance(result, int)

    def test_batched_preserves_dict_mutations(self):
        """Test that _batched doesn't mutate the original dictionaries."""
        original_items = [{"id": i, "value": f"val_{i}"} for i in range(3)]
        items_copy = [item.copy() for item in original_items]

        _batches = list(_batched(original_items, 2))

        # Verify original items weren't mutated
        assert original_items == items_copy

    def test_coerce_point_id_boundary_conditions(self):
        """Test _coerce_point_id with boundary integer values."""
        # Test minimum and maximum integer values
        assert _coerce_point_id(0, fallback=999) == 0
        assert _coerce_point_id(-1, fallback=999) == -1
        assert _coerce_point_id(2**31 - 1, fallback=0) == 2**31 - 1
        assert _coerce_point_id(-(2**31), fallback=0) == -(2**31)

    def test_batched_with_single_large_batch(self):
        """Test _batched when batch size equals number of items."""
        items = [{"id": i} for i in range(100)]
        batches = list(_batched(items, 100))

        assert len(batches) == 1
        assert len(batches[0]) == 100

    def test_coerce_point_id_with_numeric_string_leading_zeros(self):
        """Test _coerce_point_id with numeric strings containing leading zeros."""
        result = _coerce_point_id("00123", fallback=0)
        assert result == 123
        assert isinstance(result, int)

        result = _coerce_point_id("  00456  ", fallback=0)
        assert result == 456
        assert isinstance(result, int)

    def test_coerce_point_id_with_mixed_alphanumeric(self):
        """Test _coerce_point_id with mixed alphanumeric strings."""
        test_cases = [
            "123abc",
            "abc123",
            "12a34b56",
            "chunk-123-abc",
        ]

        for test_val in test_cases:
            result = _coerce_point_id(test_val, fallback=0)
            assert result == test_val
            assert isinstance(result, str)
