from __future__ import annotations

from types import SimpleNamespace

import pytest
from neo4j.exceptions import Neo4jError

from fancyrag.db.neo4j_queries import (
    collect_counts,
    collect_semantic_counts,
    count_checksum_mismatches,
    count_missing_embeddings,
    count_orphan_chunks,
    ensure_document_relationships,
    reset_database,
    rollback_ingest,
)


class RecordingDriver:
    """Test double that captures Cypher queries and returns seeded responses."""

    def __init__(self, responses: list[object] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[tuple[str, dict[str, object], str | None]] = []

    def execute_query(self, query, parameters=None, database_=None):
        params = dict(parameters or {})
        self.calls.append((" ".join(str(query).split()), params, database_))
        if self._responses:
            return self._responses.pop(0)
        return []


class ErrorDriver(RecordingDriver):
    """Driver double that raises Neo4j errors for selected calls."""

    def __init__(self, error_indices: set[int]) -> None:
        super().__init__()
        self._error_indices = set(error_indices)

    def execute_query(self, query, parameters=None, database_=None):
        index = len(self.calls)
        if index in self._error_indices:
            raise Neo4jError("boom")
        return super().execute_query(query, parameters, database_)


def _chunk_metadata(uid: str, *, sequence: int = 1, index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        uid=uid,
        sequence=sequence,
        index=index,
        checksum="deadbeef",
        relative_path="docs/file.txt",
        git_commit="abc123",
    )


def _qa_chunk(uid: str) -> SimpleNamespace:
    return SimpleNamespace(uid=uid)


def _qa_source(uid: str, *, ingest_run_key: str | None = "run-1") -> SimpleNamespace:
    return SimpleNamespace(
        chunks=[_qa_chunk(uid)],
        relative_path="docs/file.txt",
        ingest_run_key=ingest_run_key,
    )


def test_reset_database_executes_delete():
    driver = RecordingDriver()
    reset_database(driver, database="neo4j")
    assert driver.calls == [
        ("MATCH (n) DETACH DELETE n", {}, "neo4j")
    ]


def test_ensure_document_relationships_builds_payload(tmp_path):
    driver = RecordingDriver()
    source_path = tmp_path / "sample.md"
    source_path.write_text("data", encoding="utf-8")

    ensure_document_relationships(
        driver,
        database="neo4j",
        source_path=source_path,
        relative_path="docs/sample.md",
        git_commit="abc123",
        document_checksum="cafebabe",
        chunks_metadata=[_chunk_metadata("chunk-1")],
    )

    assert len(driver.calls) == 1
    query, params, database = driver.calls[0]
    assert "MERGE (doc:Document {source_path: $source_path})" in query
    assert params["source_path"] == str(source_path)
    assert params["document_name"] == source_path.name
    assert params["chunk_payload"][0]["uid"] == "chunk-1"
    assert database == "neo4j"


def test_rollback_ingest_executes_cleanup_queries():
    driver = RecordingDriver()
    rollback_ingest(
        driver,
        database="neo4j",
        sources=[_qa_source("chunk-1"), _qa_source("chunk-2", ingest_run_key=None)],
    )

    queries = [call[0] for call in driver.calls]
    assert "MATCH ()-[rel]-()" in queries[0]
    assert "MATCH (node)" in queries[1]
    assert "MATCH (c:Chunk" in queries[2]
    assert "MATCH (doc:Document" in queries[3]


def test_collect_counts_normalises_driver_payloads():
    driver = RecordingDriver(
        responses=[([{"value": 2}],), [{"value": 5}], SimpleNamespace(records=[{"value": 7}])]
    )
    counts = collect_counts(driver, database="neo4j")
    assert counts == {"documents": 2, "chunks": 5, "relationships": 7}


def test_collect_counts_handles_driver_errors():
    driver = ErrorDriver({1})
    counts = collect_counts(driver, database="neo4j")
    # documents count succeeds, chunks fails (ignored), relationships succeeds
    assert counts["documents"] == 0
    assert counts["relationships"] == 0


def test_count_helpers_return_zero_for_empty_inputs():
    driver = RecordingDriver()
    assert count_missing_embeddings(driver, database=None, chunk_uids=[]) == 0
    assert count_orphan_chunks(driver, database=None, chunk_uids=[]) == 0
    assert count_checksum_mismatches(driver, database=None, chunk_rows=[]) == 0


def test_count_helpers_extract_values():
    driver = RecordingDriver(responses=[([{"value": 3}],), ([{"value": 1}],)])
    missing = count_missing_embeddings(driver, database="neo4j", chunk_uids=["chunk-1"])
    orphans = count_orphan_chunks(driver, database="neo4j", chunk_uids=["chunk-1"])
    assert missing == 3
    assert orphans == 1


def test_count_checksum_mismatches_uses_rows_payload():
    driver = RecordingDriver(responses=[[{"value": 4}]])
    rows = [{"uid": "chunk-1", "checksum": "deadbeef"}]
    assert count_checksum_mismatches(driver, database="neo4j", chunk_rows=rows) == 4


def test_collect_semantic_counts_runs_queries():
    driver = RecordingDriver(
        responses=[[{"value": 6}], ([{"value": 2}],), ([{"value": 1}],)]
    )
    counts = collect_semantic_counts(driver, database=None, source_tag="semantic")
    assert counts == {
        "nodes_in_db": 6,
        "relationships_in_db": 2,
        "orphan_entities": 1,
    }


@pytest.mark.parametrize("chunk_uids", [["chunk-1"], ["chunk-1", "chunk-2"]])
def test_rollback_ignores_missing_run_keys(chunk_uids):
    driver = RecordingDriver()
    sources = [
        SimpleNamespace(
            chunks=[SimpleNamespace(uid=uid) for uid in chunk_uids],
            relative_path="docs/file.md",
            ingest_run_key=None,
        )
    ]
    rollback_ingest(driver, database=None, sources=sources)
    # Without run keys, only chunk and document cleanup queries run
    assert len(driver.calls) == 2
