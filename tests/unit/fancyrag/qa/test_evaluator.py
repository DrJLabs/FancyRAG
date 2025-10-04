from __future__ import annotations

from types import SimpleNamespace

from fancyrag.qa.evaluator import (
    IngestionQaEvaluator,
    QaChunkRecord,
    QaSourceRecord,
    QaThresholds,
    collect_counts,
)


class SequencedDriver:
    """Driver stub that returns pre-seeded query results in order."""

    def __init__(self, results: list[list[dict[str, int]]]):
        self._results = [list(result) for result in results]
        self.calls: list[str] = []

    def execute_query(self, query, parameters=None, database_=None):  # noqa: D401 - mimics Neo4j driver
        if not self._results:
            raise AssertionError("Driver received more queries than configured")
        self.calls.append(" ".join(str(query).split()))
        payload = self._results.pop(0)
        return payload


def _chunk(text: str = "hello world", uid: str = "chunk-1") -> QaChunkRecord:
    return QaChunkRecord(uid=uid, checksum="deadbeef", text=text)


def test_evaluator_pass_generates_reports(tmp_path):
    driver = SequencedDriver(
        results=[
            [{"value": 2}],  # documents
            [{"value": 4}],  # chunks
            [{"value": 4}],  # relationships
            [{"value": 0}],  # missing embeddings
            [{"value": 0}],  # orphan chunks
            [{"value": 0}],  # checksum mismatches
        ]
    )
    sources = [
        QaSourceRecord(
            path="/src/input.txt",
            relative_path="docs/input.txt",
            document_checksum="abc123",
            git_commit="commit-sha",
            chunks=[_chunk()],
            ingest_run_key=None,
        )
    ]
    thresholds = QaThresholds(
        max_missing_embeddings=1,
        max_orphan_chunks=1,
        max_checksum_mismatches=1,
    )

    report_root = tmp_path / "qa"
    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=thresholds,
        report_root=report_root,
        report_version="v1",
    )

    result = evaluator.evaluate()

    assert result.passed is True
    assert result.metrics["graph_counts"] == {"documents": 2, "chunks": 4, "relationships": 4}
    assert result.metrics["files"] == [
        {
            "relative_path": "docs/input.txt",
            "git_commit": "commit-sha",
            "document_checksum": "abc123",
            "chunks": 1,
        }
    ]
    qa_dir = next(report_root.iterdir())
    assert (qa_dir / "quality_report.json").exists()
    assert (qa_dir / "quality_report.md").exists()
    assert not driver._results


def test_evaluator_flags_threshold_breaches(tmp_path):
    driver = SequencedDriver(
        results=[
            [{"value": 1}],
            [{"value": 1}],
            [{"value": 1}],
            [{"value": 5}],  # missing embeddings
            [{"value": 2}],  # orphan chunks
            [{"value": 3}],  # checksum mismatches
            [{"value": 7}],  # semantic nodes
            [{"value": 8}],  # semantic relationships
            [{"value": 4}],  # semantic orphan entities
        ]
    )
    sources = [
        QaSourceRecord(
            path="/src/input.txt",
            relative_path="docs/input.txt",
            document_checksum="abc123",
            git_commit=None,
            chunks=[_chunk(), _chunk(uid="chunk-2")],
            ingest_run_key="run-1",
        )
    ]
    thresholds = QaThresholds()
    semantic_summary = SimpleNamespace(
        enabled=True,
        chunks_processed=10,
        chunk_failures=2,
        nodes_written=5,
        relationships_written=6,
        source_tag="semantic.tag",
    )

    report_root = tmp_path / "qa"
    evaluator = IngestionQaEvaluator(
        driver=driver,
        database=None,
        sources=sources,
        thresholds=thresholds,
        report_root=report_root,
        report_version="v1",
        semantic_summary=semantic_summary,
    )

    result = evaluator.evaluate()

    assert result.passed is False
    assert any("missing_embeddings=5" in reason for reason in result.anomalies)
    assert any("orphan_chunks=2" in reason for reason in result.anomalies)
    assert any("checksum_mismatches=3" in reason for reason in result.anomalies)
    assert result.metrics["semantic"] == {
        "chunks_processed": 10,
        "chunk_failures": 2,
        "nodes_written": 5,
        "relationships_written": 6,
        "nodes_in_db": 7,
        "relationships_in_db": 8,
        "orphan_entities": 4,
    }
    assert not driver._results


def test_collect_counts_handles_tuple_result():
    class DummyDriver:
        def execute_query(self, query, database_=None):
            if "HAS_CHUNK" in query:
                return ([{"value": 8}],)
            if "Document" in query:
                return ([{"value": 5}],)
            if "Chunk" in query:
                return ([{"value": 12}],)
            return ([{"value": 0}],)

    counts = collect_counts(DummyDriver(), database="neo4j")
    assert counts == {"documents": 5, "chunks": 12, "relationships": 8}
