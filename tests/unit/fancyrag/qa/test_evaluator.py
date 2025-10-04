from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest

from fancyrag.db.neo4j_queries import collect_counts
from fancyrag.qa.evaluator import (
    IngestionQaEvaluator,
    QaChunkRecord,
    QaSourceRecord,
    QaThresholds,
    QaResult,
)


class SequencedDriver:
    """Driver stub that returns pre-seeded query results in order."""

    def __init__(self, results: list[list[dict[str, int]]]):
        self._results = [list(result) for result in results]
        self.calls: list[str] = []

    def execute_query(self, query, parameters=None, database_=None):  # mimics Neo4j driver
        _ = parameters
        _ = database_
        if not self._results:
            raise AssertionError
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
        def execute_query(self, query, parameters=None, database_=None):
            _ = parameters
            _ = database_
            if "HAS_CHUNK" in query:
                return ([{"value": 8}],)
            if "Document" in query:
                return ([{"value": 5}],)
            if "Chunk" in query:
                return ([{"value": 12}],)
            return ([{"value": 0}],)

    counts = collect_counts(DummyDriver(), database="neo4j")
    assert counts == {"documents": 5, "chunks": 12, "relationships": 8}


def test_token_histogram_bucketing(tmp_path):
    # Instantiate evaluator; we won't call evaluate(), only histogram helpers.
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )
    counts = evaluator._token_histogram([0, 1, 64, 65, 129, 2048, 5000])
    # Expected buckets based on TOKEN_BINS = (64, 128, 256, 512, 1024, 2048)

    assert counts["<= 64"] == 3     # 0, 1, 64
    assert counts["<= 128"] == 1    # 65
    assert counts["<= 256"] == 1    # 129
    assert counts["<= 512"] == 0
    assert counts["<= 1024"] == 0
    assert counts["<= 2048"] == 1   # 2048
    assert counts["> 2048"] == 1    # 5000


def test_estimate_tokens_minimum_and_rounding():
    # Empty -> 0; non-empty minimum 1; 5 chars -> ceil(5/4)=2
    assert IngestionQaEvaluator._estimate_tokens("") == 0
    assert IngestionQaEvaluator._estimate_tokens("a") == 1
    assert IngestionQaEvaluator._estimate_tokens("abcd") == 1
    assert IngestionQaEvaluator._estimate_tokens("abcde") == 2


def test_render_markdown_includes_histogram_and_files(tmp_path):
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],
        thresholds=QaThresholds(max_missing_embeddings=0),
        report_root=tmp_path,
        report_version="v1",
    )
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "All good",
        "metrics": {
            "files_processed": 1,
            "chunks_processed": 2,
            "token_estimate": {
                "total": 3,
                "mean": 1.5,
                "histogram": {"<= 64": 2, "> 2048": 0},
            },
            "missing_embeddings": 0,
            "orphan_chunks": 0,
            "checksum_mismatches": 0,
            "files": [
                {
                    "relative_path": "docs/input.txt",
                    "chunks": 2,
                    "document_checksum": "abc",
                    "git_commit": "sha",
                }
            ],
        },
        "thresholds": {"max_missing_embeddings": 0},
        "anomalies": [],
    }
    md = evaluator._render_markdown(payload)

    assert "# Ingestion QA Report (v1)" in md
    assert "Token Histogram" in md
    assert "| <= 64 | 2 |" in md
    assert "## Files" in md
    assert "docs/input.txt" in md
    assert "abc" in md
    assert "sha" in md
    assert "All thresholds satisfied" in md


def test_evaluate_report_paths_are_relative(tmp_path):
    # Driver stub: return a single [{value: 0}] row for any query
    class NoOpDriver:
        def execute_query(self, _q, parameters=None, database_=None):
            _ = parameters
            _ = database_
            return [{"value": 0}]

    sources = [
        QaSourceRecord(
            path="/abs/sample.txt",
            relative_path="docs/sample.txt",
            document_checksum="deadbeef",
            git_commit=None,
            chunks=[QaChunkRecord(uid="u1", checksum="c1", text="hello world")],
            ingest_run_key=None,
        )
    ]
    evaluator = IngestionQaEvaluator(
        driver=NoOpDriver(),
        database=None,
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path / "qa",
        report_version="v1",
    )
    result = evaluator.evaluate()
    json_path = Path(result.report_json)
    md_path = Path(result.report_markdown)

    if not json_path.is_absolute():
        json_path = (tmp_path / "qa") / json_path
    if not md_path.is_absolute():
        md_path = (tmp_path / "qa") / md_path

    assert json_path.exists()
    assert md_path.exists()


# --- Additional comprehensive tests for evaluator.py ---


def test_collect_counts_handles_neo4j_error():
    """Verify collect_counts logs warning and continues when Neo4j query fails."""
    class FailingDriver:
        def execute_query(self, query, parameters=None, database_=None):
            _ = query
            _ = parameters
            _ = database_
            from neo4j.exceptions import Neo4jError
            raise Neo4jError()

    counts = collect_counts(FailingDriver(), database="neo4j")
    assert counts == {"documents": 0, "chunks": 0, "relationships": 0}


def test_collect_counts_handles_various_record_formats():
    """Test collect_counts with different record response formats."""
    class VariedFormatDriver:
        def __init__(self):
            self.call_count = 0

        def execute_query(self, query, parameters=None, database_=None):
            _ = database_
            _ = parameters
            self.call_count += 1
            # Test different response formats
            if "HAS_CHUNK" in query:
                # Format 3: Tuple/list access
                return ([[15]], None, None)
            if "Document" in query:
                # Format 1: Dictionary with value key
                return ([{"value": 10}], None, None)
            elif "Chunk" in query and "HAS_CHUNK" not in query:
                # Format 2: Object with value attribute
                from types import SimpleNamespace
                return ([SimpleNamespace(value=20)], None, None)
            return ([{"value": 0}], None, None)

    driver = VariedFormatDriver()
    counts = collect_counts(driver, database="neo4j")
    assert counts["documents"] == 10
    assert counts["chunks"] == 20
    assert counts["relationships"] == 15


def test_collect_counts_with_empty_results():
    """Test collect_counts when queries return empty result sets."""
    class EmptyResultDriver:
        def execute_query(self, query, parameters=None, database_=None):
            _ = query
            _ = parameters
            _ = database_
            return ([], None, None)

    counts = collect_counts(EmptyResultDriver(), database=None)
    assert counts == {"documents": 0, "chunks": 0, "relationships": 0}


def test_collect_counts_with_none_values():
    """Test collect_counts when query returns None values."""
    class NoneValueDriver:
        def execute_query(self, query, parameters=None, database_=None):
            _ = query
            _ = parameters
            _ = database_
            return ([{"value": None}], None, None)

    counts = collect_counts(NoneValueDriver(), database="test")
    assert counts.get("documents", 0) == 0
    assert counts.get("chunks", 0) == 0
    assert counts.get("relationships", 0) == 0


def test_evaluator_with_empty_sources_list(tmp_path):
    """Verify evaluator handles empty sources list gracefully."""
    driver = SequencedDriver(
        results=[
            [{"value": 0}],  # documents
            [{"value": 0}],  # chunks
            [{"value": 0}],  # relationships
        ]
    )

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=[],  # Empty sources
        thresholds=QaThresholds(),
        report_root=tmp_path / "qa",
        report_version="v1",
    )

    result = evaluator.evaluate()
    assert result.passed is True
    assert result.metrics["files_processed"] == 0
    assert result.metrics["chunks_processed"] == 0
    assert result.metrics["missing_embeddings"] == 0


def test_evaluator_with_single_chunk(tmp_path):
    """Test evaluator with minimal single-chunk scenario."""
    driver = SequencedDriver(
        results=[
            [{"value": 1}],  # documents
            [{"value": 1}],  # chunks
            [{"value": 1}],  # relationships
            [{"value": 0}],  # missing embeddings
            [{"value": 0}],  # orphan chunks
            [{"value": 0}],  # checksum mismatches
        ]
    )

    sources = [
        QaSourceRecord(
            path="/src/tiny.txt",
            relative_path="tiny.txt",
            document_checksum="abc",
            git_commit=None,
            chunks=[QaChunkRecord(uid="c1", checksum="x", text="hi")],
        )
    ]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database=None,
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path / "qa",
        report_version="v1",
    )

    result = evaluator.evaluate()
    assert result.passed is True
    assert result.metrics["files_processed"] == 1
    assert result.metrics["chunks_processed"] == 1


def test_evaluator_all_thresholds_exceeded(tmp_path):
    """Test scenario where all QA thresholds are exceeded simultaneously."""
    driver = SequencedDriver(
        results=[
            [{"value": 10}],
            [{"value": 20}],
            [{"value": 15}],
            [{"value": 5}],  # missing embeddings exceeds threshold
            [{"value": 3}],  # orphan chunks exceeds threshold
            [{"value": 2}],  # checksum mismatches exceeds threshold
        ]
    )

    sources = [
        QaSourceRecord(
            path="/src/test.txt",
            relative_path="test.txt",
            document_checksum="abc",
            git_commit="sha1",
            chunks=[_chunk(), _chunk(uid="c2")],
        )
    ]

    thresholds = QaThresholds(
        max_missing_embeddings=0,
        max_orphan_chunks=0,
        max_checksum_mismatches=0,
    )

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=thresholds,
        report_root=tmp_path / "qa",
        report_version="v1",
    )

    result = evaluator.evaluate()
    assert result.passed is False
    assert len(result.anomalies) == 3
    assert any("missing_embeddings=5" in a for a in result.anomalies)
    assert any("orphan_chunks=3" in a for a in result.anomalies)
    assert any("checksum_mismatches=2" in a for a in result.anomalies)


def test_evaluator_semantic_all_thresholds_exceeded(tmp_path):
    """Test semantic enrichment with all thresholds exceeded."""
    from types import SimpleNamespace

    driver = SequencedDriver(
        results=[
            [{"value": 5}],  # documents
            [{"value": 10}],  # chunks
            [{"value": 8}],  # relationships
            [{"value": 0}],  # missing embeddings
            [{"value": 0}],  # orphan chunks
            [{"value": 0}],  # checksum mismatches
            [{"value": 100}],  # semantic nodes
            [{"value": 50}],  # semantic relationships
            [{"value": 15}],  # semantic orphan entities - exceeds threshold
        ]
    )

    semantic_summary = SimpleNamespace(
        enabled=True,
        chunks_processed=10,
        chunk_failures=5,  # Exceeds threshold
        nodes_written=100,
        relationships_written=50,
        source_tag="semantic.test",
    )

    thresholds = QaThresholds(
        max_semantic_failures=2,
        max_semantic_orphans=10,
    )

    sources = [
        QaSourceRecord(
            path="/src/test.txt",
            relative_path="test.txt",
            document_checksum="abc",
            git_commit=None,
            chunks=[_chunk()],
        )
    ]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=thresholds,
        report_root=tmp_path / "qa",
        report_version="v1",
        semantic_summary=semantic_summary,
    )

    result = evaluator.evaluate()
    assert result.passed is False
    assert any("semantic_chunk_failures=5" in a for a in result.anomalies)
    assert any("semantic_orphan_entities=15" in a for a in result.anomalies)
    assert result.metrics["semantic"]["chunk_failures"] == 5
    assert result.metrics["semantic"]["orphan_entities"] == 15


def test_evaluator_semantic_disabled(tmp_path):
    """Verify semantic metrics are not collected when semantic enrichment is disabled."""
    from types import SimpleNamespace

    driver = SequencedDriver(
        results=[
            [{"value": 1}],
            [{"value": 2}],
            [{"value": 2}],
            [{"value": 0}],
            [{"value": 0}],
            [{"value": 0}],
        ]
    )

    semantic_summary = SimpleNamespace(
        enabled=False,  # Disabled
        chunks_processed=0,
        chunk_failures=0,
        nodes_written=0,
        relationships_written=0,
        source_tag="semantic.disabled",
    )

    sources = [_chunk_record()]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database=None,
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path / "qa",
        report_version="v1",
        semantic_summary=semantic_summary,
    )

    result = evaluator.evaluate()
    assert "semantic" not in result.metrics


def test_token_histogram_empty_list(tmp_path):
    """Test histogram generation with empty token list."""
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    histogram = evaluator._token_histogram([])
    assert all(count == 0 for count in histogram.values())
    assert "<= 64" in histogram
    assert "> 2048" in histogram


def test_token_histogram_single_value(tmp_path):
    """Test histogram with a single token count."""
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    histogram = evaluator._token_histogram([100])
    assert histogram["<= 128"] == 1
    assert sum(histogram.values()) == 1


def test_token_histogram_boundary_values(tmp_path):
    """Test histogram bucketing at exact boundary values."""
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    # Test exact boundary values from TOKEN_BINS
    counts = [64, 128, 256, 512, 1024, 2048]
    histogram = evaluator._token_histogram(counts)

    assert histogram["<= 64"] == 1
    assert histogram["<= 128"] == 1
    assert histogram["<= 256"] == 1
    assert histogram["<= 512"] == 1
    assert histogram["<= 1024"] == 1
    assert histogram["<= 2048"] == 1
    assert histogram["> 2048"] == 0


def test_token_histogram_large_values(tmp_path):
    """Test histogram with very large token counts."""
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    counts = [10000, 50000, 100000]
    histogram = evaluator._token_histogram(counts)
    assert histogram["> 2048"] == 3


def test_bucket_label_formatting():
    """Test bucket label generation for various scenarios."""
    assert IngestionQaEvaluator._bucket_label(None, 64) == "<= 64"
    assert IngestionQaEvaluator._bucket_label(2048, None) == "> 2048"
    assert IngestionQaEvaluator._bucket_label(None, None) == "unknown"
    assert IngestionQaEvaluator._bucket_label(64, 128) == "65-128"
    assert IngestionQaEvaluator._bucket_label(0, 10) == "1-10"


def test_estimate_tokens_edge_cases():
    """Test token estimation with edge cases."""
    # Empty string
    assert IngestionQaEvaluator._estimate_tokens("") == 0

    # Single character (minimum is 1)
    assert IngestionQaEvaluator._estimate_tokens("x") == 1

    # Exactly 4 characters
    assert IngestionQaEvaluator._estimate_tokens("abcd") == 1

    # 5 characters - should ceil(5/4) = 2
    assert IngestionQaEvaluator._estimate_tokens("abcde") == 2

    # Very long text
    long_text = "a" * 10000
    assert IngestionQaEvaluator._estimate_tokens(long_text) == 2500

    # Unicode characters
    assert IngestionQaEvaluator._estimate_tokens("🔥🔥🔥") > 0


def test_compute_totals_empty_chunks(tmp_path):
    """Test _compute_totals with no chunks."""
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    totals = evaluator._compute_totals()
    assert totals["files_processed"] == 0
    assert totals["chunks_processed"] == 0
    assert totals["token_estimate"]["total"] == 0
    assert totals["token_estimate"]["max"] == 0
    assert totals["token_estimate"]["mean"] == 0.0
    assert totals["char_lengths"]["total"] == 0
    assert totals["char_lengths"]["max"] == 0
    assert totals["char_lengths"]["mean"] == 0.0


def test_compute_totals_single_chunk(tmp_path):
    """Test _compute_totals with a single chunk."""
    sources = [
        QaSourceRecord(
            path="/test.txt",
            relative_path="test.txt",
            document_checksum="abc",
            git_commit=None,
            chunks=[QaChunkRecord(uid="c1", checksum="x", text="hello world")],
        )
    ]

    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    totals = evaluator._compute_totals()
    assert totals["files_processed"] == 1
    assert totals["chunks_processed"] == 1
    assert totals["char_lengths"]["total"] == 11  # "hello world"
    assert totals["token_estimate"]["mean"] > 0


def test_compute_totals_multiple_files_multiple_chunks(tmp_path):
    """Test _compute_totals with multiple files and chunks."""
    sources = [
        QaSourceRecord(
            path="/file1.txt",
            relative_path="file1.txt",
            document_checksum="abc",
            git_commit=None,
            chunks=[
                QaChunkRecord(uid="c1", checksum="x", text="a" * 100),
                QaChunkRecord(uid="c2", checksum="y", text="b" * 200),
            ],
        ),
        QaSourceRecord(
            path="/file2.txt",
            relative_path="file2.txt",
            document_checksum="def",
            git_commit=None,
            chunks=[
                QaChunkRecord(uid="c3", checksum="z", text="c" * 50),
            ],
        ),
    ]

    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    totals = evaluator._compute_totals()
    assert totals["files_processed"] == 2
    assert totals["chunks_processed"] == 3
    assert totals["char_lengths"]["total"] == 350
    assert totals["char_lengths"]["max"] == 200
    assert totals["token_estimate"]["max"] > 0


def test_qa_result_dataclass_attributes():
    """Test QaResult dataclass has all expected attributes."""
    from datetime import datetime, timezone

    result = QaResult(
        status="pass",
        summary="Test summary",
        metrics={"test": 1},
        anomalies=[],
        thresholds=QaThresholds(),
        report_json="report.json",
        report_markdown="report.md",
        timestamp=datetime.now(timezone.utc),
        version="v1",
        duration_ms=100,
    )

    assert result.status == "pass"
    assert result.summary == "Test summary"
    assert result.metrics == {"test": 1}
    assert result.anomalies == []
    assert isinstance(result.thresholds, QaThresholds)
    assert result.report_json == "report.json"
    assert result.report_markdown == "report.md"
    assert isinstance(result.timestamp, datetime)
    assert result.version == "v1"
    assert result.duration_ms == 100


def test_qa_result_passed_property():
    """Test QaResult.passed property logic."""
    from datetime import datetime, timezone

    passing = QaResult(
        status="pass",
        summary="",
        metrics={},
        anomalies=[],
        thresholds=QaThresholds(),
        report_json="",
        report_markdown="",
        timestamp=datetime.now(timezone.utc),
        version="v1",
        duration_ms=0,
    )
    assert passing.passed is True

    failing = QaResult(
        status="fail",
        summary="",
        metrics={},
        anomalies=["error"],
        thresholds=QaThresholds(),
        report_json="",
        report_markdown="",
        timestamp=datetime.now(timezone.utc),
        version="v1",
        duration_ms=0,
    )
    assert failing.passed is False

    unknown = QaResult(
        status="unknown",
        summary="",
        metrics={},
        anomalies=[],
        thresholds=QaThresholds(),
        report_json="",
        report_markdown="",
        timestamp=datetime.now(timezone.utc),
        version="v1",
        duration_ms=0,
    )
    assert unknown.passed is False


def test_qa_thresholds_default_values():
    """Test QaThresholds has correct default values."""
    thresholds = QaThresholds()
    assert thresholds.max_missing_embeddings == 0
    assert thresholds.max_orphan_chunks == 0
    assert thresholds.max_checksum_mismatches == 0
    assert thresholds.max_semantic_failures == 0
    assert thresholds.max_semantic_orphans == 0


def test_qa_thresholds_custom_values():
    """Test QaThresholds accepts custom values."""
    thresholds = QaThresholds(
        max_missing_embeddings=10,
        max_orphan_chunks=5,
        max_checksum_mismatches=3,
        max_semantic_failures=2,
        max_semantic_orphans=1,
    )
    assert thresholds.max_missing_embeddings == 10
    assert thresholds.max_orphan_chunks == 5
    assert thresholds.max_checksum_mismatches == 3
    assert thresholds.max_semantic_failures == 2
    assert thresholds.max_semantic_orphans == 1


def test_qa_chunk_record_attributes():
    """Test QaChunkRecord dataclass attributes."""
    chunk = QaChunkRecord(uid="test-uid", checksum="abc123", text="test content")
    assert chunk.uid == "test-uid"
    assert chunk.checksum == "abc123"
    assert chunk.text == "test content"


def test_qa_source_record_attributes():
    """Test QaSourceRecord dataclass attributes."""
    chunks = [QaChunkRecord(uid="c1", checksum="x", text="content")]
    source = QaSourceRecord(
        path="/abs/path.txt",
        relative_path="path.txt",
        document_checksum="doc123",
        git_commit="sha1",
        chunks=chunks,
        ingest_run_key="run-key",
    )
    assert source.path == "/abs/path.txt"
    assert source.relative_path == "path.txt"
    assert source.document_checksum == "doc123"
    assert source.git_commit == "sha1"
    assert len(source.chunks) == 1
    assert source.ingest_run_key == "run-key"


def test_qa_source_record_optional_fields():
    """Test QaSourceRecord with optional fields as None."""
    source = QaSourceRecord(
        path="/path.txt",
        relative_path="path.txt",
        document_checksum="abc",
        git_commit=None,
        chunks=[],
        ingest_run_key=None,
    )
    assert source.git_commit is None
    assert source.ingest_run_key is None
    assert source.chunks == []


def test_evaluator_duration_tracking(tmp_path):
    """Verify evaluator tracks execution duration."""
    driver = SequencedDriver(
        results=[
            [{"value": 0}],
            [{"value": 0}],
            [{"value": 0}],
            [{"value": 0}],
            [{"value": 0}],
            [{"value": 0}],
        ]
    )

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database=None,
        sources=[_chunk_record()],
        thresholds=QaThresholds(),
        report_root=tmp_path / "qa",
        report_version="v1",
    )

    result = evaluator.evaluate()
    assert result.duration_ms >= 0
    assert "qa_evaluation_ms" in result.metrics
    assert result.metrics["qa_evaluation_ms"] >= 0


def test_collect_semantic_counts_when_disabled(tmp_path):
    """Test _collect_semantic_counts returns empty dict when disabled."""
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
        semantic_summary=None,
    )

    counts = evaluator._collect_semantic_counts()
    assert counts == {}


def test_collect_semantic_counts_queries_database(tmp_path):
    """Test _collect_semantic_counts executes correct queries."""
    from types import SimpleNamespace

    class CountingDriver:
        def __init__(self):
            self.queries = []

        def execute_query(self, query, parameters, database_=None):
            _ = parameters
            _ = database_
            self.queries.append(query)
            if "AND NOT" in query:
                return ([{"value": 5}], None, None)
            elif ")-[r]->" in query:
                return ([{"value": 10}], None, None)
            else:
                return ([{"value": 15}], None, None)

    driver = CountingDriver()
    semantic_summary = SimpleNamespace(
        enabled=True,
        chunks_processed=0,
        chunk_failures=0,
        nodes_written=0,
        relationships_written=0,
        source_tag="test.tag",
    )

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=[_chunk_record()],
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
        semantic_summary=semantic_summary,
    )

    counts = evaluator._collect_semantic_counts()
    assert len(driver.queries) == 3
    assert counts["nodes_in_db"] >= 0
    assert counts["relationships_in_db"] >= 0
    assert counts["orphan_entities"] >= 0


# Helper functions for tests
def _chunk_record():
    """Helper to create a minimal QaSourceRecord with one chunk."""
    return QaSourceRecord(
        path="/test.txt",
        relative_path="test.txt",
        document_checksum="abc",
        git_commit=None,
        chunks=[QaChunkRecord(uid="c1", checksum="x", text="test")],
    )


def test_evaluator_with_no_sources(tmp_path):
    """Test evaluator handles empty sources list."""
    driver = SequencedDriver(
        results=[
            [{"value": 0}],  # documents
            [{"value": 0}],  # chunks
            [{"value": 0}],  # relationships
        ]
    )

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=[],
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    result = evaluator.evaluate()
    assert result.passed is True
    assert result.metrics["files_processed"] == 0
    assert result.metrics["chunks_processed"] == 0


def test_evaluator_computes_token_statistics(tmp_path):
    """Test evaluator computes token estimation statistics."""
    driver = SequencedDriver(
        results=[
            [{"value": 1}], [{"value": 3}], [{"value": 3}],
            [{"value": 0}], [{"value": 0}], [{"value": 0}],
        ]
    )

    sources = [
        QaSourceRecord(
            path="/src/file.txt",
            relative_path="file.txt",
            document_checksum="abc",
            git_commit=None,
            chunks=[
                _chunk(text="a" * 100, uid="chunk-1"),  # 25 tokens
                _chunk(text="b" * 200, uid="chunk-2"),  # 50 tokens
                _chunk(text="c" * 400, uid="chunk-3"),  # 100 tokens
            ],
        )
    ]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    result = evaluator.evaluate()

    token_stats = result.metrics["token_estimate"]
    assert token_stats["total"] == 175  # 25 + 50 + 100
    assert token_stats["max"] == 100
    assert token_stats["mean"] == pytest.approx(58.33, rel=0.01)
    assert "histogram" in token_stats


def test_evaluator_computes_char_length_statistics(tmp_path):
    """Test evaluator computes character length statistics."""
    driver = SequencedDriver(
        results=[
            [{"value": 1}], [{"value": 2}], [{"value": 2}],
            [{"value": 0}], [{"value": 0}], [{"value": 0}],
        ]
    )

    sources = [
        QaSourceRecord(
            path="/src/file.txt",
            relative_path="file.txt",
            document_checksum="abc",
            git_commit=None,
            chunks=[
                _chunk(text="x" * 50, uid="chunk-1"),
                _chunk(text="y" * 150, uid="chunk-2"),
            ],
        )
    ]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    result = evaluator.evaluate()

    char_stats = result.metrics["char_lengths"]
    assert char_stats["total"] == 200
    assert char_stats["max"] == 150
    assert char_stats["mean"] == 100.0


def test_evaluator_includes_semantic_metrics_when_enabled(tmp_path):
    """Test evaluator includes semantic metrics when enrichment enabled."""
    driver = SequencedDriver(
        results=[
            [{"value": 1}], [{"value": 1}], [{"value": 1}],
            [{"value": 0}], [{"value": 0}], [{"value": 0}],
            [{"value": 10}],  # semantic nodes
            [{"value": 15}],  # semantic relationships
            [{"value": 2}],   # semantic orphans
        ]
    )

    semantic_summary = SimpleNamespace(
        enabled=True,
        chunks_processed=5,
        chunk_failures=1,
        nodes_written=8,
        relationships_written=12,
        source_tag="test.tag",
    )

    sources = [
        QaSourceRecord(
            path="/src/file.txt",
            relative_path="file.txt",
            document_checksum="abc",
            git_commit=None,
            chunks=[_chunk()],
        )
    ]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
        semantic_summary=semantic_summary,
    )

    result = evaluator.evaluate()

    assert "semantic" in result.metrics
    sem = result.metrics["semantic"]
    assert sem["chunks_processed"] == 5
    assert sem["chunk_failures"] == 1
    assert sem["nodes_written"] == 8
    assert sem["relationships_written"] == 12
    assert sem["nodes_in_db"] == 10
    assert sem["relationships_in_db"] == 15
    assert sem["orphan_entities"] == 2


def test_evaluator_flags_semantic_chunk_failures(tmp_path):
    """Test evaluator flags excessive semantic chunk failures."""
    driver = SequencedDriver(
        results=[
            [{"value": 1}], [{"value": 1}], [{"value": 1}],
            [{"value": 0}], [{"value": 0}], [{"value": 0}],
            [{"value": 5}], [{"value": 5}], [{"value": 0}],
        ]
    )

    semantic_summary = SimpleNamespace(
        enabled=True,
        chunks_processed=10,
        chunk_failures=5,  # Exceeds threshold
        nodes_written=5,
        relationships_written=5,
        source_tag="test",
    )

    sources = [QaSourceRecord(
        path="/file.txt",
        relative_path="file.txt",
        document_checksum="abc",
        git_commit=None,
        chunks=[_chunk()],
    )]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=QaThresholds(max_semantic_failures=3),
        report_root=tmp_path,
        report_version="v1",
        semantic_summary=semantic_summary,
    )

    result = evaluator.evaluate()

    assert result.passed is False
    assert any("semantic_chunk_failures=5" in a for a in result.anomalies)


def test_evaluator_flags_semantic_orphan_entities(tmp_path):
    """Test evaluator flags excessive semantic orphan entities."""
    driver = SequencedDriver(
        results=[
            [{"value": 1}], [{"value": 1}], [{"value": 1}],
            [{"value": 0}], [{"value": 0}], [{"value": 0}],
            [{"value": 10}], [{"value": 10}], [{"value": 5}],  # 5 orphans
        ]
    )

    semantic_summary = SimpleNamespace(
        enabled=True,
        chunks_processed=10,
        chunk_failures=0,
        nodes_written=10,
        relationships_written=10,
        source_tag="test",
    )

    sources = [QaSourceRecord(
        path="/file.txt",
        relative_path="file.txt",
        document_checksum="abc",
        git_commit=None,
        chunks=[_chunk()],
    )]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=QaThresholds(max_semantic_orphans=2),
        report_root=tmp_path,
        report_version="v1",
        semantic_summary=semantic_summary,
    )

    result = evaluator.evaluate()

    assert result.passed is False
    assert any("semantic_orphan_entities=5" in a for a in result.anomalies)


def test_evaluator_per_file_metrics(tmp_path):
    """Test evaluator includes per-file breakdown."""
    driver = SequencedDriver(
        results=[
            [{"value": 2}], [{"value": 5}], [{"value": 5}],
            [{"value": 0}], [{"value": 0}], [{"value": 0}],
        ]
    )

    sources = [
        QaSourceRecord(
            path="/src/file1.txt",
            relative_path="file1.txt",
            document_checksum="abc123",
            git_commit="commit1",
            chunks=[_chunk(uid="c1"), _chunk(uid="c2")],
        ),
        QaSourceRecord(
            path="/src/file2.txt",
            relative_path="file2.txt",
            document_checksum="def456",
            git_commit="commit2",
            chunks=[_chunk(uid="c3"), _chunk(uid="c4"), _chunk(uid="c5")],
        ),
    ]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    result = evaluator.evaluate()

    files = result.metrics["files"]
    assert len(files) == 2
    assert files[0]["relative_path"] == "file1.txt"
    assert files[0]["git_commit"] == "commit1"
    assert files[0]["document_checksum"] == "abc123"
    assert files[0]["chunks"] == 2
    assert files[1]["relative_path"] == "file2.txt"
    assert files[1]["chunks"] == 3


def test_evaluator_creates_timestamped_report_directory(tmp_path):
    """Test evaluator creates timestamped subdirectory for reports."""
    driver = SequencedDriver(
        results=[
            [{"value": 1}], [{"value": 1}], [{"value": 1}],
            [{"value": 0}], [{"value": 0}], [{"value": 0}],
        ]
    )

    sources = [QaSourceRecord(
        path="/file.txt",
        relative_path="file.txt",
        document_checksum="abc",
        git_commit=None,
        chunks=[_chunk()],
    )]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path / "qa",
        report_version="v1",
    )

    evaluator.evaluate()

    # Should create a timestamped subdirectory
    qa_dirs = list((tmp_path / "qa").iterdir())
    assert len(qa_dirs) == 1
    assert qa_dirs[0].is_dir()

    # Should contain both reports
    assert (qa_dirs[0] / "quality_report.json").exists()
    assert (qa_dirs[0] / "quality_report.md").exists()


def test_evaluator_summary_string_format(tmp_path):
    """Test evaluator creates properly formatted summary string."""
    driver = SequencedDriver(
        results=[
            [{"value": 1}], [{"value": 1}], [{"value": 1}],
            [{"value": 2}], [{"value": 3}], [{"value": 1}],
        ]
    )

    sources = [QaSourceRecord(
        path="/file.txt",
        relative_path="file.txt",
        document_checksum="abc",
        git_commit=None,
        chunks=[_chunk()],
    )]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=QaThresholds(max_missing_embeddings=5),
        report_root=tmp_path,
        report_version="v1",
    )

    result = evaluator.evaluate()

    assert "QA PASS:" in result.summary
    assert "missing_embeddings=2" in result.summary
    assert "orphan_chunks=3" in result.summary
    assert "checksum_mismatches=1" in result.summary


def test_estimate_tokens_empty_string():
    """Test _estimate_tokens returns 0 for empty string."""
    assert IngestionQaEvaluator._estimate_tokens("") == 0


def test_estimate_tokens_minimum_one():
    """Test _estimate_tokens returns minimum 1 for non-empty."""
    assert IngestionQaEvaluator._estimate_tokens("a") == 1
    assert IngestionQaEvaluator._estimate_tokens("abc") == 1


def test_estimate_tokens_rounding():
    """Test _estimate_tokens rounds up character count / 4."""
    # 4 chars = 1 token
    assert IngestionQaEvaluator._estimate_tokens("abcd") == 1
    # 5 chars = 2 tokens (rounded up)
    assert IngestionQaEvaluator._estimate_tokens("abcde") == 2
    # 8 chars = 2 tokens
    assert IngestionQaEvaluator._estimate_tokens("a" * 8) == 2
    # 9 chars = 3 tokens (rounded up)
    assert IngestionQaEvaluator._estimate_tokens("a" * 9) == 3


def test_bucket_label_formatting():
    """Test _bucket_label creates proper labels."""
    assert IngestionQaEvaluator._bucket_label(None, 64) == "<= 64"
    assert IngestionQaEvaluator._bucket_label(2048, None) == "> 2048"
    assert IngestionQaEvaluator._bucket_label(64, 128) == "65-128"
    assert IngestionQaEvaluator._bucket_label(None, None) == "unknown"


def test_token_histogram_empty_list():
    """Test _token_histogram handles empty token list."""
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],
        thresholds=QaThresholds(),
        report_root=Path("/tmp"),
        report_version="v1",
    )

    histogram = evaluator._token_histogram([])

    # All buckets should be zero
    assert all(count == 0 for count in histogram.values())


def test_token_histogram_boundary_values():
    """Test _token_histogram handles boundary values correctly."""
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],
        thresholds=QaThresholds(),
        report_root=Path("/tmp"),
        report_version="v1",
    )

    # Test exact boundary values
    histogram = evaluator._token_histogram([64, 128, 256, 512, 1024, 2048])

    assert histogram["<= 64"] == 1
    assert histogram["<= 128"] == 1
    assert histogram["<= 256"] == 1
    assert histogram["<= 512"] == 1
    assert histogram["<= 1024"] == 1
    assert histogram["<= 2048"] == 1
    assert histogram["> 2048"] == 0


def test_token_histogram_exceeds_largest_bin():
    """Test _token_histogram places values exceeding largest bin correctly."""
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],
        thresholds=QaThresholds(),
        report_root=Path("/tmp"),
        report_version="v1",
    )

    histogram = evaluator._token_histogram([3000, 5000, 10000])

    assert histogram["> 2048"] == 3


def test_qa_thresholds_dataclass_defaults():
    """Test QaThresholds has correct default values."""
    thresholds = QaThresholds()

    assert thresholds.max_missing_embeddings == 0
    assert thresholds.max_orphan_chunks == 0
    assert thresholds.max_checksum_mismatches == 0
    assert thresholds.max_semantic_failures == 0
    assert thresholds.max_semantic_orphans == 0


def test_qa_chunk_record_dataclass():
    """Test QaChunkRecord stores expected fields."""
    chunk = QaChunkRecord(
        uid="test-uid",
        checksum="abc123",
        text="sample text"
    )

    assert chunk.uid == "test-uid"
    assert chunk.checksum == "abc123"
    assert chunk.text == "sample text"


def test_qa_source_record_dataclass():
    """Test QaSourceRecord stores expected fields."""
    source = QaSourceRecord(
        path="/absolute/path.txt",
        relative_path="relative/path.txt",
        document_checksum="doc123",
        git_commit="abc456",
        chunks=[],
        ingest_run_key="run-1"
    )

    assert source.path == "/absolute/path.txt"
    assert source.relative_path == "relative/path.txt"
    assert source.document_checksum == "doc123"
    assert source.git_commit == "abc456"
    assert source.chunks == []
    assert source.ingest_run_key == "run-1"


def test_qa_source_record_default_run_key():
    """Test QaSourceRecord defaults ingest_run_key to None."""
    source = QaSourceRecord(
        path="/path.txt",
        relative_path="path.txt",
        document_checksum="abc",
        git_commit=None,
        chunks=[]
    )

    assert source.ingest_run_key is None


def test_evaluator_records_evaluation_duration(tmp_path):
    """Test evaluator includes evaluation duration in metrics."""
    driver = SequencedDriver(
        results=[
            [{"value": 1}], [{"value": 1}], [{"value": 1}],
            [{"value": 0}], [{"value": 0}], [{"value": 0}],
        ]
    )

    sources = [QaSourceRecord(
        path="/file.txt",
        relative_path="file.txt",
        document_checksum="abc",
        git_commit=None,
        chunks=[_chunk()],
    )]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    result = evaluator.evaluate()

    assert "qa_evaluation_ms" in result.metrics
    assert isinstance(result.metrics["qa_evaluation_ms"], int)
    assert result.metrics["qa_evaluation_ms"] >= 0
    assert result.duration_ms == result.metrics["qa_evaluation_ms"]


def test_evaluator_sanitizes_report_payload(tmp_path, monkeypatch):
    """Test evaluator sanitizes sensitive data in reports."""
    monkeypatch.setenv("SECRET_KEY", "sk-secret-value")

    driver = SequencedDriver(
        results=[
            [{"value": 1}], [{"value": 1}], [{"value": 1}],
            [{"value": 0}], [{"value": 0}], [{"value": 0}],
        ]
    )

    # Create a chunk with sensitive data in text
    sources = [QaSourceRecord(
        path="/file.txt",
        relative_path="file.txt",
        document_checksum="abc",
        git_commit=None,
        chunks=[_chunk(text="API key sk-secret-value should be redacted")],
    )]

    evaluator = IngestionQaEvaluator(
        driver=driver,
        database="neo4j",
        sources=sources,
        thresholds=QaThresholds(),
        report_root=tmp_path,
        report_version="v1",
    )

    evaluator.evaluate()

    # Read the JSON report
    qa_dirs = list(tmp_path.iterdir())
    json_report = (qa_dirs[0] / "quality_report.json").read_text()

    # Secret should be redacted in the report
    assert "sk-secret-value" not in json_report
    assert "***" in json_report