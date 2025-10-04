from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fancyrag.qa.evaluator import (
    IngestionQaEvaluator,
    QaChunkRecord,
    QaSourceRecord,
    QaThresholds,
    collect_counts,
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
        def execute_query(self, query, database_=None):
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
        def execute_query(self, _q, _params=None, database_=None):
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
        def execute_query(self, query, database_=None):
            _ = query
            _ = database_
            from neo4j.exceptions import Neo4jError
            raise Neo4jError()

    counts = collect_counts(FailingDriver(), database="neo4j")
    # Should return empty dict, not raise
    assert counts == {}


def test_collect_counts_handles_various_record_formats():
    """Test collect_counts with different record response formats."""
    class VariedFormatDriver:
        def __init__(self):
            self.call_count = 0

        def execute_query(self, query, database_=None):
            _ = database_
            self.call_count += 1
            # Test different response formats
            if "Document" in query:
                # Format 1: Dictionary with value key
                return ([{"value": 10}], None, None)
            elif "Chunk" in query and "HAS_CHUNK" not in query:
                # Format 2: Object with value attribute
                from types import SimpleNamespace
                return ([SimpleNamespace(value=20)], None, None)
            elif "HAS_CHUNK" in query:
                # Format 3: Tuple/list access
                return ([[15]], None, None)
            return ([{"value": 0}], None, None)

    driver = VariedFormatDriver()
    counts = collect_counts(driver, database="neo4j")
    assert counts["documents"] == 10
    assert counts["chunks"] == 20
    assert counts["relationships"] == 15


def test_collect_counts_with_empty_results():
    """Test collect_counts when queries return empty result sets."""
    class EmptyResultDriver:
        def execute_query(self, query, database_=None):
            _ = query
            _ = database_
            return ([], None, None)

    counts = collect_counts(EmptyResultDriver(), database=None)
    # Should handle empty results gracefully
    assert "documents" not in counts or counts["documents"] == 0
    assert "chunks" not in counts or counts["chunks"] == 0
    assert "relationships" not in counts or counts["relationships"] == 0


def test_collect_counts_with_none_values():
    """Test collect_counts when query returns None values."""
    class NoneValueDriver:
        def execute_query(self, query, database_=None):
            _ = query
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


def test_query_value_with_mapping_record():
    """Test _query_value when record is a Mapping."""
    class TestDriver:
        def execute_query(self, query, parameters, database_=None):
            _ = query
            _ = parameters
            _ = database_
            return ([{"value": 42}], None, None)

    evaluator = IngestionQaEvaluator(
        driver=TestDriver(),
        database=None,
        sources=[_chunk_record()],
        thresholds=QaThresholds(),
        report_root=Path("."),  # nosec
        report_version="v1",
    )

    value = evaluator._query_value("MATCH (n) RETURN count(n)", {})
    assert value == 42


def test_query_value_with_object_attribute():
    """Test _query_value when record has value attribute."""
    from types import SimpleNamespace

    class TestDriver:
        def execute_query(self, query, parameters, database_=None):
            _ = query
            _ = parameters
            _ = database_
            return ([SimpleNamespace(value=99)], None, None)

    evaluator = IngestionQaEvaluator(
        driver=TestDriver(),
        database=None,
        sources=[_chunk_record()],
        thresholds=QaThresholds(),
        report_root=Path("."),  # nosec
        report_version="v1",
    )

    value = evaluator._query_value("MATCH (n) RETURN count(n)", {})
    assert value == 99


def test_query_value_with_list_access():
    """Test _query_value when record requires list index access."""
    class TestDriver:
        def execute_query(self, query, parameters, database_=None):
            _ = query
            _ = parameters
            _ = database_
            return ([[77]], None, None)

    evaluator = IngestionQaEvaluator(
        driver=TestDriver(),
        database=None,
        sources=[_chunk_record()],
        thresholds=QaThresholds(),
        report_root=Path("."),  # nosec
        report_version="v1",
    )

    value = evaluator._query_value("MATCH (n) RETURN count(n)", {})
    assert value == 77


def test_query_value_with_empty_result():
    """Test _query_value when query returns no records."""
    class TestDriver:
        def execute_query(self, query, parameters, database_=None):
            _ = query
            _ = parameters
            _ = database_
            return ([], None, None)

    evaluator = IngestionQaEvaluator(
        driver=TestDriver(),
        database=None,
        sources=[_chunk_record()],
        thresholds=QaThresholds(),
        report_root=Path("."),  # nosec
        report_version="v1",
    )

    value = evaluator._query_value("MATCH (n) RETURN count(n)", {})
    assert value == 0


def test_query_value_with_none_value():
    """Test _query_value when record value is None."""
    class TestDriver:
        def execute_query(self, query, parameters, database_=None):
            _ = query
            _ = parameters
            _ = database_
            return ([{"value": None}], None, None)

    evaluator = IngestionQaEvaluator(
        driver=TestDriver(),
        database=None,
        sources=[_chunk_record()],
        thresholds=QaThresholds(),
        report_root=Path("."),  # nosec
        report_version="v1",
    )

    value = evaluator._query_value("MATCH (n) RETURN count(n)", {})
    assert value == 0


def test_query_value_with_no_sources():
    """Test _query_value returns 0 when no sources provided."""
    evaluator = IngestionQaEvaluator(
        driver=object(),
        database=None,
        sources=[],  # No sources
        thresholds=QaThresholds(),
        report_root=Path("."),  # nosec
        report_version="v1",
    )

    value = evaluator._query_value("MATCH (n) RETURN count(n)", {})
    assert value == 0


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