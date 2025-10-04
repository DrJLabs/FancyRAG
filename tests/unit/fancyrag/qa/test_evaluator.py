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
    # Ensure reported paths are relative strings and point to the right artifacts
    import os

    assert result.report_json.endswith("quality_report.json")
    assert result.report_markdown.endswith("quality_report.md")
    assert not os.path.isabs(result.report_json)
    assert not os.path.isabs(result.report_markdown)