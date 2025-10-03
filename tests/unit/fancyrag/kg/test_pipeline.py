from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import fancyrag.kg.pipeline as pipeline


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")


def _stub_settings(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "OpenAISettings",
        types.SimpleNamespace(
            load=lambda actor: types.SimpleNamespace(
                chat_model="gpt-4.1-mini",
                embedding_model="text-embedding-3-small",
                embedding_dimensions=1536,
                max_attempts=3,
            )
        ),
    )
    monkeypatch.setattr(pipeline, "SharedOpenAIClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pipeline, "SharedOpenAIEmbedder", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pipeline, "SharedOpenAILLM", lambda *_args, **_kwargs: object())


def _stub_graph_driver(monkeypatch):
    class DummyDriver:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        pipeline,
        "GraphDatabase",
        types.SimpleNamespace(driver=lambda *_, **__: DummyDriver()),
    )


def _stub_pipeline_dependencies(monkeypatch):
    monkeypatch.setattr(pipeline, "_execute_pipeline", lambda **_kwargs: "run-id")
    monkeypatch.setattr(pipeline, "_ensure_document_relationships", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_build_chunk_metadata", lambda *_a, **_k: [])
    monkeypatch.setattr(pipeline, "_relative_to_repo", lambda path, _base=None: Path(path).name)
    monkeypatch.setattr(pipeline, "_collect_counts", lambda *_a, **_k: {})
    monkeypatch.setattr(
        pipeline,
        "_run_semantic_enrichment",
        lambda **_k: pipeline.SemanticEnrichmentStats(),
    )

    class DummyEvaluator:
        def __init__(self, **kwargs):
            self.thresholds = kwargs.get("thresholds")

        def evaluate(self):
            return types.SimpleNamespace(
                passed=True,
                status="pass",
                summary={},
                version="v1",
                report_json="{}",
                report_markdown="# report",
                thresholds=self.thresholds,
                metrics={"graph_counts": {}},
                anomalies=[],
                duration_ms=0,
            )

    monkeypatch.setattr(pipeline, "IngestionQaEvaluator", DummyEvaluator)


def test_run_pipeline_writes_log(monkeypatch, tmp_path, capsys):
    _stub_settings(monkeypatch)
    _stub_graph_driver(monkeypatch)
    _stub_pipeline_dependencies(monkeypatch)

    source_file = tmp_path / "sample.txt"
    source_file.write_text("hello world", encoding="utf-8")

    options = pipeline.PipelineOptions(
        source=source_file,
        source_dir=None,
        include_patterns=None,
        profile=None,
        chunk_size=None,
        chunk_overlap=None,
        database="neo4j",
        log_path=tmp_path / "kg-log.json",
        qa_report_dir=tmp_path,
        qa_limits=pipeline.QaLimits(),
        semantic_enabled=False,
        semantic_max_concurrency=5,
        reset_database=False,
    )

    result = pipeline.run_pipeline(options)
    assert result["status"] == "success"

    output = capsys.readouterr().out.strip().splitlines()
    assert output
    assert json.loads(output[0])["status"] == "success"
    assert options.log_path.exists()


def test_run_pipeline_validates_chunk_size(monkeypatch, tmp_path):
    _stub_settings(monkeypatch)
    _stub_graph_driver(monkeypatch)
    _stub_pipeline_dependencies(monkeypatch)

    source_file = tmp_path / "sample.txt"
    source_file.write_text("hello world", encoding="utf-8")

    options = pipeline.PipelineOptions(
        source=source_file,
        source_dir=None,
        include_patterns=None,
        profile=None,
        chunk_size=0,
        chunk_overlap=50,
        database=None,
        log_path=tmp_path / "kg-log.json",
        qa_report_dir=tmp_path,
        qa_limits=pipeline.QaLimits(),
        semantic_enabled=False,
        semantic_max_concurrency=5,
        reset_database=False,
    )

    with pytest.raises(ValueError):
        pipeline.run_pipeline(options)
