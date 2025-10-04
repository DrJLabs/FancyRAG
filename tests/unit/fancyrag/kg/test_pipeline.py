from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path

import pytest

import fancyrag.kg.pipeline as pipeline
from fancyrag.splitters import CachingFixedSizeSplitter


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


def test_run_pipeline_reuses_cached_splitter_results(monkeypatch, tmp_path):
    _stub_settings(monkeypatch)
    _stub_graph_driver(monkeypatch)
    monkeypatch.setattr(pipeline, "_ensure_document_relationships", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "_collect_counts", lambda *_a, **_k: {})
    monkeypatch.setattr(
        pipeline,
        "_run_semantic_enrichment",
        lambda **_k: pipeline.SemanticEnrichmentStats(),
    )
    monkeypatch.setattr(pipeline, "_resolve_git_commit", lambda: "test-sha")

    class DummyEvaluator:
        def __init__(self, **kwargs):
            """
            Initialize the evaluator with optional threshold configuration.
            
            Parameters:
                thresholds (optional): Threshold settings (e.g., dict or namespace) used to configure evaluation criteria; stored on the instance as `self.thresholds`.
            """
            self.thresholds = kwargs.get("thresholds")

        def evaluate(self):
            """
            Return a fixed evaluation result representing a successful QA evaluation.
            
            Returns:
                types.SimpleNamespace: An object with evaluation fields:
                    - passed: `True`.
                    - status: `"pass"`.
                    - summary: empty dict.
                    - version: `"v1"`.
                    - report_json: `"{}"`.
                    - report_markdown: `"# report"`.
                    - thresholds: the evaluator's configured thresholds.
                    - metrics: dict containing `"graph_counts": {}`.
                    - anomalies: empty list.
                    - duration_ms: `0`.
            """
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

    class RecordingSplitter(CachingFixedSizeSplitter):
        def __init__(self, *args, **kwargs):
            """
            Initialize the splitter and prepare a counter for tracking run invocations per input key.
            
            Forwards all positional and keyword arguments to the base class initializer and creates
            self.run_calls, a dict that maps input keys (strings) to the number of times `run` has been invoked for that key.
            """
            super().__init__(*args, **kwargs)
            self.run_calls: dict[str, int] = {}

        async def run(self, text, config=None) -> pipeline.TextChunks:  # type: ignore[override]
            """
            Record and run the splitter for the given text, tracking how many times each input key is invoked.
            
            Parameters:
                text (str | Sequence[str]): Input text or sequence of texts to split; used as the key for invocation counting.
                config (optional): Optional splitter configuration passed through to the base run method.
            
            Returns:
                pipeline.TextChunks: The chunks produced by the splitter for the given input.
            """
            key = text if isinstance(text, str) else tuple(text)
            self.run_calls[key] = self.run_calls.get(key, 0) + 1
            return await super().run(text, config)

    splitter_holder: dict[str, RecordingSplitter] = {}

    def fake_build(config):
        """
        Create and return a RecordingSplitter configured from the provided config and save it into splitter_holder.
        
        Parameters:
            config: An object with `chunk_size` and `chunk_overlap` attributes used to configure the splitter.
        
        Returns:
            splitter: The created RecordingSplitter instance.
        
        Notes:
            The created instance is stored in splitter_holder["instance"] as a side effect.
        """
        splitter = RecordingSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        splitter_holder["instance"] = splitter
        return splitter

    async def _warm_cache(splitter: RecordingSplitter, text: str):
        """
        Prime the splitter's cache by running it on the provided text.
        
        Parameters:
            splitter (RecordingSplitter): Splitter instance whose cache should be populated for the given text.
            text (str): Input text to process and store in the splitter's cache.
        """
        await splitter.run(text)

    def fake_execute_pipeline(**kwargs):
        """
        Populate the provided splitter's cache using the given source text and return a fixed run identifier.
        
        Parameters:
            splitter: The splitter instance whose cache will be warmed.
            source_text (str): Text to process and populate the splitter's cache.
        
        Returns:
            run_id (str): The fixed identifier "run-id".
        """
        splitter = kwargs["splitter"]
        text = kwargs["source_text"]
        asyncio.run(_warm_cache(splitter, text))
        return "run-id"

    monkeypatch.setattr(pipeline, "build_caching_splitter", fake_build)
    monkeypatch.setattr(pipeline, "_execute_pipeline", fake_execute_pipeline)
    monkeypatch.setattr(pipeline, "_relative_to_repo", lambda path, base=None: Path(path).name)

    source_file = tmp_path / "sample.txt"
    content = "hello world"
    source_file.write_text(content, encoding="utf-8")

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

    pipeline.run_pipeline(options)

    splitter = splitter_holder["instance"]
    assert splitter.run_calls[content] == 1

    scope = str(source_file.resolve())
    with splitter.scoped(scope):
        cached_result = splitter.get_cached(content)
        assert cached_result is not None
        cached_uids = {chunk.uid for chunk in cached_result.chunks}

    with splitter.scoped(scope):
        replay = asyncio.run(splitter.run(content))

    assert {chunk.uid for chunk in replay.chunks}.isdisjoint(cached_uids)


def test_build_chunk_metadata_rejects_duplicate_uids():
    chunk_a = types.SimpleNamespace(text="one", index=0, uid="dup", metadata=None)
    chunk_b = types.SimpleNamespace(text="two", index=1, uid="dup", metadata=None)

    with pytest.raises(ValueError):
        pipeline._build_chunk_metadata(  # noqa: SLF001
            [chunk_a, chunk_b],
            relative_path="sample.txt",
            git_commit="abc123",
        )