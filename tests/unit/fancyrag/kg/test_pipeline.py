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

    class RecordingSplitter(CachingFixedSizeSplitter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.run_calls: dict[str, int] = {}

        async def run(self, text, config=None) -> pipeline.TextChunks:  # type: ignore[override]
            key = text if isinstance(text, str) else tuple(text)
            self.run_calls[key] = self.run_calls.get(key, 0) + 1
            return await super().run(text, config)

    splitter_holder: dict[str, RecordingSplitter] = {}

    def fake_build(config):
        splitter = RecordingSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        splitter_holder["instance"] = splitter
        return splitter

    async def _warm_cache(splitter: RecordingSplitter, text: str):
        await splitter.run(text)

    def fake_execute_pipeline(**kwargs):
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


def test_build_chunk_metadata_with_valid_chunks():
    """Test _build_chunk_metadata with valid chunk objects."""
    chunk_a = types.SimpleNamespace(text="first text", index=0, uid="uid-1", metadata=None)

    chunk_b = types.SimpleNamespace(text="second text", index=1, uid="uid-2", metadata=None)
    chunk_c = types.SimpleNamespace(text="third text", index=2, uid="uid-3", metadata=None)

    result = pipeline._build_chunk_metadata(  # noqa: SLF001
        [chunk_a, chunk_b, chunk_c],
        relative_path="docs/example.md",
        git_commit="abc123def",
    )

    assert len(result) == 3
    assert result[0].uid == "uid-1"
    assert result[0].sequence == 1
    assert result[0].index == 0
    assert result[0].relative_path == "docs/example.md"
    assert result[0].git_commit == "abc123def"
    assert result[0].checksum  # Should be computed

    assert result[1].sequence == 2
    assert result[2].sequence == 3


def test_build_chunk_metadata_with_none_git_commit():
    """Test _build_chunk_metadata handles None git_commit correctly."""
    chunk = types.SimpleNamespace(text="content", index=0, uid="uid-1", metadata=None)

    result = pipeline._build_chunk_metadata(  # noqa: SLF001
        [chunk],
        relative_path="file.txt",
        git_commit=None,
    )

    assert len(result) == 1
    assert result[0].git_commit is None


def test_build_chunk_metadata_missing_uid_raises_error():
    """Test that missing uid attribute raises ValueError."""
    chunk_no_uid = types.SimpleNamespace(text="content", index=0, metadata=None)

    with pytest.raises(ValueError, match="chunk object missing uid"):
        pipeline._build_chunk_metadata(  # noqa: SLF001
            [chunk_no_uid],
            relative_path="file.txt",
            git_commit="abc",
        )


def test_build_chunk_metadata_none_uid_raises_error():
    """Test that None uid value raises ValueError."""
    chunk_none_uid = types.SimpleNamespace(text="content", index=0, uid=None, metadata=None)

    with pytest.raises(ValueError, match="chunk object missing uid"):
        pipeline._build_chunk_metadata(  # noqa: SLF001
            [chunk_none_uid],
            relative_path="file.txt",
            git_commit="abc",
        )


def test_build_chunk_metadata_empty_chunks_list():
    """Test _build_chunk_metadata with empty chunks list."""
    result = pipeline._build_chunk_metadata(  # noqa: SLF001
        [],
        relative_path="empty.txt",
        git_commit="abc",
    )

    assert result == []


def test_build_chunk_metadata_handles_empty_text():
    """Test _build_chunk_metadata handles chunks with empty text."""
    chunk_empty = types.SimpleNamespace(text="", index=0, uid="uid-empty", metadata=None)
    chunk_none = types.SimpleNamespace(text=None, index=1, uid="uid-none", metadata=None)

    result = pipeline._build_chunk_metadata(  # noqa: SLF001
        [chunk_empty, chunk_none],
        relative_path="sparse.txt",
        git_commit="def",
    )

    assert len(result) == 2
    assert result[0].checksum  # Empty text should still have checksum
    assert result[1].checksum  # None text treated as empty


def test_build_chunk_metadata_checksum_differs_for_different_text():
    """Test that different text produces different checksums."""
    chunk_a = types.SimpleNamespace(text="alpha", index=0, uid="uid-a", metadata=None)

    chunk_b = types.SimpleNamespace(text="beta", index=1, uid="uid-b", metadata=None)

    result = pipeline._build_chunk_metadata(  # noqa: SLF001
        [chunk_a, chunk_b],
        relative_path="test.txt",
        git_commit="abc",
    )

    assert result[0].checksum != result[1].checksum


def test_build_chunk_metadata_duplicate_uid_in_middle():
    """Test duplicate UID detection when duplicate is not at start."""
    chunks = [
        types.SimpleNamespace(text="one", index=0, uid="unique-1", metadata=None),
        types.SimpleNamespace(text="two", index=1, uid="unique-2", metadata=None),
        types.SimpleNamespace(text="three", index=2, uid="unique-1", metadata=None),  # Duplicate
    ]

    with pytest.raises(ValueError, match="duplicate chunk uid"):
        pipeline._build_chunk_metadata(  # noqa: SLF001
            chunks,
            relative_path="dup.txt",
            git_commit="abc",
        )


def test_build_chunk_metadata_sequence_starts_at_one():
    """Test that sequence numbers start at 1, not 0."""
    chunk = types.SimpleNamespace(text="content", index=0, uid="uid-1", metadata=None)

    result = pipeline._build_chunk_metadata(  # noqa: SLF001
        [chunk],
        relative_path="file.txt",
        git_commit="abc",
    )

    assert result[0].sequence == 1


def test_build_chunk_metadata_uses_fallback_index():
    """Test that missing index falls back to sequence-1."""
    chunk_no_index = types.SimpleNamespace(text="content", uid="uid-1", metadata=None)

    result = pipeline._build_chunk_metadata(  # noqa: SLF001
        [chunk_no_index],
        relative_path="file.txt",
        git_commit="abc",
    )

    # When index is missing, should use sequence-1 (which is 1-1=0)
    assert result[0].index == 0


def test_build_chunk_metadata_preserves_provided_index():
    """Test that provided index is preserved even if it differs from sequence."""
    chunk = types.SimpleNamespace(text="content", index=42, uid="uid-1", metadata=None)

    result = pipeline._build_chunk_metadata(  # noqa: SLF001
        [chunk],
        relative_path="file.txt",
        git_commit="abc",
    )

    assert result[0].index == 42
    assert result[0].sequence == 1  # Sequence still starts at 1


def test_build_chunk_metadata_with_special_characters_in_path():
    """Test _build_chunk_metadata with special characters in relative_path."""
    chunk = types.SimpleNamespace(text="content", index=0, uid="uid-1", metadata=None)
    special_path = "docs/项目/file with spaces & symbols\\!.md"

    result = pipeline._build_chunk_metadata(  # noqa: SLF001
        [chunk],
        relative_path=special_path,
        git_commit="abc",
    )

    assert result[0].relative_path == special_path


def test_build_chunk_metadata_with_long_git_commit():
    """Test _build_chunk_metadata with full SHA git commit."""
    chunk = types.SimpleNamespace(text="content", index=0, uid="uid-1", metadata=None)
    long_commit = "a" * 40  # Full SHA-1 hash length

    result = pipeline._build_chunk_metadata(  # noqa: SLF001
        [chunk],
        relative_path="file.txt",
        git_commit=long_commit,
    )

    assert result[0].git_commit == long_commit


def test_build_chunk_metadata_with_many_chunks():
    """Test _build_chunk_metadata scales with many chunks."""
    chunks = [
        types.SimpleNamespace(text=f"chunk {i}", index=i, uid=f"uid-{i}", metadata=None)
        for i in range(100)
    ]

    result = pipeline._build_chunk_metadata(  # noqa: SLF001
        chunks,
        relative_path="large.txt",
        git_commit="abc",
    )

    assert len(result) == 100
    assert result[0].sequence == 1
    assert result[99].sequence == 100
    assert all(r.uid == f"uid-{i}" for i, r in enumerate(result))