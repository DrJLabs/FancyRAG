from __future__ import annotations
import contextlib
import os
import types
from pathlib import Path
from typing import Any

import pytest

from fancyrag.kg import phases
from fancyrag.kg.pipeline import SemanticEnrichmentStats, SemanticQaSummary


def test_resolve_settings_honours_overrides():
    presets = {
        "default": {"chunk_size": 100, "chunk_overlap": 10, "include": ("*.txt",)},
        "text": {"chunk_size": 200, "chunk_overlap": 20, "include": ("*.md",)},
    }

    resolved = phases.resolve_settings(
        profile="text",
        chunk_size=512,
        chunk_overlap=0,
        include_patterns_override=("*.rst",),
        semantic_enabled=True,
        semantic_max_concurrency=4,
        profile_presets=presets,
        default_profile="default",
        ensure_positive=lambda value, name: value if value > 0 else (_ for _ in ()).throw(
            ValueError(name)
        ),
        ensure_non_negative=lambda value, name: value if value >= 0 else (_ for _ in ()).throw(
            ValueError(name)
        ),
    )

    assert resolved.profile == "text"
    assert resolved.chunk_size == 512
    assert resolved.chunk_overlap == 0
    assert resolved.include_patterns == ("*.rst",)
    assert resolved.semantic_max_concurrency == 4


def test_discover_sources_directory_mode(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    file_a = docs_dir / "a.txt"
    file_b = docs_dir / "b.md"
    file_a.write_text("alpha", encoding="utf-8")
    file_b.write_text("beta", encoding="utf-8")

    result = phases.discover_sources(
        source=docs_dir / "ignored.txt",
        source_dir=docs_dir,
        include_patterns=("*.txt", "*.md"),
        relative_to_repo=lambda path, base=None: str(Path(path).relative_to(base or tmp_path)),
        read_source=lambda path: path.read_text(encoding="utf-8"),
        read_directory_source=lambda path: path.read_text(encoding="utf-8"),
        discover_source_files=lambda directory, patterns: sorted(
            directory.glob(patterns[0])
        )
        + sorted(directory.glob(patterns[1])),
        compute_checksum=lambda text: f"cs:{text}",
        source_spec_factory=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )

    assert result.source_mode == "directory"
    assert result.source_root == docs_dir
    assert len(result.sources) == 2
    paths = {spec.path for spec in result.sources}
    assert paths == {file_a, file_b}


def test_discover_sources_file_mode(tmp_path: Path):
    file_path = tmp_path / "single.txt"
    file_path.write_text("content", encoding="utf-8")

    result = phases.discover_sources(
        source=file_path,
        source_dir=None,
        include_patterns=(),
        relative_to_repo=lambda path, base=None: Path(path).name,
        read_source=lambda path: path.read_text(encoding="utf-8"),
        read_directory_source=lambda path: path.read_text(encoding="utf-8"),
        discover_source_files=lambda *_args, **_kwargs: [],
        compute_checksum=lambda text: f"hash:{len(text)}",
        source_spec_factory=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )

    assert result.source_mode == "file"
    assert result.source_root == file_path
    assert len(result.sources) == 1
    spec = result.sources[0]
    assert spec.path == file_path
    assert spec.checksum == "hash:7"


def test_build_clients_invokes_factories():
    settings = types.SimpleNamespace()

    calls: dict[str, Any] = {}

    def shared_client_factory(settings_obj):
        calls["shared_client"] = settings_obj
        return "shared-client"

    def embedder_factory(client, settings_obj):
        calls["embedder"] = (client, settings_obj)
        return "embedder"

    def llm_factory(client, settings_obj):
        calls["llm"] = (client, settings_obj)
        return "llm"

    def splitter_config_factory(chunk_size, chunk_overlap):
        calls["splitter_config"] = (chunk_size, chunk_overlap)
        return types.SimpleNamespace(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def splitter_factory(config):
        calls["splitter"] = config
        return "splitter"

    bundle = phases.build_clients(
        settings=settings,
        chunk_size=256,
        chunk_overlap=32,
        shared_client_factory=shared_client_factory,
        embedder_factory=embedder_factory,
        llm_factory=llm_factory,
        splitter_config_factory=splitter_config_factory,
        splitter_factory=splitter_factory,
    )

    assert calls["shared_client"] is settings
    assert bundle.shared_client == "shared-client"
    assert bundle.embedder == "embedder"
    assert bundle.llm == "llm"
    assert bundle.splitter == "splitter"
    assert calls["splitter_config"] == (256, 32)


class _StubSplitter:
    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}
        self.scopes: list[str] = []
        self.run_calls: list[str] = []

    def scoped(self, scope: str):
        self.scopes.append(scope)
        return contextlib.nullcontext(self)

    def get_cached(self, text: str):
        return self.cache.get(text)

    async def run(self, text: str):
        self.run_calls.append(text)
        chunks = [
            types.SimpleNamespace(uid="chunk-1", text=text, index=0),
            types.SimpleNamespace(uid="chunk-2", text=text.upper(), index=1),
        ]
        result = types.SimpleNamespace(chunks=chunks)
        self.cache[text] = result
        return result


class _ChunkMeta:
    def __init__(self, uid: str, checksum: str):
        self.uid = uid
        self.sequence = 1 if uid.endswith("1") else 2
        self.index = self.sequence - 1
        self.checksum = checksum
        self.relative_path = "doc.txt"
        self.git_commit = "sha"


class _RestrictedEnviron(dict):
    """Mapping that raises when forbidden environment keys are accessed."""

    def __init__(self, backing: dict[str, str], forbidden: set[str]):
        super().__init__(backing)
        self._backing = dict(backing)
        self._forbidden = forbidden

    def _check(self, key: str) -> None:
        if key in self._forbidden:
            raise AssertionError(f"helper accessed forbidden environment variable: {key}")

    def __getitem__(self, key: str) -> str:
        self._check(key)
        return self._backing[key]

    def get(self, key: str, default: Any = None) -> Any:
        self._check(key)
        return self._backing.get(key, default)

    def __contains__(self, key: object) -> bool:  # pragma: no cover - defensive
        return key in self._backing

    def copy(self) -> dict[str, str]:  # pragma: no cover - defensive
        return dict(self._backing)


def test_ingest_source_with_semantic_enrichment(monkeypatch, tmp_path):
    spec = types.SimpleNamespace(
        path=tmp_path / "doc.txt",
        relative_path="doc.txt",
        text="hello world",
        checksum="abc123",
    )
    options = types.SimpleNamespace(database="neo4j")
    splitter = _StubSplitter()
    clients = phases.ClientBundle(
        shared_client="client",
        embedder="embedder",
        llm="llm",
        splitter=splitter,
    )

    captured: dict[str, Any] = {}

    def execute_pipeline(**kwargs):
        captured["execute"] = kwargs
        return "run-id"

    def build_chunk_metadata(chunks, *, relative_path, git_commit):
        captured["metadata_input"] = (chunks, relative_path, git_commit)
        return [_ChunkMeta("chunk-1", "chk1"), _ChunkMeta("chunk-2", "chk2")]

    def ensure_document_relationships(driver, **kwargs):
        captured["ensure_relationships"] = kwargs

    def run_semantic_enrichment(**kwargs):
        captured["semantic"] = kwargs
        stats = SemanticEnrichmentStats()
        stats.chunks_processed = 2
        stats.chunk_failures = 0
        stats.nodes_written = 4
        stats.relationships_written = 6
        return stats

    artifacts = phases.ingest_source(
        spec=spec,
        options=options,
        uri="bolt://localhost:7687",
        auth=("neo4j", "secret"),
        driver="driver",
        clients=clients,
        git_commit="sha",
        reset_database=True,
        execute_pipeline=execute_pipeline,
        build_chunk_metadata=build_chunk_metadata,
        ensure_document_relationships=ensure_document_relationships,
        semantic_enabled=True,
        semantic_max_concurrency=2,
        run_semantic_enrichment=run_semantic_enrichment,
        semantic_stats_factory=SemanticEnrichmentStats,
        ingest_run_key_factory=lambda: "kg-build:test",
    )

    assert artifacts.run_id == "run-id"
    assert artifacts.qa_source.relative_path == "doc.txt"
    assert artifacts.log_entry["chunks"] == 2
    assert len(artifacts.chunk_entries) == 2
    assert artifacts.semantic_stats.nodes_written == 4
    assert captured["execute"]["reset_database"] is True
    assert captured["semantic"]["max_concurrency"] == 2
    assert splitter.run_calls == ["hello world"]


def test_ingest_source_without_semantic(monkeypatch, tmp_path):
    spec = types.SimpleNamespace(
        path=tmp_path / "doc.txt",
        relative_path="doc.txt",
        text="hello world",
        checksum="abc123",
    )
    options = types.SimpleNamespace(database=None)
    splitter = _StubSplitter()
    clients = phases.ClientBundle(
        shared_client="client",
        embedder="embedder",
        llm="llm",
        splitter=splitter,
    )

    artifacts = phases.ingest_source(
        spec=spec,
        options=options,
        uri="bolt://localhost:7687",
        auth=("neo4j", "secret"),
        driver="driver",
        clients=clients,
        git_commit=None,
        reset_database=False,
        execute_pipeline=lambda **_: "run-id",
        build_chunk_metadata=lambda *args, **kwargs: [_ChunkMeta("chunk-1", "chk1")],
        ensure_document_relationships=lambda *_a, **_k: None,
        semantic_enabled=False,
        semantic_max_concurrency=1,
        run_semantic_enrichment=lambda **_: (_ for _ in ()).throw(RuntimeError("should not run")),
        semantic_stats_factory=SemanticEnrichmentStats,
        ingest_run_key_factory=lambda: "key",
    )

    assert artifacts.semantic_stats.nodes_written == 0
    assert artifacts.semantic_stats.relationships_written == 0


def test_perform_qa_success_no_fallback(tmp_path):
    qa_limits = types.SimpleNamespace(
        max_missing_embeddings=0,
        max_orphan_chunks=0,
        max_checksum_mismatches=0,
        max_semantic_failures=0,
        max_semantic_orphans=0,
    )
    semantic_totals = SemanticEnrichmentStats()
    semantic_totals.chunks_processed = 2
    semantic_totals.nodes_written = 4

    collect_calls: list[Any] = []

    def collect_counts(*_args, **_kwargs):
        collect_calls.append(1)
        return {"nodes": 10}

    class DummyEvaluator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def evaluate(self):
            return types.SimpleNamespace(
                passed=True,
                status="pass",
                summary={},
                version="v1",
                report_json="{}",
                report_markdown="# report",
                thresholds=self.kwargs["thresholds"],
                metrics={"graph_counts": {"nodes": 5}},
                anomalies=[],
                duration_ms=42,
            )

    outcome = phases.perform_qa(
        driver="driver",
        database="neo4j",
        qa_sources=["source"],
        semantic_enabled=True,
        semantic_totals=semantic_totals,
        qa_limits=qa_limits,
        qa_report_dir=tmp_path,
        qa_report_version="1.0",
        qa_evaluator_factory=DummyEvaluator,
        collect_counts=collect_counts,
        rollback_ingest=lambda *_a, **_k: None,
        semantic_summary_factory=SemanticQaSummary,
    )

    assert outcome.qa_section["status"] == "pass"
    assert outcome.counts == {"nodes": 5}
    assert outcome.semantic_summary.nodes_written == 4
    assert collect_calls == []


def test_perform_qa_triggers_counts_fallback(tmp_path):
    qa_limits = types.SimpleNamespace(
        max_missing_embeddings=0,
        max_orphan_chunks=0,
        max_checksum_mismatches=0,
        max_semantic_failures=0,
        max_semantic_orphans=0,
    )
    semantic_totals = SemanticEnrichmentStats()

    collect_calls: list[str] = []

    def collect_counts(*_args, **_kwargs):
        collect_calls.append("called")
        return {"nodes": 7}

    class DummyEvaluator:
        def __init__(self, **_kwargs):
            pass

        def evaluate(self):
            return types.SimpleNamespace(
                passed=True,
                status="pass",
                summary={},
                version="v1",
                report_json="{}",
                report_markdown="# report",
                thresholds=types.SimpleNamespace(),
                metrics={"graph_counts": {}},
                anomalies=[],
                duration_ms=10,
            )

    outcome = phases.perform_qa(
        driver="driver",
        database=None,
        qa_sources=[],
        semantic_enabled=False,
        semantic_totals=semantic_totals,
        qa_limits=qa_limits,
        qa_report_dir=tmp_path,
        qa_report_version="1.0",
        qa_evaluator_factory=DummyEvaluator,
        collect_counts=collect_counts,
        rollback_ingest=lambda *_a, **_k: None,
        semantic_summary_factory=SemanticQaSummary,
    )

    assert outcome.counts == {"nodes": 7}
    assert collect_calls == ["called"]


def test_perform_qa_failure_rolls_back(tmp_path):
    qa_limits = types.SimpleNamespace(
        max_missing_embeddings=0,
        max_orphan_chunks=0,
        max_checksum_mismatches=0,
        max_semantic_failures=0,
        max_semantic_orphans=0,
    )
    semantic_totals = SemanticEnrichmentStats()

    rollback_calls: list[str] = []

    def rollback(*_args, **_kwargs):
        rollback_calls.append("rollback")

    class FailingEvaluator:
        def __init__(self, **_kwargs):
            pass

        def evaluate(self):
            return types.SimpleNamespace(
                passed=False,
                status="fail",
                summary={},
                version="v1",
                report_json="{}",
                report_markdown="# report",
                thresholds=types.SimpleNamespace(),
                metrics={"graph_counts": {}},
                anomalies=["bad"],
                duration_ms=99,
            )

    with pytest.raises(RuntimeError, match="Ingestion QA gating failed"):
        phases.perform_qa(
            driver="driver",
            database="neo4j",
            qa_sources=["src"],
            semantic_enabled=False,
            semantic_totals=semantic_totals,
            qa_limits=qa_limits,
            qa_report_dir=tmp_path,
            qa_report_version="1.0",
            qa_evaluator_factory=FailingEvaluator,
            collect_counts=lambda *_a, **_k: {},
            rollback_ingest=rollback,
            semantic_summary_factory=SemanticQaSummary,
        )

    assert rollback_calls == ["rollback"]


def test_helpers_do_not_touch_environment(monkeypatch, tmp_path):
    snapshot = dict(os.environ)
    forbidden = {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "QDRANT_API_KEY",
    }
    monkeypatch.setattr(os, "environ", _RestrictedEnviron(snapshot, forbidden))

    presets = {
        "default": {"chunk_size": 256, "chunk_overlap": 32, "include": ("*.txt",)},
    }

    resolved = phases.resolve_settings(
        profile=None,
        chunk_size=None,
        chunk_overlap=None,
        include_patterns_override=None,
        semantic_enabled=False,
        semantic_max_concurrency=2,
        profile_presets=presets,
        default_profile="default",
        ensure_positive=lambda value, name: value if value > 0 else (_ for _ in ()).throw(ValueError(name)),
        ensure_non_negative=lambda value, name: value if value >= 0 else (_ for _ in ()).throw(ValueError(name)),
    )
    assert resolved.chunk_size == 256

    source_file = tmp_path / "alpha.txt"
    source_file.write_text("alpha", encoding="utf-8")

    discovery = phases.discover_sources(
        source=source_file,
        source_dir=None,
        include_patterns=(),
        relative_to_repo=lambda path, base=None: Path(path).name,
        read_source=lambda path: path.read_text(encoding="utf-8"),
        read_directory_source=lambda path: path.read_text(encoding="utf-8"),
        discover_source_files=lambda *_args, **_kwargs: [],
        compute_checksum=lambda text: f"hash:{len(text)}",
        source_spec_factory=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    assert discovery.source_mode == "file"

    bundle = phases.build_clients(
        settings=types.SimpleNamespace(),
        chunk_size=resolved.chunk_size,
        chunk_overlap=resolved.chunk_overlap,
        shared_client_factory=lambda settings: object(),
        embedder_factory=lambda client, settings: object(),
        llm_factory=lambda client, settings: object(),
        splitter_config_factory=lambda **_: types.SimpleNamespace(chunk_size=resolved.chunk_size, chunk_overlap=resolved.chunk_overlap),
        splitter_factory=lambda _config: _StubSplitter(),
    )
    assert isinstance(bundle, phases.ClientBundle)

    spec = types.SimpleNamespace(
        path=source_file,
        relative_path="alpha.txt",
        text="alpha",
        checksum="abc",
    )

    artifacts = phases.ingest_source(
        spec=spec,
        options=types.SimpleNamespace(database=None),
        uri="bolt://localhost:7687",
        auth=("neo4j", "secret"),
        driver="driver",
        clients=bundle,
        git_commit=None,
        reset_database=False,
        execute_pipeline=lambda **_: "run-id",
        build_chunk_metadata=lambda chunks, *, relative_path, git_commit: [_ChunkMeta("chunk-1", "chk1")],
        ensure_document_relationships=lambda *_a, **_k: None,
        semantic_enabled=False,
        semantic_max_concurrency=1,
        run_semantic_enrichment=lambda **_: (_ for _ in ()).throw(RuntimeError("should not run")),
        semantic_stats_factory=SemanticEnrichmentStats,
        ingest_run_key_factory=lambda: "key",
    )
    assert artifacts.qa_source.relative_path == "alpha.txt"

    qa_limits = types.SimpleNamespace(
        max_missing_embeddings=0,
        max_orphan_chunks=0,
        max_checksum_mismatches=0,
        max_semantic_failures=0,
        max_semantic_orphans=0,
    )

    class DummyEvaluator:
        def __init__(self, **_kwargs):
            pass

        def evaluate(self):
            return types.SimpleNamespace(
                passed=True,
                status="pass",
                summary={},
                version="v1",
                report_json="{}",
                report_markdown="# report",
                thresholds=types.SimpleNamespace(),
                metrics={"graph_counts": {"nodes": 1}},
                anomalies=[],
                duration_ms=1,
            )

    outcome = phases.perform_qa(
        driver="driver",
        database=None,
        qa_sources=[artifacts.qa_source],
        semantic_enabled=False,
        semantic_totals=SemanticEnrichmentStats(),
        qa_limits=qa_limits,
        qa_report_dir=tmp_path,
        qa_report_version="1.0",
        qa_evaluator_factory=DummyEvaluator,
        collect_counts=lambda *_a, **_k: {"nodes": 1},
        rollback_ingest=lambda *_a, **_k: None,
        semantic_summary_factory=SemanticQaSummary,
    )
    assert outcome.counts == {"nodes": 1}
