from __future__ import annotations

import os
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
STUBS = ROOT / "stubs"
for path in (STUBS, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fancyrag.cli.kg_build_main as kg  # noqa: E402


def test_parse_args_defaults():
    args = kg._parse_args([])
    assert pathlib.Path(args.source) == kg.DEFAULT_SOURCE
    assert args.source_dir is None
    assert args.include_patterns is None
    assert args.profile is None
    assert args.chunk_size is None
    assert args.chunk_overlap is None
    assert args.semantic_enabled is False
    assert args.semantic_max_concurrency == 5
    assert pathlib.Path(args.log_path) == kg.DEFAULT_LOG_PATH
    assert pathlib.Path(args.qa_report_dir) == kg.DEFAULT_QA_DIR


def test_parse_args_enable_semantic_sets_flag():
    args = kg._parse_args(["--enable-semantic"])
    assert args.semantic_enabled is True


def test_run_invokes_ensure_env_before_pipeline(monkeypatch, tmp_path):
    required_vars: list[str] = []

    def fake_ensure_env(var: str) -> str:
        required_vars.append(var)
        value = os.environ.get(var)
        if value is None:
            value = f"stub-{var.lower()}"
            os.environ[var] = value
        return value

    monkeypatch.setattr(kg, "ensure_env", fake_ensure_env)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")

    monkeypatch.setattr(
        kg,
        "OpenAISettings",
        types.SimpleNamespace(load=lambda actor: types.SimpleNamespace(
            chat_model="gpt-4.1-mini",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            max_attempts=3,
        )),
    )

    class DummyClient:
        def __init__(self, settings):
            self.settings = settings

    monkeypatch.setattr(kg, "SharedOpenAIClient", DummyClient)
    monkeypatch.setattr(kg, "SharedOpenAIEmbedder", lambda *args, **kwargs: object())
    monkeypatch.setattr(kg, "SharedOpenAILLM", lambda *args, **kwargs: object())

    class DummySplitter:
        def __init__(self, *args, **kwargs):
            self._cache: dict[str, types.SimpleNamespace] = {}

        def scoped(self, _token: str):
            class _Context:
                def __enter__(self_inner):
                    return self

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Context()

        def get_cached(self, text: str):
            return self._cache.get(text)

        async def run(self, text: str):
            result = types.SimpleNamespace(chunks=[])
            self._cache[text] = result
            return result

    monkeypatch.setattr(kg, "CachingFixedSizeSplitter", DummySplitter)

    monkeypatch.setattr(kg, "_execute_pipeline", lambda **_: "run-id")
    monkeypatch.setattr(kg, "_ensure_document_relationships", lambda *args, **kwargs: None)
    monkeypatch.setattr(kg, "_build_chunk_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr(kg, "_collect_counts", lambda *args, **kwargs: {})
    monkeypatch.setattr(kg, "_relative_to_repo", lambda path, base=None: pathlib.Path(path).name)
    monkeypatch.setattr(kg, "_run_semantic_enrichment", lambda **kwargs: kg.SemanticEnrichmentStats())

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

    monkeypatch.setattr(kg, "IngestionQaEvaluator", DummyEvaluator)

    class DummyDriver:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        kg,
        "GraphDatabase",
        types.SimpleNamespace(driver=lambda *_, **__: DummyDriver()),
    )

    output_path = tmp_path / "log.json"

    result = kg.run(["--log-path", str(output_path)])

    assert result["status"] == "success"
    assert output_path.exists()
    assert required_vars == [
        "OPENAI_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
    ]


def test_scripts_wrapper_exposes_packaged_main():
    import importlib

    script_mod = importlib.import_module("scripts.kg_build")
    assert script_mod.main is kg.main
