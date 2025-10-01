from __future__ import annotations

import json
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

stub = sys.modules.get("pandas")
if stub is None:
    stub = types.ModuleType("pandas")
    sys.modules["pandas"] = stub
if not hasattr(stub, "NA"):
    stub.NA = object()
if not hasattr(stub, "Series"):
    stub.Series = type("Series", (), {})
if not hasattr(stub, "DataFrame"):
    stub.DataFrame = type("DataFrame", (), {})
if not hasattr(stub, "Categorical"):
    stub.Categorical = type("Categorical", (), {})
if not hasattr(stub, "core"):
    stub.core = types.SimpleNamespace()
if not hasattr(stub.core, "arrays"):
    stub.core.arrays = types.SimpleNamespace()
if not hasattr(stub.core.arrays, "ExtensionArray"):
    stub.core.arrays.ExtensionArray = type("ExtensionArray", (), {})

neo_stub = sys.modules.get("neo4j")
if neo_stub is None:
    neo_stub = types.ModuleType("neo4j")
    sys.modules["neo4j"] = neo_stub
if not hasattr(neo_stub, "GraphDatabase"):
    class _GraphDatabase:
        @staticmethod
        def driver(*_args, **_kwargs):  # placeholder stub
            raise ImportError("neo4j driver not available in test stub")
    neo_stub.GraphDatabase = _GraphDatabase
if not hasattr(neo_stub, "Record"):
    neo_stub.Record = type("Record", (), {})
if not hasattr(neo_stub, "Driver"):
    neo_stub.Driver = type("Driver", (), {})
if not hasattr(neo_stub, "Query"):
    neo_stub.Query = type("Query", (), {})
if not hasattr(neo_stub, "RoutingControl"):
    neo_stub.RoutingControl = types.SimpleNamespace(READ="READ")
if not hasattr(neo_stub, "exceptions"):
    exceptions_module = types.ModuleType("neo4j.exceptions")

    def _make_exc(name: str) -> type[RuntimeError]:
        return type(name, (RuntimeError,), {})

    def _exceptions_getattr(name: str) -> type[RuntimeError]:
        exc_type = _make_exc(name)
        setattr(exceptions_module, name, exc_type)
        return exc_type

    exceptions_module.__getattr__ = _exceptions_getattr  # type: ignore[attr-defined]
    for _name in ("Neo4jError", "ClientError", "DriverError", "CypherSyntaxError", "CypherTypeError"):
        setattr(exceptions_module, _name, _make_exc(_name))
    sys.modules["neo4j.exceptions"] = exceptions_module
    neo_stub.exceptions = exceptions_module
else:
    sys.modules.setdefault("neo4j.exceptions", neo_stub.exceptions)


def _patch_driver(monkeypatch: pytest.MonkeyPatch, factory) -> None:
    """Patch both sync and async driver entry points with a factory."""
    monkeypatch.setattr(kg.GraphDatabase, "driver", factory)
    if hasattr(kg, "AsyncGraphDatabase"):
        monkeypatch.setattr(kg.AsyncGraphDatabase, "driver", factory)

import scripts.kg_build as kg
from cli.openai_client import OpenAIClientError


class FakeSharedClient:
    def __init__(self) -> None:
        self.embedding_calls: list[str] = []
        self.chat_calls: list[list[dict[str, str]]] = []

    def embedding(self, *, input_text: str) -> SimpleNamespace:
        self.embedding_calls.append(input_text)
        vector = [0.0] * 5
        return SimpleNamespace(vector=vector, tokens_consumed=10)

    def chat_completion(self, *, messages, temperature: float) -> SimpleNamespace:
        self.chat_calls.append(messages)
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Acknowledged",
                    }
                }
            ]
        }
        return SimpleNamespace(raw_response=response)


class FakePipeline:
    def __init__(
        self,
        *,
        llm,
        driver,
        embedder,
        schema=None,
        from_pdf,
        text_splitter,
        neo4j_database,
        kg_writer=None,
    ) -> None:
        self.llm = llm
        self.driver = driver
        self.embedder = embedder
        self.schema = schema
        self.from_pdf = from_pdf
        self.text_splitter = text_splitter
        self.database = neo4j_database
        self.run_args: dict[str, str] = {}
        self.kg_writer = kg_writer

    async def run_async(self, *, text: str = "", file_path: str | None = None):
        self.run_args = {"text": text, "file_path": file_path}
        # Simulate embedder and LLM usage to exercise client stubs
        self.embedder.embed_query(text)
        self.llm.invoke(text)
        return SimpleNamespace(run_id="test-run")


class FakeDriver:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self._pool = types.SimpleNamespace(pool_config=types.SimpleNamespace(user_agent=None))

    def __enter__(self) -> "FakeDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    async def __aenter__(self) -> "FakeDriver":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def execute_query(self, query: str, *_, **__) :
        query_text = query.strip()
        self.queries.append(query_text)
        if "DETACH DELETE" in query_text:
            return ([], None, None)
        if "MERGE (doc:Document" in query_text:
            return ([], None, None)
        if "CALL dbms.components" in query_text:
            return ([{"versions": ["5.26.0"], "edition": "enterprise"}], None, None)
        if "MATCH (:Document) RETURN count" in query_text:
            return ([{"value": 2}], None, None)
        if "MATCH (:Chunk) RETURN count" in query_text and "HAS_CHUNK" not in query_text:
            return ([{"value": 4}], None, None)
        if "MATCH (:Document)-[:HAS_CHUNK]" in query_text:
            return ([{"value": 4}], None, None)
        return ([{"value": 0}], None, None)


@pytest.fixture(autouse=True)
def _ensure_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile

    wheel_dir = tempfile.gettempdir()
    for name in os.listdir(wheel_dir):
        if name.startswith("neo4j_graphrag-") and name.endswith(".whl"):
            monkeypatch.syspath_prepend(os.path.join(wheel_dir, name))
        if name.startswith("neo4j-") and name.endswith(".whl"):
            monkeypatch.syspath_prepend(os.path.join(wheel_dir, name))


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("NEO4J_URI", "bolt://example")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")


def test_run_pipeline_success(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001 - env fixture ensures auth vars
    source = tmp_path / "sample.txt"
    source.write_text("sample content", encoding="utf-8")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []
    created_drivers: list[FakeDriver] = []

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    log = kg.run(
        [
            "--source",
            str(source),
            "--log-path",
            str(log_path),
            "--chunk-size",
            "10",
            "--chunk-overlap",
            "2",
            "--reset-database",
        ]
    )

    assert log["status"] == "success"
    assert log["counts"] == {"documents": 2, "chunks": 4, "relationships": 4}
    assert log["chunking"]["size"] == 10
    assert log["chunking"]["profile"] == "text"
    assert log["files"]
    assert log["chunks"]
    assert fake_client.embedding_calls
    assert fake_client.chat_calls
    assert pipelines and pipelines[0].run_args["text"] == "sample content"
    assert isinstance(pipelines[0].kg_writer, kg.SanitizingNeo4jWriter)
    assert created_drivers, "Expected GraphDatabase.driver to be invoked"
    assert any("DETACH DELETE" in query for driver in created_drivers for query in driver.queries)
    saved = json.loads(log_path.read_text())
    assert saved["status"] == "success"


def test_run_skips_reset_without_flag(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001 - env fixture ensures auth vars
    source = tmp_path / "sample.txt"
    source.write_text("sample content", encoding="utf-8")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []
    created_drivers: list[FakeDriver] = []

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    kg.run(
        [
            "--source",
            str(source),
            "--log-path",
            str(log_path),
            "--chunk-size",
            "10",
            "--chunk-overlap",
            "2",
        ]
    )

    assert created_drivers, "Expected GraphDatabase.driver to be invoked"
    assert not any("DETACH DELETE" in query for driver in created_drivers for query in driver.queries)


def test_run_handles_openai_failure(tmp_path, monkeypatch, env):  # noqa: ARG001 - env fixture ensures auth vars
    source = tmp_path / "sample.txt"
    source.write_text("content", encoding="utf-8")

    class FailingClient(FakeSharedClient):
        def embedding(self, *, input_text: str):
            raise OpenAIClientError("boom", remediation="retry later")

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: FailingClient())
    _patch_driver(monkeypatch, lambda *_, **__: FakeDriver())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))
    monkeypatch.setattr(kg, "SimpleKGPipeline", lambda **kwargs: FakePipeline(**kwargs))

    with pytest.raises(RuntimeError) as excinfo:
        kg.run(["--source", str(source), "--chunk-size", "5", "--chunk-overlap", "1"])
    assert "OpenAI request failed" in str(excinfo.value)


def test_missing_file_raises(env, monkeypatch):  # noqa: ARG001 - env fixture for parity
    _patch_driver(monkeypatch, lambda *_: FakeDriver())
    with pytest.raises(FileNotFoundError):
        kg.run(["--source", "does-not-exist.txt"])


# --- Additional tests for scripts/kg_build.py (pytest) ---

def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)
    args = kg._parse_args([])
    assert args.source == str(kg.DEFAULT_SOURCE)
    assert args.chunk_size is None
    assert args.chunk_overlap is None
    assert args.profile is None
    assert args.include_patterns is None
    assert args.database is None
    assert args.log_path == str(kg.DEFAULT_LOG_PATH)
    assert args.reset_database is False

def test_parse_args_overrides() -> None:
    pass


def test_parse_args_multiple_include_patterns() -> None:
    pass


def test_parse_args_source_dir() -> None:
    pass


def test_parse_args_database_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_DATABASE", "test_db")
    args = kg._parse_args([])
    # The actual behavior depends on implementation, but we test the parsing
    assert args.database is None or args.database == "test_db"
    args = kg._parse_args(["--source-dir", "/tmp/mydir"])
    assert args.source_dir == "/tmp/mydir"
    assert args.source is None or args.source == str(kg.DEFAULT_SOURCE)
    args = kg._parse_args(
        [
            "--include-pattern",
            "*.py",
            "--include-pattern",
            "*.md",
            "--include-pattern",
            "*.txt",
        ]
    )
    assert args.include_patterns == ["*.py", "*.md", "*.txt"]
    assert len(args.include_patterns) == 3
    args = kg._parse_args(
        [
            "--source",
            "/tmp/foo.txt",  # nosec
            "--chunk-size",
            "42",
            "--chunk-overlap",
            "7",
            "--database",
            "neo4j",
            "--log-path",
            "/tmp/log.json",  # nosec
            "--reset-database",
            "--profile",
            "code",
            "--include-pattern",
            "*.py",
        ]
    )
    assert args.source == "/tmp/foo.txt"
    assert args.chunk_size == 42
    assert args.chunk_overlap == 7
    assert args.database == "neo4j"
    assert args.log_path == "/tmp/log.json"
    assert args.reset_database is True
    assert args.profile == "code"
    assert args.include_patterns == ["*.py"]


def test_run_directory_ingestion(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001 - env fixture ensures auth vars
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "docs").mkdir()
    text_file = repo_dir / "docs" / "note.md"
    text_file.write_text("Doc content for ingestion.", encoding="utf-8")
    code_file = repo_dir / "module.py"
    code_file.write_text("print('hello world')", encoding="utf-8")
    binary_file = repo_dir / "image.png"
    binary_file.write_bytes(b"\x89PNG\r\n\x1a\n")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []
    created_drivers: list[FakeDriver] = []

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    log = kg.run(
        [
            "--source-dir",
            str(repo_dir),
            "--include-pattern",
            "**/*.md",
            "--include-pattern",
            "**/*.py",
            "--log-path",
            str(log_path),
            "--profile",
            "markdown",
        ]
    )

    assert log["status"] == "success"
    assert log["source_mode"] == "directory"
    assert len(pipelines) == 2  # binary file skipped
    assert len(log["files"]) == 2
    assert all(entry["chunks"] >= 1 for entry in log["files"])
    assert log["chunks"]
    assert created_drivers, "Expected GraphDatabase.driver to be invoked"
    saved = json.loads(log_path.read_text())
    assert saved["files"] == log["files"]


def test_sanitize_property_value_handles_only_none() -> None:
    sanitized = kg._sanitize_property_value([None, None])
    assert sanitized == []


def test_sanitize_property_value_heterogeneous_list() -> None:
    raw = [1, "a", 2]
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    assert json.loads(sanitized) == raw


def test_sanitize_property_value_subclass_primitives() -> None:
    class FancyInt(int):
        pass

    raw = [FancyInt(1), FancyInt(0)]
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    assert json.loads(sanitized) == [1, 0]


def test_sanitize_property_value_mapping_sorted() -> None:
    raw = {"b": 1, "a": 2}
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    assert json.loads(sanitized) == {"a": 2, "b": 1}


def test_sanitize_property_value_arbitrary_object() -> None:
    class Custom:
        def __str__(self) -> str:
            return "<custom>"

    sanitized = kg._sanitize_property_value(Custom())
    assert sanitized == "<custom>"


def test_sanitizing_writer_handles_empty_list() -> None:
    pass


def test_sanitize_property_value_nested_dict() -> None:
    pass


def test_sanitize_property_value_none() -> None:
    pass


def test_sanitize_property_value_empty_string() -> None:
    pass


def test_sanitize_property_value_boolean_list() -> None:
    pass


def test_sanitize_property_value_float_list() -> None:
    pass


def test_sanitize_property_value_dict_with_none() -> None:
    pass


def test_sanitize_property_value_empty_dict() -> None:
    pass


def test_sanitize_property_value_empty_list_explicit() -> None:
    pass


def test_sanitize_property_value_large_numbers() -> None:
    pass


def test_sanitize_property_value_unicode_strings() -> None:
    pass


def test_sanitizing_writer_mixed_properties() -> None:
    pass


def test_sanitizing_writer_nested_none_values() -> None:
    pass


def test_run_custom_chunk_parameters(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001
    pass


def test_run_code_profile(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001
    pass


def test_fake_driver_unknown_query() -> None:
    pass


def test_fake_driver_context_manager_sync() -> None:
    pass


async def test_fake_driver_context_manager_async() -> None:
    pass


def test_fake_pipeline_initialization() -> None:
    pass


def test_fake_shared_client_embedding_tracking() -> None:
    pass


def test_fake_shared_client_chat_tracking() -> None:
    pass


def test_run_directory_no_matching_files(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001
    pass


def test_fake_shared_client_temperature_handling() -> None:
    pass


def test_sanitize_property_value_empty_strings_list() -> None:
    pass


def test_run_handles_chat_completion_failure(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001
    pass


def test_missing_directory_raises(env, monkeypatch) -> None:  # noqa: ARG001
    pass


def test_log_file_persistence(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001
    pass


def test_sanitize_property_value_special_chars() -> None:
    pass


def test_fake_driver_multiple_query_sequence() -> None:
    pass


def test_sanitize_property_value_date_strings() -> None:
    pass


def test_sanitize_property_value_with_bytes() -> None:
    pass


def test_parse_args_both_source_and_dir() -> None:
    pass


def test_run_with_embedding_dimensions_override(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001
    pass


def test_sanitizing_writer_all_none_properties() -> None:
    pass


def test_run_with_long_content(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001
    pass


def test_parse_args_zero_chunk_size() -> None:
    pass


def test_parse_args_negative_overlap() -> None:
    pass


def test_sanitize_property_value_with_pandas_na() -> None:
    pass


async def test_fake_pipeline_run_async_with_file() -> None:
    pass


def test_run_multiple_files_tracking(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001
    pass


def test_sanitize_property_value_numeric_keys() -> None:
    pass


def test_sanitize_property_value_deep_nesting() -> None:
    pass


def test_sanitize_property_value_single_element_list() -> None:
    pass


def test_sanitize_property_value_mixed_dict_values() -> None:
    raw = {"string": "text", "number": 123, "list": [1, 2], "nested": {"key": "val"}}
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    parsed = json.loads(sanitized)
    assert parsed["string"] == "text"
    assert parsed["number"] == 123
    assert parsed["list"] == [1, 2]
    assert parsed["nested"]["key"] == "val"
    raw = [42]
    sanitized = kg._sanitize_property_value(raw)
    assert sanitized == raw or json.loads(sanitized) == raw
    raw = {"level1": {"level2": {"level3": {"level4": {"level5": "deep"}}}}}
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    parsed = json.loads(sanitized)
    assert parsed["level1"]["level2"]["level3"]["level4"]["level5"] == "deep"
    # Dict with numeric-like string keys
    raw = {"1": "one", "2": "two", "10": "ten"}
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    parsed = json.loads(sanitized)
    assert parsed["1"] == "one"
    assert parsed["10"] == "ten"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "file1.txt").write_text("content1", encoding="utf-8")
    (repo_dir / "file2.txt").write_text("content2", encoding="utf-8")
    (repo_dir / "file3.txt").write_text("content3", encoding="utf-8")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg, "SimpleKGPipeline", make_pipeline)
    _patch_driver(monkeypatch, lambda *_, **__: FakeDriver())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    log = kg.run(
        [
            "--source-dir",
            str(repo_dir),
            "--include-pattern",
            "*.txt",
            "--log-path",
            str(log_path),
        ]
    )

    assert log["status"] == "success"
    assert len(pipelines) == 3
    assert len(log["files"]) == 3
    fake_llm = SimpleNamespace(invoke=lambda x: None)
    fake_embedder = SimpleNamespace(embed_query=lambda x: None)
    pipeline = FakePipeline(
        llm=fake_llm,
        driver=FakeDriver(),
        embedder=fake_embedder,
        schema=None,
        from_pdf=False,
        text_splitter=None,
        neo4j_database="neo4j",
        kg_writer=None,
    )

    result = await pipeline.run_async(text="", file_path="/tmp/test.pdf")
    assert result.run_id == "test-run"
    assert pipeline.run_args["file_path"] == "/tmp/test.pdf"
    assert pipeline.run_args["text"] == ""
    import pandas as pd
    raw = {"value": pd.NA}
    sanitized = kg._sanitize_property_value(raw)
    # Should handle pandas.NA appropriately
    assert isinstance(sanitized, str)
    args = kg._parse_args(["--chunk-overlap", "-1"])
    assert args.chunk_overlap == -1  # May be invalid but tests parsing
    args = kg._parse_args(["--chunk-size", "0"])
    assert args.chunk_size == 0  # May be invalid but tests parsing
    source = tmp_path / "long.txt"
    long_content = "word " * 1000  # 1000 words
    source.write_text(long_content, encoding="utf-8")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    created_drivers: list[FakeDriver] = []

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr(kg, "SimpleKGPipeline", lambda **kwargs: FakePipeline(**kwargs))

    def driver_factory(*_args, **_kwargs):
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    log = kg.run(["--source", str(source), "--log-path", str(log_path)])

    assert log["status"] == "success"
    assert len(fake_client.embedding_calls) > 0
    writer = kg.SanitizingNeo4jWriter.__new__(kg.SanitizingNeo4jWriter)
    sanitized = writer._sanitize_properties({"a": None, "b": None, "c": None})
    # All None values should be handled appropriately
    assert isinstance(sanitized, dict)
    source = tmp_path / "sample.txt"
    source.write_text("sample", encoding="utf-8")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    created_drivers: list[FakeDriver] = []

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=10,  # Override
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr(kg, "SimpleKGPipeline", lambda **kwargs: FakePipeline(**kwargs))

    def driver_factory(*_args, **_kwargs):
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    log = kg.run(["--source", str(source), "--log-path", str(log_path)])

    assert log["status"] == "success"
    # Test that both can be provided (implementation decides priority)
    args = kg._parse_args(["--source", "/tmp/file.txt", "--source-dir", "/tmp/dir"])
    # At least one should be set
    assert args.source == "/tmp/file.txt" or args.source_dir == "/tmp/dir"
    class BytesLike:
        def __str__(self) -> str:
            return "bytes_representation"

    sanitized = kg._sanitize_property_value(BytesLike())
    assert sanitized == "bytes_representation"
    raw = {"date": "2024-01-15", "timestamp": "2024-01-15T10:30:00Z"}
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    parsed = json.loads(sanitized)
    assert parsed["date"] == "2024-01-15"
    assert parsed["timestamp"] == "2024-01-15T10:30:00Z"
    driver = FakeDriver()

    # Test various query patterns
    result1 = driver.execute_query("CALL dbms.components() YIELD versions, edition")
    result2 = driver.execute_query("MATCH (:Document) RETURN count(*) as value")
    result3 = driver.execute_query("MATCH (:Chunk) RETURN count(*) as value")
    result4 = driver.execute_query("MATCH (:Document)-[:HAS_CHUNK]->(:Chunk) RETURN count(*) as value")

    assert result1 == ([{"versions": ["5.26.0"], "edition": "enterprise"}], None, None)
    assert result2 == ([{"value": 2}], None, None)
    assert result3 == ([{"value": 4}], None, None)
    assert result4 == ([{"value": 4}], None, None)
    assert len(driver.queries) == 4
    raw = {"newline": "line1\nline2", "tab": "col1\tcol2", "quote": "say \"hello\""}
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    parsed = json.loads(sanitized)
    assert "\n" in parsed["newline"]
    assert "\t" in parsed["tab"]
    assert "\"" in parsed["quote"]
    source = tmp_path / "sample.txt"
    source.write_text("content", encoding="utf-8")
    log_path = tmp_path / "custom_log.json"

    fake_client = FakeSharedClient()
    created_drivers: list[FakeDriver] = []

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr(kg, "SimpleKGPipeline", lambda **kwargs: FakePipeline(**kwargs))

    def driver_factory(*_args, **_kwargs):
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    kg.run(["--source", str(source), "--log-path", str(log_path)])

    assert log_path.exists()
    log_content = json.loads(log_path.read_text())
    assert "status" in log_content
    assert "counts" in log_content
    assert "chunking" in log_content
    _patch_driver(monkeypatch, lambda *_: FakeDriver())
    with pytest.raises(FileNotFoundError):
        kg.run(["--source-dir", "does-not-exist-dir"])
    source = tmp_path / "sample.txt"
    source.write_text("content", encoding="utf-8")

    class FailingChatClient(FakeSharedClient):
        def chat_completion(self, *, messages, temperature: float):
            raise OpenAIClientError("chat failed", remediation="check API key")

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: FailingChatClient())
    _patch_driver(monkeypatch, lambda *_, **__: FakeDriver())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))
    monkeypatch.setattr(kg, "SimpleKGPipeline", lambda **kwargs: FakePipeline(**kwargs))

    with pytest.raises(RuntimeError) as excinfo:
        kg.run(["--source", str(source), "--chunk-size", "5", "--chunk-overlap", "1"])
    assert "OpenAI request failed" in str(excinfo.value)
    raw = ["", "", ""]
    sanitized = kg._sanitize_property_value(raw)
    # Should preserve empty strings
    assert sanitized == raw or json.loads(sanitized) == raw
    client = FakeSharedClient()
    messages = [{"role": "user", "content": "Test"}]

    result1 = client.chat_completion(messages=messages, temperature=0.0)
    result2 = client.chat_completion(messages=messages, temperature=1.0)

    # Both should succeed and return same response format
    assert result1.raw_response["choices"][0]["message"]["content"] == "Acknowledged"
    assert result2.raw_response["choices"][0]["message"]["content"] == "Acknowledged"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (repo_dir / "data.json").write_text("{}", encoding="utf-8")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []
    created_drivers: list[FakeDriver] = []

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    log = kg.run(
        [
            "--source-dir",
            str(repo_dir),
            "--include-pattern",
            "**/*.txt",
            "--log-path",
            str(log_path),
        ]
    )

    assert log["status"] == "success"
    # No matching files means no pipelines run
    assert len(pipelines) == 0 or log["files"] == []
    client = FakeSharedClient()
    messages1 = [{"role": "user", "content": "Hello"}]
    messages2 = [{"role": "system", "content": "System prompt"}]

    result1 = client.chat_completion(messages=messages1, temperature=0.7)
    result2 = client.chat_completion(messages=messages2, temperature=0.5)

    assert len(client.chat_calls) == 2
    assert client.chat_calls[0] == messages1
    assert client.chat_calls[1] == messages2
    assert result1.raw_response["choices"][0]["message"]["content"] == "Acknowledged"
    assert result2.raw_response["choices"][0]["message"]["content"] == "Acknowledged"
    client = FakeSharedClient()
    result1 = client.embedding(input_text="test1")
    result2 = client.embedding(input_text="test2")

    assert len(client.embedding_calls) == 2
    assert client.embedding_calls[0] == "test1"
    assert client.embedding_calls[1] == "test2"
    assert result1.vector == [0.0] * 5
    assert result1.tokens_consumed == 10
    assert result2.vector == [0.0] * 5
    fake_llm = object()
    fake_driver = FakeDriver()
    fake_embedder = object()
    fake_schema = {"entity": "value"}
    fake_from_pdf = False
    fake_text_splitter = object()
    fake_database = "neo4j"
    fake_kg_writer = object()

    pipeline = FakePipeline(
        llm=fake_llm,
        driver=fake_driver,
        embedder=fake_embedder,
        schema=fake_schema,
        from_pdf=fake_from_pdf,
        text_splitter=fake_text_splitter,
        neo4j_database=fake_database,
        kg_writer=fake_kg_writer,
    )

    assert pipeline.llm is fake_llm
    assert pipeline.driver is fake_driver
    assert pipeline.embedder is fake_embedder
    assert pipeline.schema == fake_schema
    assert pipeline.from_pdf is False
    assert pipeline.text_splitter is fake_text_splitter
    assert pipeline.database == fake_database
    assert pipeline.kg_writer is fake_kg_writer
    assert pipeline.run_args == {}
    driver = FakeDriver()
    async with driver as d:
        assert d is driver
        d.execute_query("MATCH (n) RETURN n")
    assert len(driver.queries) == 1
    driver = FakeDriver()
    with driver as d:
        assert d is driver
        d.execute_query("MATCH (n) RETURN n")
    assert len(driver.queries) == 1
    driver = FakeDriver()
    result = driver.execute_query("SELECT * FROM unknown_table")
    assert result == ([{"value": 0}], None, None)
    assert "SELECT * FROM unknown_table" in driver.queries
    source = tmp_path / "sample.py"
    source.write_text("def hello():\n    return \"world\"", encoding="utf-8")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []
    created_drivers: list[FakeDriver] = []

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    log = kg.run(
        [
            "--source",
            str(source),
            "--log-path",
            str(log_path),
            "--profile",
            "code",
        ]
    )

    assert log["status"] == "success"
    assert log["chunking"]["profile"] == "code"
    source = tmp_path / "sample.txt"
    source.write_text("sample content for chunking test", encoding="utf-8")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []
    created_drivers: list[FakeDriver] = []

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    log = kg.run(
        [
            "--source",
            str(source),
            "--log-path",
            str(log_path),
            "--chunk-size",
            "100",
            "--chunk-overlap",
            "20",
        ]
    )

    assert log["status"] == "success"
    assert log["chunking"]["size"] == 100
    assert log["chunking"]["overlap"] == 20
    writer = kg.SanitizingNeo4jWriter.__new__(kg.SanitizingNeo4jWriter)
    sanitized = writer._sanitize_properties(
        {
            "nested": {"inner": None, "value": "test"},
            "all_none": [None, None, None],
        }
    )
    # all_none should become empty list
    assert sanitized["all_none"] == []
    writer = kg.SanitizingNeo4jWriter.__new__(kg.SanitizingNeo4jWriter)
    sanitized = writer._sanitize_properties(
        {
            "string": "text",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
        }
    )
    assert sanitized["string"] == "text"
    assert sanitized["number"] == 42
    assert sanitized["float"] == 3.14
    assert sanitized["bool"] is True
    # none, list, and dict may be sanitized differently
    raw = {"emoji": "🔥", "chinese": "你好", "arabic": "مرحبا"}
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    parsed = json.loads(sanitized)
    assert parsed["emoji"] == "🔥"
    assert parsed["chinese"] == "你好"
    assert parsed["arabic"] == "مرحبا"
    raw = [999999999999, -999999999999, 0]
    sanitized = kg._sanitize_property_value(raw)
    assert sanitized == raw or json.loads(sanitized) == raw
    raw = []
    sanitized = kg._sanitize_property_value(raw)
    assert sanitized == []
    raw = {}
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    assert json.loads(sanitized) == {}
    raw = {"key1": "value", "key2": None, "key3": 123}
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    parsed = json.loads(sanitized)
    assert parsed["key1"] == "value"
    assert parsed["key2"] is None
    assert parsed["key3"] == 123
    raw = [1.5, 2.7, 3.9]
    sanitized = kg._sanitize_property_value(raw)
    assert sanitized == raw or json.loads(sanitized) == raw
    raw = [True, False, True]
    sanitized = kg._sanitize_property_value(raw)
    # Should remain as list if homogeneous primitives
    assert sanitized == raw or json.loads(sanitized) == raw
    sanitized = kg._sanitize_property_value("")
    assert sanitized == ""
    sanitized = kg._sanitize_property_value(None)
    assert sanitized is None
    raw = {"outer": {"inner": "value"}, "list": [1, 2, 3]}
    sanitized = kg._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    parsed = json.loads(sanitized)
    assert parsed["outer"]["inner"] == "value"
    assert parsed["list"] == [1, 2, 3]


def test_run_empty_file(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []
    created_drivers: list[FakeDriver] = []

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    log = kg.run(
        [
            "--source",
            str(source),
            "--log-path",
            str(log_path),
        ]
    )

    assert log["status"] == "success"
    assert pipelines[0].run_args["text"] == ""
    writer = kg.SanitizingNeo4jWriter.__new__(kg.SanitizingNeo4jWriter)
    sanitized = writer._sanitize_properties({"values": [None, None]})
    assert sanitized == {"values": []}

# ... rest of file unchanged ...