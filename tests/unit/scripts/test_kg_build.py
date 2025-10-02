from __future__ import annotations

import asyncio
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

import scripts.kg_build as kg  # noqa: E402
from cli.openai_client import OpenAIClientError  # noqa: E402


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
        """
        Initialize the FakeDriver test double with default state used by unit tests.
        
        Creates:
        - queries: list of executed Cypher query texts.
        - _pool.pool_config.user_agent: placeholder for driver user agent.
        - qa_missing_embeddings: count of chunks missing embeddings for QA.
        - qa_orphan_chunks: count of orphaned chunks for QA.
        - qa_checksum_mismatches: count of checksum mismatches for QA.
        - graph_counts: dictionary providing default counts for "documents", "chunks", and "relationships".
        """
        self.queries: list[str] = []
        self._pool = types.SimpleNamespace(pool_config=types.SimpleNamespace(user_agent=None))
        self.qa_missing_embeddings = 0
        self.qa_orphan_chunks = 0
        self.qa_checksum_mismatches = 0
        self.graph_counts = {
            "documents": 2,
            "chunks": 4,
            "relationships": 4,
        }

    def __enter__(self) -> "FakeDriver":
        """
        Enter the context manager and return the FakeDriver instance.
        
        Returns:
            FakeDriver: The FakeDriver instance being managed.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    async def __aenter__(self) -> "FakeDriver":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """
        Exit the asynchronous context for the driver.
        
        Parameters:
            exc_type (type | None): Exception type if raised inside the context, otherwise None.
            exc (BaseException | None): Exception instance if raised inside the context, otherwise None.
            tb (types.TracebackType | None): Traceback object if an exception was raised, otherwise None.
        """
        return None

    def execute_query(self, query: str, parameters=None, **kwargs):
        """
        Simulate executing a Cypher query against the fake Neo4j driver and return a synthetic result set.
        
        Parameters:
            query (str): The Cypher query string to evaluate; matching substrings determine which synthetic response to return.
            parameters (dict | None): Optional query parameters (unused by response logic but accepted for signature compatibility).
        
        Returns:
            tuple: A 3-tuple (rows, summary, statistics) where `rows` is a list of dictionaries representing query results (often with a `"value"` key for counts), and `summary` and `statistics` are `None` placeholders.
        """
        query_text = query.strip()
        self.queries.append(query_text)
        if "DETACH DELETE" in query_text:
            return ([], None, None)
        if "ingest_run_key" in query_text:
            return ([], None, None)
        if "MERGE (doc:Document" in query_text:
            return ([], None, None)
        if "CALL dbms.components" in query_text:
            return ([{"versions": ["5.26.0"], "edition": "enterprise"}], None, None)
        if "c.embedding" in query_text:
            return ([{"value": self.qa_missing_embeddings}], None, None)
        if "NOT ( (:Document)-[:HAS_CHUNK]->(c) )" in query_text:
            return ([{"value": self.qa_orphan_chunks}], None, None)
        if "coalesce(c.checksum" in query_text:
            return ([{"value": self.qa_checksum_mismatches}], None, None)
        if "MATCH (:Document) RETURN count" in query_text:
            return ([{"value": self.graph_counts.get("documents", 0)}], None, None)
        if "MATCH (:Chunk) RETURN count" in query_text and "HAS_CHUNK" not in query_text:
            return ([{"value": self.graph_counts.get("chunks", 0)}], None, None)
        if "MATCH (:Document)-[:HAS_CHUNK]" in query_text:
            return ([{"value": self.graph_counts.get("relationships", 0)}], None, None)
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
    qa_dir = tmp_path / "qa"
    qa_dir = tmp_path / "qa"
    qa_dir = tmp_path / "qa"
    qa_dir = tmp_path / "qa"
    qa_dir = tmp_path / "qa"
    qa_dir = tmp_path / "qa"

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
            "--qa-report-dir",
            str(qa_dir),
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
    assert "qa" in log
    qa_section = log["qa"]
    assert qa_section["status"] == "pass"
    assert qa_section["report_version"] == kg.QA_REPORT_VERSION
    assert qa_section["duration_ms"] >= 0
    assert "qa_evaluation_ms" in qa_section["metrics"]
    def _resolve_report(path_str: str) -> pathlib.Path:
        """
        Resolve a path string to a pathlib.Path, preferring an existing absolute path, then a repository-root-relative path, and finally a root-relative path.
        
        Parameters:
            path_str (str): Path string provided by the caller; may be absolute or relative.
        
        Returns:
            pathlib.Path: The chosen Path. If an absolute existing path matching `path_str` is found it is returned; otherwise the path is resolved relative to the repository root (if available) and returned if it exists; if neither exists, a root-relative Path is returned.
        """
        candidate = pathlib.Path(path_str)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        repo_candidate = (kg._resolve_repo_root() or pathlib.Path.cwd()) / candidate
        if repo_candidate.exists():
            return repo_candidate
        root_candidate = pathlib.Path("/") / candidate
        return root_candidate

    report_json_path = _resolve_report(qa_section["report_json"])
    assert report_json_path.exists()
    report_md_path = _resolve_report(qa_section["report_markdown"])
    assert report_md_path.exists()
    json_payload = json.loads(report_json_path.read_text())
    serialized = json.dumps(json_payload)
    assert "/tmp/" not in serialized
    md_payload = report_md_path.read_text()
    assert "/tmp/" not in md_payload


def test_run_skips_reset_without_flag(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001 - env fixture ensures auth vars
    source = tmp_path / "sample.txt"
    source.write_text("sample content", encoding="utf-8")
    log_path = tmp_path / "log.json"
    qa_dir = tmp_path / "qa"

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
            "--qa-report-dir",
            str(qa_dir),
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
        kg.run(
            [
                "--source",
                str(source),
                "--chunk-size",
                "5",
                "--chunk-overlap",
                "1",
                "--qa-report-dir",
                str(tmp_path / "qa"),
            ]
        )
    assert "OpenAI request failed" in str(excinfo.value)


def test_run_fails_on_qa_threshold(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001 - env fixture ensures auth vars
    source = tmp_path / "sample.txt"
    source.write_text("qa failure content", encoding="utf-8")

    fake_client = FakeSharedClient()
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

    failing_driver = FakeDriver()
    failing_driver.qa_missing_embeddings = 2

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr(kg, "SimpleKGPipeline", lambda **kwargs: FakePipeline(**kwargs))
    _patch_driver(monkeypatch, lambda *_, **__: failing_driver)
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda *_, **__: settings))

    with pytest.raises(RuntimeError) as excinfo:
        kg.run(
            [
                "--source",
                str(source),
                "--qa-report-dir",
                str(tmp_path / "qa"),
            ]
        )

    assert "Ingestion QA gating failed" in str(excinfo.value)
    # Ensure rollback attempted
    assert any("DETACH DELETE" in query for query in failing_driver.queries)
    assert any("rel.ingest_run_key" in query for query in failing_driver.queries)
    assert any(
        "node.ingest_run_key" in query and "DETACH DELETE" in query
        for query in failing_driver.queries
    )


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
    assert args.qa_report_dir == str(kg.DEFAULT_QA_DIR)
    assert args.qa_max_missing_embeddings == 0
    assert args.qa_max_orphan_chunks == 0
    assert args.qa_max_checksum_mismatches == 0

def test_parse_args_database_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that _parse_args picks up the NEO4J_DATABASE environment variable and sets args.database accordingly.
    """
    monkeypatch.setenv("NEO4J_DATABASE", "test_db")
    args = kg._parse_args([])
    assert args.database == "test_db"
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)


def test_parse_args_source_dir_option(tmp_path: pathlib.Path) -> None:
    source_dir = tmp_path / "input"
    args = kg._parse_args(["--source-dir", str(source_dir)])
    assert args.source_dir == str(source_dir)
    assert args.source is None or args.source == str(kg.DEFAULT_SOURCE)


def test_parse_args_include_patterns() -> None:
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


def test_parse_args_overrides(tmp_path: pathlib.Path) -> None:
    src_file = tmp_path / "foo.txt"
    log_file = tmp_path / "log.json"
    args = kg._parse_args(
        [
            "--source",
            str(src_file),
            "--chunk-size",
            "42",
            "--chunk-overlap",
            "7",
            "--database",
            "neo4j",
        "--log-path",
        str(log_file),
        "--reset-database",
        "--profile",
        "code",
        "--qa-report-dir",
        str(tmp_path / "qa"),
        "--qa-max-missing-embeddings",
        "5",
        "--qa-max-orphan-chunks",
        "3",
        "--qa-max-checksum-mismatches",
        "2",
        "--include-pattern",
        "*.py",
    ]
)
    assert args.source == str(src_file)
    assert args.chunk_size == 42
    assert args.chunk_overlap == 7
    assert args.database == "neo4j"
    assert args.log_path == str(log_file)
    assert args.reset_database is True
    assert args.profile == "code"
    assert args.include_patterns == ["*.py"]
    assert args.qa_report_dir == str(tmp_path / "qa")
    assert args.qa_max_missing_embeddings == 5
    assert args.qa_max_orphan_chunks == 3
    assert args.qa_max_checksum_mismatches == 2


def test_run_directory_ingestion(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001 - env fixture ensures auth vars
    """
    Verifies that directory ingestion picks up specified file types, skips binary files, and records chunking and QA results.
    
    Creates a temporary repository with a markdown file, a Python file, and a binary file, runs the ingestion with include patterns for .md and .py, and asserts that:
    - the run completes successfully with source_mode "directory",
    - only the text files are ingested (binary skipped) and each file produces at least one chunk,
    - pipeline instances and a Neo4j driver are created,
    - the log is written to the provided log path and matches the in-memory log structure,
    - a QA section is present in the log with status "pass".
    """
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
    qa_dir = tmp_path / "qa"

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
            "--qa-report-dir",
            str(qa_dir),
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
    assert "qa" in log
    qa_section = log["qa"]
    assert qa_section["status"] == "pass"


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


def test_splitter_cache_scoped_per_source() -> None:
    splitter = kg.CachingFixedSizeSplitter(chunk_size=200, chunk_overlap=0)
    text = "identical content across files"

    with splitter.scoped("first-file"):
        result_a = asyncio.run(splitter.run(text))
        cached_a = splitter.get_cached(text)

    with splitter.scoped("second-file"):
        result_b = asyncio.run(splitter.run(text))
        cached_b = splitter.get_cached(text)

    assert cached_a is result_a
    assert cached_b is result_b
    assert result_a is not result_b
    uid_a = {chunk.uid for chunk in result_a.chunks}
    uid_b = {chunk.uid for chunk in result_b.chunks}
    assert uid_a.isdisjoint(uid_b)


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


def test_sanitizing_writer_drops_none_values() -> None:
    writer = kg.SanitizingNeo4jWriter.__new__(kg.SanitizingNeo4jWriter)
    sanitized = writer._sanitize_properties({"values": [None, None]})
    assert sanitized == {"values": []}
