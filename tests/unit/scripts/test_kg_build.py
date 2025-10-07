from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import types
from types import SimpleNamespace
from typing import Any

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
    for module in (kg_pipeline, kg):
        graph_db = getattr(module, "GraphDatabase", None)
        if graph_db is not None:
            monkeypatch.setattr(graph_db, "driver", factory)
        async_graph_db = getattr(module, "AsyncGraphDatabase", None)
        if async_graph_db is not None:
            monkeypatch.setattr(async_graph_db, "driver", factory)

import fancyrag.cli.kg_build_main as kg
import fancyrag.kg.pipeline as kg_pipeline
from fancyrag.splitters import CachingFixedSizeSplitter
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


def _make_settings_loader(openai_settings, neo4j_settings):
    """Produce a callable mirroring ``get_settings`` with requirement checks."""

    def _loader(*, require=None, **_kwargs):
        if require is not None:
            required = set(require)
            assert {"openai", "neo4j"}.issubset(required)
        return SimpleNamespace(openai=openai_settings, neo4j=neo4j_settings)

    return _loader


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
        self.semantic_nodes = 0
        self.semantic_relationships = 0
        self.semantic_orphans = 0

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
        if "n.semantic_source" in query_text and "AND NOT" in query_text:
            return ([{"value": self.semantic_orphans}], None, None)
        if "n.semantic_source" in query_text:
            return ([{"value": self.semantic_nodes}], None, None)
        if "r.semantic_source" in query_text:
            return ([{"value": self.semantic_relationships}], None, None)
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

    settings = kg_pipeline.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg_pipeline, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        """
        Create and register a FakePipeline instance using the provided constructor keyword arguments.
        
        The new FakePipeline is appended to the module-level `pipelines` list as a side effect.
        
        Parameters:
            **kwargs (dict): Keyword arguments forwarded to FakePipeline constructor.
        
        Returns:
            FakePipeline: The created and registered FakePipeline instance.
        """
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg_pipeline, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        """
        Create a new FakeDriver, append it to the created_drivers list, and return it.
        
        Parameters:
            *_args: Positional arguments are accepted and ignored.
            **_kwargs: Keyword arguments are accepted and ignored.
        
        Returns:
            driver (FakeDriver): The newly created FakeDriver instance that was appended to created_drivers.
        """
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())

    neo4j_stub = SimpleNamespace(uri="bolt://localhost:7687", database=None)
    neo4j_stub.auth = lambda: ("neo4j", "password")
    monkeypatch.setattr(
        kg_pipeline,
        "_get_settings",
        _make_settings_loader(settings, neo4j_stub),
    )

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
    assert isinstance(pipelines[0].kg_writer, kg_pipeline.SanitizingNeo4jWriter)
    assert created_drivers, "Expected GraphDatabase.driver to be invoked"
    assert any("DETACH DELETE" in query for driver in created_drivers for query in driver.queries)
    saved = json.loads(log_path.read_text())
    assert saved["status"] == "success"
    assert "qa" in log
    qa_section = log["qa"]
    assert qa_section["status"] == "pass"
    assert qa_section["report_version"] == kg_pipeline.QA_REPORT_VERSION
    assert qa_section["duration_ms"] >= 0
    assert "qa_evaluation_ms" in qa_section["metrics"]
    def _resolve_report(path_str: str) -> pathlib.Path:
        """
        Select the filesystem path for a report by preferring an existing absolute path, then a repository-root-relative existing path, and finally a root-relative path.
        
        Parameters:
            path_str (str): Path string provided by the caller; may be absolute or relative.
        
        Returns:
            pathlib.Path: The chosen path. If an absolute existing path matching `path_str` exists it is returned; otherwise an existing path resolved relative to the repository root (if available) is returned; if neither exists, a root-relative Path is returned.
        """
        candidate = pathlib.Path(path_str)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        repo_candidate = (kg_pipeline._resolve_repo_root() or pathlib.Path.cwd()) / candidate
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
    """
    Verify that running the ingestion CLI without the reset-database flag completes without performing a database reset.
    
    Sets up a temporary source file, stubs the OpenAI client, pipeline factory, and Neo4j driver, and runs the CLI with chunking and QA options to confirm the run proceeds without issuing a DETACH DELETE reset query.
    """
    source = tmp_path / "sample.txt"
    source.write_text("sample content", encoding="utf-8")
    log_path = tmp_path / "log.json"
    qa_dir = tmp_path / "qa"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []
    created_drivers: list[FakeDriver] = []

    settings = kg_pipeline.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg_pipeline, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        """
        Create and register a FakePipeline instance using the provided constructor keyword arguments.
        
        The new FakePipeline is appended to the module-level `pipelines` list as a side effect.
        
        Parameters:
            **kwargs (dict): Keyword arguments forwarded to FakePipeline constructor.
        
        Returns:
            FakePipeline: The created and registered FakePipeline instance.
        """
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg_pipeline, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        """
        Create a new FakeDriver, append it to the created_drivers list, and return it.
        
        Parameters:
            *_args: Positional arguments are accepted and ignored.
            **_kwargs: Keyword arguments are accepted and ignored.
        
        Returns:
            driver (FakeDriver): The newly created FakeDriver instance that was appended to created_drivers.
        """
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())

    neo4j_stub = SimpleNamespace(uri="bolt://localhost:7687", database=None)
    neo4j_stub.auth = lambda: ("neo4j", "password")
    monkeypatch.setattr(
        kg_pipeline,
        "_get_settings",
        _make_settings_loader(settings, neo4j_stub),
    )

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


def test_run_with_semantic_enrichment(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001 - env fixture ensures auth vars
    source = tmp_path / "sample.txt"
    source.write_text("sample content", encoding="utf-8")
    log_path = tmp_path / "log.json"
    qa_dir = tmp_path / "qa"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []
    created_drivers: list[FakeDriver] = []

    settings = kg_pipeline.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg_pipeline, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        """
        Create and register a FakePipeline instance using the provided constructor keyword arguments.
        
        The new FakePipeline is appended to the module-level `pipelines` list as a side effect.
        
        Parameters:
            **kwargs (dict): Keyword arguments forwarded to FakePipeline constructor.
        
        Returns:
            FakePipeline: The created and registered FakePipeline instance.
        """
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg_pipeline, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        """
        Create and return a FakeDriver preconfigured with semantic counters and register it.
        
        This factory ignores any positional and keyword arguments. It constructs a FakeDriver,
        sets semantic_nodes to 12, semantic_relationships to 7, and semantic_orphans to 0,
        appends the instance to the module-level created_drivers list, and returns the instance.
        
        Returns:
            FakeDriver: The newly created and configured FakeDriver instance.
        """
        driver = FakeDriver()
        driver.semantic_nodes = 12
        driver.semantic_relationships = 7
        driver.semantic_orphans = 0
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())

    neo4j_stub = SimpleNamespace(uri="bolt://localhost:7687", database=None)
    neo4j_stub.auth = lambda: ("neo4j", "password")
    monkeypatch.setattr(
        kg_pipeline,
        "_get_settings",
        _make_settings_loader(settings, neo4j_stub),
    )

    semantic_calls: list[dict[str, Any]] = []

    def fake_semantic(**kwargs) -> kg_pipeline.SemanticEnrichmentStats:
        """
        Provide a deterministic SemanticEnrichmentStats for tests and record received keyword arguments.
        
        Parameters:
            kwargs: Any keyword arguments provided by the caller; they are recorded in the shared `semantic_calls` list and otherwise ignored.
        
        Returns:
            A SemanticEnrichmentStats instance with chunks_processed=2, chunk_failures=0, nodes_written=4, and relationships_written=3.
        """
        semantic_calls.append(kwargs)
        return kg_pipeline.SemanticEnrichmentStats(
            chunks_processed=2,
            chunk_failures=0,
            nodes_written=4,
            relationships_written=3,
        )

    monkeypatch.setattr(kg_pipeline, "_run_semantic_enrichment", fake_semantic)

    log = kg.run(
        [
            "--source",
            str(source),
            "--log-path",
            str(log_path),
            "--qa-report-dir",
            str(qa_dir),
            "--enable-semantic",
            "--semantic-max-concurrency",
            "3",
        ]
    )

    assert semantic_calls, "semantic enrichment helper was not invoked"
    assert semantic_calls[0]["max_concurrency"] == 3
    assert log["semantic"] == {
        "chunks_processed": 2,
        "chunk_failures": 0,
        "nodes_written": 4,
        "relationships_written": 3,
    }
    qa_metrics = log["qa"]["metrics"].get("semantic")
    assert qa_metrics is not None
    assert qa_metrics["chunks_processed"] == 2
    assert qa_metrics["chunk_failures"] == 0
    assert qa_metrics["nodes_written"] == 4
    assert qa_metrics["relationships_written"] == 3
    # database counters sourced from the fake driver
    assert qa_metrics["nodes_in_db"] == 12
    assert qa_metrics["relationships_in_db"] == 7
    assert qa_metrics["orphan_entities"] == 0
    assert created_drivers, "Expected GraphDatabase.driver to be invoked"
    assert not any("DETACH DELETE" in query for driver in created_drivers for query in driver.queries)


def test_run_handles_openai_failure(tmp_path, monkeypatch, env):  # noqa: ARG001 - env fixture ensures auth vars
    source = tmp_path / "sample.txt"
    source.write_text("content", encoding="utf-8")

    class FailingClient(FakeSharedClient):
        def embedding(self, *, input_text: str):
            """
            Simulate an embedding request that always fails by raising an OpenAIClientError.
            
            Parameters:
                input_text (str): The text that would be sent for embedding (recorded by callers).
            
            Raises:
                OpenAIClientError: Always raised with message "boom" and remediation "retry later".
            """
            raise OpenAIClientError("boom", remediation="retry later")

    settings = kg_pipeline.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg_pipeline, "SharedOpenAIClient", lambda *_args, **_kwargs: FailingClient())
    _patch_driver(monkeypatch, lambda *_, **__: FakeDriver())

    neo4j_stub = SimpleNamespace(uri="bolt://localhost:7687", database=None)
    neo4j_stub.auth = lambda: ("neo4j", "password")
    monkeypatch.setattr(
        kg_pipeline,
        "_get_settings",
        _make_settings_loader(settings, neo4j_stub),
    )
    monkeypatch.setattr(kg_pipeline, "SimpleKGPipeline", lambda **kwargs: FakePipeline(**kwargs))

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
    settings = kg_pipeline.OpenAISettings(
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

    monkeypatch.setattr(kg_pipeline, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)
    monkeypatch.setattr(kg_pipeline, "SimpleKGPipeline", lambda **kwargs: FakePipeline(**kwargs))
    _patch_driver(monkeypatch, lambda *_, **__: failing_driver)

    neo4j_stub = SimpleNamespace(uri="bolt://localhost:7687", database=None)
    neo4j_stub.auth = lambda: ("neo4j", "password")
    monkeypatch.setattr(
        kg_pipeline,
        "_get_settings",
        _make_settings_loader(settings, neo4j_stub),
    )

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
    Verify directory ingestion includes specified text file types, excludes binary files, and records chunking and QA results.

    Creates a temporary repository containing a Markdown file, a Python file, and a binary file, runs ingestion with include patterns for `**/*.md` and `**/*.py`, and asserts that the run completes with `source_mode` set to "directory", only the text files are ingested (binary skipped) with at least one chunk per file, pipeline instances and a Neo4j driver were created, the on-disk log matches the in-memory log structure, and the QA section reports `"status": "pass"`.
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

    settings = kg_pipeline.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg_pipeline, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        """
        Create and register a FakePipeline instance using the provided constructor keyword arguments.
        
        The new FakePipeline is appended to the module-level `pipelines` list as a side effect.
        
        Parameters:
            **kwargs (dict): Keyword arguments forwarded to FakePipeline constructor.
        
        Returns:
            FakePipeline: The created and registered FakePipeline instance.
        """
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg_pipeline, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        """
        Create a new FakeDriver, append it to the created_drivers list, and return it.
        
        Parameters:
            *_args: Positional arguments are accepted and ignored.
            **_kwargs: Keyword arguments are accepted and ignored.
        
        Returns:
            driver (FakeDriver): The newly created FakeDriver instance that was appended to created_drivers.
        """
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())

    neo4j_stub = SimpleNamespace(uri="bolt://localhost:7687", database=None)
    neo4j_stub.auth = lambda: ("neo4j", "password")
    monkeypatch.setattr(
        kg_pipeline,
        "_get_settings",
        _make_settings_loader(settings, neo4j_stub),
    )

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


def test_run_with_external_qa_report_dir(tmp_path, monkeypatch, env) -> None:  # noqa: ARG001 - env fixture ensures auth vars
    """Ensure qa_report_dir outside the repo produces absolute, on-disk report paths."""

    source = tmp_path / "input.txt"
    source.write_text("Example content", encoding="utf-8")
    qa_dir = tmp_path / "qa-outside"
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []
    created_drivers: list[FakeDriver] = []

    settings = kg_pipeline.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg_pipeline, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg_pipeline, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())

    neo4j_stub = SimpleNamespace(uri="bolt://localhost:7687", database=None)
    neo4j_stub.auth = lambda: ("neo4j", "password")
    monkeypatch.setattr(
        kg_pipeline,
        "_get_settings",
        _make_settings_loader(settings, neo4j_stub),
    )

    log = kg.run(
        [
            "--source",
            str(source),
            "--qa-report-dir",
            str(qa_dir),
            "--log-path",
            str(log_path),
        ]
    )

    qa_section = log["qa"]
    json_path = pathlib.Path(qa_section["report_json"])
    markdown_path = pathlib.Path(qa_section["report_markdown"])

    assert qa_section["status"] == "pass"
    assert json_path.is_absolute()
    assert markdown_path.is_absolute()
    assert json_path.exists()
    assert markdown_path.exists()
    assert json_path.is_relative_to(qa_dir)
    assert markdown_path.is_relative_to(qa_dir)
    assert pipelines, "Expected pipeline to run"
    assert created_drivers, "Expected driver factory to be used"


def test_sanitize_property_value_handles_only_none() -> None:
    sanitized = kg_pipeline._sanitize_property_value([None, None])
    assert sanitized == []


def test_sanitize_property_value_heterogeneous_list() -> None:
    raw = [1, "a", 2]
    sanitized = kg_pipeline._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    assert json.loads(sanitized) == raw


def test_sanitize_property_value_subclass_primitives() -> None:
    class FancyInt(int):
        pass

    raw = [FancyInt(1), FancyInt(0)]
    sanitized = kg_pipeline._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    assert json.loads(sanitized) == [1, 0]


def test_sanitize_property_value_mapping_sorted() -> None:
    raw = {"b": 1, "a": 2}
    sanitized = kg_pipeline._sanitize_property_value(raw)
    assert isinstance(sanitized, str)
    assert json.loads(sanitized) == {"a": 2, "b": 1}


def test_sanitize_property_value_arbitrary_object() -> None:
    class Custom:
        def __str__(self) -> str:
            """
            Return a concise string representation of the object for display.
            
            Returns:
                str: The fixed string "<custom>".
            """
            return "<custom>"

    sanitized = kg_pipeline._sanitize_property_value(Custom())
    assert sanitized == "<custom>"


def test_splitter_cache_scoped_per_source() -> None:
    splitter = CachingFixedSizeSplitter(chunk_size=200, chunk_overlap=0)
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
    """
    Verify running the ingestion pipeline on an empty file completes successfully and invokes the pipeline with empty text.
    """
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    pipelines: list[FakePipeline] = []
    created_drivers: list[FakeDriver] = []

    settings = kg_pipeline.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=3,
        backoff_seconds=0.5,
        enable_fallback=True,
    )

    monkeypatch.setattr(kg_pipeline, "SharedOpenAIClient", lambda *_args, **_kwargs: fake_client)

    def make_pipeline(**kwargs):
        """
        Create and register a FakePipeline instance using the provided constructor keyword arguments.
        
        The new FakePipeline is appended to the module-level `pipelines` list as a side effect.
        
        Parameters:
            **kwargs (dict): Keyword arguments forwarded to FakePipeline constructor.
        
        Returns:
            FakePipeline: The created and registered FakePipeline instance.
        """
        pipeline = FakePipeline(**kwargs)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(kg_pipeline, "SimpleKGPipeline", make_pipeline)

    def driver_factory(*_args, **_kwargs):
        """
        Create a new FakeDriver, append it to the created_drivers list, and return it.
        
        Parameters:
            *_args: Positional arguments are accepted and ignored.
            **_kwargs: Keyword arguments are accepted and ignored.
        
        Returns:
            driver (FakeDriver): The newly created FakeDriver instance that was appended to created_drivers.
        """
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, lambda *_, **__: driver_factory())

    neo4j_stub = SimpleNamespace(uri="bolt://localhost:7687", database=None)
    neo4j_stub.auth = lambda: ("neo4j", "password")
    monkeypatch.setattr(
        kg_pipeline,
        "_get_settings",
        _make_settings_loader(settings, neo4j_stub),
    )

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
    writer = kg_pipeline.SanitizingNeo4jWriter.__new__(kg_pipeline.SanitizingNeo4jWriter)
    sanitized = writer._sanitize_properties({"values": [None, None]})
    assert sanitized == {"values": []}
# Testing library/framework: pytest (project-wide standard). These tests follow existing conventions.

class TestFakeDriver:
    """Comprehensive tests for the FakeDriver test double."""

    def test_initialization_sets_default_state(self):
        driver = FakeDriver()
        assert driver.queries == []
        assert driver._pool.pool_config.user_agent is None
        assert driver.qa_missing_embeddings == 0
        assert driver.qa_orphan_chunks == 0
        assert driver.qa_checksum_mismatches == 0
        assert driver.graph_counts == {"documents": 2, "chunks": 4, "relationships": 4}
        assert driver.semantic_nodes == 0
        assert driver.semantic_relationships == 0
        assert driver.semantic_orphans == 0

    def test_context_manager_sync(self):
        driver = FakeDriver()
        with driver as d:
            assert d is driver
            d.queries.append("test")
        assert driver.queries == ["test"]

    def test_context_manager_async(self):
        driver = FakeDriver()
        async def _run():
            async with driver as d:
                assert d is driver
                d.queries.append("async_test")
        asyncio.run(_run())
        assert driver.queries == ["async_test"]

    def test_context_manager_handles_exceptions(self):
        driver = FakeDriver()
        try:
            with driver as d:
                d.queries.append("before_error")
                raise ValueError("test error")
        except ValueError:
            pass
        assert driver.queries == ["before_error"]

    def test_async_context_manager_handles_exceptions(self):
        driver = FakeDriver()
        async def _run():
            try:
                async with driver as d:
                    d.queries.append("async_before_error")
                    raise ValueError("async test error")
            except ValueError:
                pass
        asyncio.run(_run())
        assert driver.queries == ["async_before_error"]

    def test_execute_query_detach_delete(self):
        driver = FakeDriver()
        result = driver.execute_query("DETACH DELETE n")
        assert driver.queries == ["DETACH DELETE n"]
        assert result == ([], None, None)

    def test_execute_query_ingest_run_key(self):
        driver = FakeDriver()
        result = driver.execute_query("MATCH (n) WHERE n.ingest_run_key = $key")
        assert "ingest_run_key" in driver.queries[0]
        assert result == ([], None, None)

    def test_execute_query_merge_document(self):
        driver = FakeDriver()
        result = driver.execute_query("MERGE (doc:Document {id: $id})")
        assert "MERGE (doc:Document" in driver.queries[0]
        assert result == ([], None, None)

    def test_execute_query_dbms_components(self):
        driver = FakeDriver()
        result = driver.execute_query("CALL dbms.components() YIELD versions, edition")
        assert result == ([{"versions": ["5.26.0"], "edition": "enterprise"}], None, None)

    def test_execute_query_missing_embeddings(self):
        driver = FakeDriver()
        driver.qa_missing_embeddings = 5
        result = driver.execute_query("MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c)")
        assert result == ([{"value": 5}], None, None)

    def test_execute_query_orphan_chunks(self):
        driver = FakeDriver()
        driver.qa_orphan_chunks = 3
        result = driver.execute_query(
            "MATCH (c:Chunk) WHERE NOT ( (:Document)-[:HAS_CHUNK]->(c) ) RETURN count(c)"
        )
        assert result == ([{"value": 3}], None, None)

    def test_execute_query_checksum_mismatches(self):
        driver = FakeDriver()
        driver.qa_checksum_mismatches = 2
        result = driver.execute_query(
            "MATCH (c:Chunk) WHERE coalesce(c.checksum, '') <> expected RETURN count(c)"
        )
        assert result == ([{"value": 2}], None, None)

    def test_execute_query_semantic_nodes(self):
        driver = FakeDriver()
        driver.semantic_nodes = 10
        result = driver.execute_query(
            "MATCH (n) WHERE n.semantic_source IS NOT NULL RETURN count(n)"
        )
        assert result == ([{"value": 10}], None, None)

    def test_execute_query_semantic_relationships(self):
        driver = FakeDriver()
        driver.semantic_relationships = 7
        result = driver.execute_query(
            "MATCH ()-[r]->() WHERE r.semantic_source IS NOT NULL RETURN count(r)"
        )
        assert result == ([{"value": 7}], None, None)

    def test_execute_query_semantic_orphans(self):
        driver = FakeDriver()
        driver.semantic_orphans = 4
        result = driver.execute_query(
            "MATCH (n) WHERE n.semantic_source IS NOT NULL AND NOT (n)-[:HAS_CHUNK]->() RETURN count(n)"
        )
        assert result == ([{"value": 4}], None, None)

    def test_execute_query_document_count(self):
        driver = FakeDriver()
        driver.graph_counts["documents"] = 15
        result = driver.execute_query("MATCH (:Document) RETURN count(*)")
        assert result == ([{"value": 15}], None, None)

    def test_execute_query_chunk_count(self):
        driver = FakeDriver()
        driver.graph_counts["chunks"] = 20
        result = driver.execute_query("MATCH (:Chunk) RETURN count(*)")
        assert result == ([{"value": 20}], None, None)

    def test_execute_query_has_chunk_relationships(self):
        driver = FakeDriver()
        driver.graph_counts["relationships"] = 18
        result = driver.execute_query(
            "MATCH (:Document)-[:HAS_CHUNK]->(:Chunk) RETURN count(*)"
        )
        assert result == ([{"value": 18}], None, None)

    def test_execute_query_unknown_pattern(self):
        driver = FakeDriver()
        result = driver.execute_query("MATCH (n:Unknown) RETURN n")
        assert result == ([{"value": 0}], None, None)

    def test_execute_query_with_parameters(self):
        driver = FakeDriver()
        result = driver.execute_query(
            "MATCH (n) WHERE n.id = $id",
            parameters={"id": "test-123"}
        )
        assert result == ([{"value": 0}], None, None)
        assert driver.queries == ["MATCH (n) WHERE n.id = $id"]

    def test_execute_query_with_kwargs(self):
        driver = FakeDriver()
        result = driver.execute_query("DETACH DELETE n", database_="test_db", routing_=None)
        assert result == ([], None, None)

    def test_execute_query_tracks_all_queries(self):
        driver = FakeDriver()
        driver.execute_query("MATCH (n) RETURN n")
        driver.execute_query("DETACH DELETE n")
        driver.execute_query("MERGE (doc:Document {id: 'x'})")
        assert len(driver.queries) == 3
        assert driver.queries[0] == "MATCH (n) RETURN n"
        assert driver.queries[1] == "DETACH DELETE n"
        assert driver.queries[2] == "MERGE (doc:Document {id: 'x'})"

    def test_execute_query_strips_whitespace(self):
        driver = FakeDriver()
        result = driver.execute_query("  \n  DETACH DELETE n  \n  ")
        assert driver.queries[0] == "DETACH DELETE n"
        assert result == ([], None, None)

    def test_graph_counts_can_be_modified(self):
        driver = FakeDriver()
        driver.graph_counts["documents"] = 100
        driver.graph_counts["chunks"] = 500
        driver.graph_counts["relationships"] = 400
        doc_result = driver.execute_query("MATCH (:Document) RETURN count(*)")
        chunk_result = driver.execute_query("MATCH (:Chunk) RETURN count(*)")
        rel_result = driver.execute_query("MATCH (:Document)-[:HAS_CHUNK]->(:Chunk) RETURN count(*)")
        assert doc_result == ([{"value": 100}], None, None)
        assert chunk_result == ([{"value": 500}], None, None)
        assert rel_result == ([{"value": 400}], None, None)

    def test_qa_counters_can_be_set_independently(self):
        driver = FakeDriver()
        driver.qa_missing_embeddings = 12
        driver.qa_orphan_chunks = 8
        driver.qa_checksum_mismatches = 3
        assert driver.qa_missing_embeddings == 12
        assert driver.qa_orphan_chunks == 8
        assert driver.qa_checksum_mismatches == 3

    def test_semantic_counters_can_be_set_independently(self):
        driver = FakeDriver()
        driver.semantic_nodes = 25
        driver.semantic_relationships = 40
        driver.semantic_orphans = 5
        assert driver.semantic_nodes == 25
        assert driver.semantic_relationships == 40
        assert driver.semantic_orphans == 5


class TestFakeSharedClient:
    """Comprehensive tests for the FakeSharedClient test double."""

    def test_initialization(self):
        client = FakeSharedClient()
        assert client.embedding_calls == []
        assert client.chat_calls == []

    def test_embedding_tracks_input_text(self):
        client = FakeSharedClient()
        result = client.embedding(input_text="test embedding text")
        assert client.embedding_calls == ["test embedding text"]
        assert result.vector == [0.0] * 5
        assert result.tokens_consumed == 10

    def test_embedding_multiple_calls(self):
        client = FakeSharedClient()
        client.embedding(input_text="first")
        client.embedding(input_text="second")
        client.embedding(input_text="third")
        assert client.embedding_calls == ["first", "second", "third"]

    def test_embedding_returns_consistent_vector(self):
        client = FakeSharedClient()
        result1 = client.embedding(input_text="test1")
        result2 = client.embedding(input_text="test2")
        assert result1.vector == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert result2.vector == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert len(result1.vector) == 5
        assert len(result2.vector) == 5

    def test_embedding_returns_token_count(self):
        client = FakeSharedClient()
        result = client.embedding(input_text="some text")
        assert result.tokens_consumed == 10
        assert hasattr(result, "tokens_consumed")

    def test_chat_completion_tracks_messages(self):
        client = FakeSharedClient()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = client.chat_completion(messages=messages, temperature=0.7)
        assert client.chat_calls == [messages]
        assert result.raw_response["choices"][0]["message"]["content"] == "Acknowledged"

    def test_chat_completion_multiple_calls(self):
        client = FakeSharedClient()
        msg1 = [{"role": "user", "content": "First"}]
        msg2 = [{"role": "user", "content": "Second"}]
        msg3 = [{"role": "user", "content": "Third"}]
        client.chat_completion(messages=msg1, temperature=0.5)
        client.chat_completion(messages=msg2, temperature=0.7)
        client.chat_completion(messages=msg3, temperature=0.9)
        assert len(client.chat_calls) == 3
        assert client.chat_calls[0] == msg1
        assert client.chat_calls[1] == msg2
        assert client.chat_calls[2] == msg3

    def test_chat_completion_accepts_temperature(self):
        client = FakeSharedClient()
        client.chat_completion(messages=[], temperature=0.0)
        client.chat_completion(messages=[], temperature=0.5)
        client.chat_completion(messages=[], temperature=1.0)
        client.chat_completion(messages=[], temperature=2.0)
        assert len(client.chat_calls) == 4

    def test_chat_completion_response_structure(self):
        client = FakeSharedClient()
        result = client.chat_completion(messages=[], temperature=0.7)
        assert hasattr(result, "raw_response")
        assert "choices" in result.raw_response
        assert len(result.raw_response["choices"]) == 1
        assert "message" in result.raw_response["choices"][0]
        assert "content" in result.raw_response["choices"][0]["message"]

    def test_chat_completion_empty_messages(self):
        client = FakeSharedClient()
        result = client.chat_completion(messages=[], temperature=0.5)
        assert client.chat_calls == [[]]
        assert result.raw_response["choices"][0]["message"]["content"] == "Acknowledged"

    def test_embedding_with_empty_string(self):
        client = FakeSharedClient()
        result = client.embedding(input_text="")
        assert client.embedding_calls == [""]
        assert result.vector == [0.0] * 5
        assert result.tokens_consumed == 10

    def test_chat_completion_with_complex_messages(self):
        client = FakeSharedClient()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."},
        ]
        _ = client.chat_completion(messages=messages, temperature=0.8)
        assert client.chat_calls[0] == messages
        assert len(client.chat_calls[0]) == 4


class TestFakePipeline:
    """Comprehensive tests for the FakePipeline test double."""

    def test_initialization_stores_all_parameters(self):
        llm = object()
        driver = FakeDriver()
        embedder = object()
        schema = {"test": "schema"}
        from_pdf = True
        text_splitter = object()
        neo4j_database = "test_db"
        kg_writer = object()
        pipeline = FakePipeline(
            llm=llm,
            driver=driver,
            embedder=embedder,
            schema=schema,
            from_pdf=from_pdf,
            text_splitter=text_splitter,
            neo4j_database=neo4j_database,
            kg_writer=kg_writer,
        )
        assert pipeline.llm is llm
        assert pipeline.driver is driver
        assert pipeline.embedder is embedder
        assert pipeline.schema == schema
        assert pipeline.from_pdf is from_pdf
        assert pipeline.text_splitter is text_splitter
        assert pipeline.database == neo4j_database
        assert pipeline.kg_writer is kg_writer
        assert pipeline.run_args == {}

    def test_run_async_with_text(self):
        client = FakeSharedClient()
        class FakeEmbedder:
            def embed_query(self, text):
                client.embedding(input_text=text)
        class FakeLLM:
            def invoke(self, text):
                client.chat_completion(messages=[{"role": "user", "content": text}], temperature=0.7)
        pipeline = FakePipeline(
            llm=FakeLLM(),
            driver=FakeDriver(),
            embedder=FakeEmbedder(),
            from_pdf=False,
            text_splitter=None,
            neo4j_database="test",
        )
        result = asyncio.run(pipeline.run_async(text="test input text"))
        assert pipeline.run_args == {"text": "test input text", "file_path": None}
        assert result.run_id == "test-run"
        assert client.embedding_calls == ["test input text"]
        assert len(client.chat_calls) == 1

    def test_run_async_with_file_path(self):
        client = FakeSharedClient()
        class FakeEmbedder:
            def embed_query(self, text):
                client.embedding(input_text=text)
        class FakeLLM:
            def invoke(self, text):
                client.chat_completion(messages=[{"role": "user", "content": text}], temperature=0.7)
        pipeline = FakePipeline(
            llm=FakeLLM(),
            driver=FakeDriver(),
            embedder=FakeEmbedder(),
            from_pdf=True,
            text_splitter=None,
            neo4j_database="test",
        )
        result = asyncio.run(pipeline.run_async(text="", file_path="/path/to/file.pdf"))
        assert pipeline.run_args == {"text": "", "file_path": "/path/to/file.pdf"}
        assert result.run_id == "test-run"

    def test_run_async_exercises_embedder(self):
        embed_called = []
        class FakeEmbedder:
            def embed_query(self, text):
                embed_called.append(text)
        class FakeLLM:
            def invoke(self, text):
                pass
        pipeline = FakePipeline(
            llm=FakeLLM(),
            driver=FakeDriver(),
            embedder=FakeEmbedder(),
            from_pdf=False,
            text_splitter=None,
            neo4j_database="test",
        )
        asyncio.run(pipeline.run_async(text="embed this"))
        assert embed_called == ["embed this"]

    def test_run_async_exercises_llm(self):
        llm_called = []
        class FakeEmbedder:
            def embed_query(self, text):
                pass
        class FakeLLM:
            def invoke(self, text):
                llm_called.append(text)
        pipeline = FakePipeline(
            llm=FakeLLM(),
            driver=FakeDriver(),
            embedder=FakeEmbedder(),
            from_pdf=False,
            text_splitter=None,
            neo4j_database="test",
        )
        asyncio.run(pipeline.run_async(text="process this"))
        assert llm_called == ["process this"]

    def test_run_async_returns_run_id(self):
        class FakeEmbedder:
            def embed_query(self, text):
                pass
        class FakeLLM:
            def invoke(self, text):
                pass
        pipeline = FakePipeline(
            llm=FakeLLM(),
            driver=FakeDriver(),
            embedder=FakeEmbedder(),
            from_pdf=False,
            text_splitter=None,
            neo4j_database="test",
        )
        result = asyncio.run(pipeline.run_async(text="test"))
        assert hasattr(result, "run_id")
        assert result.run_id == "test-run"

    def test_run_async_with_both_text_and_file_path(self):
        class FakeEmbedder:
            def embed_query(self, text):
                pass
        class FakeLLM:
            def invoke(self, text):
                pass
        pipeline = FakePipeline(
            llm=FakeLLM(),
            driver=FakeDriver(),
            embedder=FakeEmbedder(),
            from_pdf=True,
            text_splitter=None,
            neo4j_database="test",
        )
        _ = asyncio.run(pipeline.run_async(text="some text", file_path="/path/file.pdf"))
        assert pipeline.run_args["text"] == "some text"
        assert pipeline.run_args["file_path"] == "/path/file.pdf"


class TestPatchDriver:
    """Tests for the _patch_driver utility function."""

    def test_patch_driver_patches_sync_driver(self, monkeypatch):
        import types
        import tests.unit.scripts.test_kg_build as test_module
        kg_module = types.ModuleType("kg")
        kg_module.GraphDatabase = types.SimpleNamespace(driver=None)
        test_module.kg = kg_module
        factory = lambda *args, **kwargs: "test_driver"
        test_module._patch_driver(monkeypatch, factory)
        assert kg_module.GraphDatabase.driver == factory

    def test_patch_driver_patches_async_driver_if_exists(self, monkeypatch):
        import types
        import tests.unit.scripts.test_kg_build as test_module
        kg_module = types.ModuleType("kg")
        kg_module.GraphDatabase = types.SimpleNamespace(driver=None)
        kg_module.AsyncGraphDatabase = types.SimpleNamespace(driver=None)
        test_module.kg = kg_module
        factory = lambda *args, **kwargs: "async_test_driver"
        test_module._patch_driver(monkeypatch, factory)
        assert kg_module.GraphDatabase.driver == factory
        assert kg_module.AsyncGraphDatabase.driver == factory

    def test_patch_driver_handles_missing_async_driver(self, monkeypatch):
        import types
        import tests.unit.scripts.test_kg_build as test_module
        kg_module = types.ModuleType("kg")
        kg_module.GraphDatabase = types.SimpleNamespace(driver=None)
        test_module.kg = kg_module
        factory = lambda *args, **kwargs: "test_driver"
        test_module._patch_driver(monkeypatch, factory)
        assert kg_module.GraphDatabase.driver == factory


class TestPandasStub:
    """Tests for pandas stub initialization."""

    def test_pandas_stub_exists_in_sys_modules(self):
        import sys
        assert "pandas" in sys.modules

    def test_pandas_stub_has_na(self):
        import sys
        stub = sys.modules["pandas"]
        assert hasattr(stub, "NA")

    def test_pandas_stub_has_series(self):
        import sys
        stub = sys.modules["pandas"]
        assert hasattr(stub, "Series")
        assert isinstance(stub.Series, type)

    def test_pandas_stub_has_dataframe(self):
        import sys
        stub = sys.modules["pandas"]
        assert hasattr(stub, "DataFrame")
        assert isinstance(stub.DataFrame, type)

    def test_pandas_stub_has_categorical(self):
        import sys
        stub = sys.modules["pandas"]
        assert hasattr(stub, "Categorical")
        assert isinstance(stub.Categorical, type)

    def test_pandas_stub_has_extension_array(self):
        import sys
        stub = sys.modules["pandas"]
        assert hasattr(stub, "core")
        assert hasattr(stub.core, "arrays")
        assert hasattr(stub.core.arrays, "ExtensionArray")
        assert isinstance(stub.core.arrays.ExtensionArray, type)


class TestNeo4jStub:
    """Tests for neo4j stub initialization."""

    def test_neo4j_stub_exists_in_sys_modules(self):
        import sys
        assert "neo4j" in sys.modules

    def test_neo4j_stub_has_graph_database(self):
        import sys
        stub = sys.modules["neo4j"]
        assert hasattr(stub, "GraphDatabase")

    def test_neo4j_graph_database_driver_raises(self):
        import sys
        stub = sys.modules["neo4j"]
        with pytest.raises(ImportError, match="neo4j driver not available"):
            stub.GraphDatabase.driver("bolt://localhost", auth=("user", "pass"))

    def test_neo4j_stub_has_record(self):
        import sys
        stub = sys.modules["neo4j"]
        assert hasattr(stub, "Record")
        assert isinstance(stub.Record, type)

    def test_neo4j_stub_has_driver(self):
        import sys
        stub = sys.modules["neo4j"]
        assert hasattr(stub, "Driver")
        assert isinstance(stub.Driver, type)

    def test_neo4j_stub_has_query(self):
        import sys
        stub = sys.modules["neo4j"]
        assert hasattr(stub, "Query")
        assert isinstance(stub.Query, type)

    def test_neo4j_stub_has_routing_control(self):
        import sys
        stub = sys.modules["neo4j"]
        assert hasattr(stub, "RoutingControl")
        assert hasattr(stub.RoutingControl, "READ")
        assert stub.RoutingControl.READ == "READ"

    def test_neo4j_stub_has_exceptions_module(self):
        import sys
        stub = sys.modules["neo4j"]
        assert hasattr(stub, "exceptions")
        assert "neo4j.exceptions" in sys.modules

    def test_neo4j_exceptions_has_predefined_errors(self):
        import sys
        exceptions = sys.modules["neo4j.exceptions"]
        expected_errors = ["Neo4jError", "ClientError", "DriverError", "CypherSyntaxError", "CypherTypeError"]
        for error_name in expected_errors:
            assert hasattr(exceptions, error_name)
            error_class = getattr(exceptions, error_name)
            assert isinstance(error_class, type)
            assert issubclass(error_class, RuntimeError)

    def test_neo4j_exceptions_dynamic_getattr(self):
        import sys
        exceptions = sys.modules["neo4j.exceptions"]
        CustomError = exceptions.CustomError
        assert isinstance(CustomError, type)
        assert issubclass(CustomError, RuntimeError)
        assert hasattr(exceptions, "CustomError")

    def test_neo4j_exceptions_created_dynamically_are_cached(self):
        import sys
        exceptions = sys.modules["neo4j.exceptions"]
        Error1 = exceptions.TestError
        Error2 = exceptions.TestError
        assert Error1 is Error2


class TestEnvFixture:
    """Tests for the env fixture."""

    def test_env_fixture_sets_openai_key(self, env):
        import os
        assert os.environ.get("OPENAI_API_KEY") == "test-key"

    def test_env_fixture_sets_neo4j_uri(self, env):
        import os
        assert os.environ.get("NEO4J_URI") == "bolt://example"

    def test_env_fixture_sets_neo4j_username(self, env):
        import os
        assert os.environ.get("NEO4J_USERNAME") == "neo4j"

    def test_env_fixture_sets_neo4j_password(self, env):
        import os
        assert os.environ.get("NEO4J_PASSWORD") == "secret"

    def test_env_fixture_values_persist_in_test(self, env):
        import os
        assert os.environ.get("OPENAI_API_KEY") == "test-key"
        assert os.environ.get("NEO4J_URI") == "bolt://example"
        assert os.environ.get("NEO4J_USERNAME") == "neo4j"
        assert os.environ.get("NEO4J_PASSWORD") == "secret"


class TestFakeDriverIntegration:
    """Integration tests combining FakeDriver with other test doubles."""

    def test_driver_with_pipeline_async_context(self):
        driver = FakeDriver()
        class FakeEmbedder:
            def embed_query(self, text):
                pass
        class FakeLLM:
            def invoke(self, text):
                pass
        pipeline = FakePipeline(
            llm=FakeLLM(),
            driver=driver,
            embedder=FakeEmbedder(),
            from_pdf=False,
            text_splitter=None,
            neo4j_database="test",
        )
        async def _run():
            async with driver:
                await pipeline.run_async(text="test")
        asyncio.run(_run())
        assert pipeline.driver is driver

    def test_driver_query_tracking_across_sessions(self):
        driver = FakeDriver()
        with driver:
            driver.execute_query("MATCH (n) RETURN n")
        with driver:
            driver.execute_query("DETACH DELETE n")
        assert len(driver.queries) == 2
        assert "MATCH (n)" in driver.queries[0]
        assert "DETACH DELETE" in driver.queries[1]

    def test_driver_state_modification_between_queries(self):
        driver = FakeDriver()
        result1 = driver.execute_query("MATCH (:Document) RETURN count(*)")
        assert result1 == ([{"value": 2}], None, None)
        driver.graph_counts["documents"] = 50
        result2 = driver.execute_query("MATCH (:Document) RETURN count(*)")
        assert result2 == ([{"value": 50}], None, None)
