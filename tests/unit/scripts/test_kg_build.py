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
        def driver(*_args, **_kwargs):  # pragma: no cover - placeholder stub
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

    def _exceptions_getattr(name: str) -> type[RuntimeError]:  # pragma: no cover - dynamic
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
        """
        Initialize a FakePipeline used in tests with the provided components.
        
        Parameters:
            llm: The language model stub used to simulate LLM invocations.
            driver: The fake Neo4j driver instance used to capture and simulate queries.
            embedder: The embedder stub used to generate embeddings for queries/text.
            schema: Optional KG schema or mapping used by the pipeline.
            from_pdf: Boolean indicating whether the source was a PDF (affects pipeline behavior in tests).
            text_splitter: Component responsible for splitting text into chunks for ingestion.
            neo4j_database: Name of the Neo4j database to target for queries.
            kg_writer: Optional writer instance (e.g., SanitizingNeo4jWriter) used to persist data to the graph.
        """
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
        """
        Store invocation arguments, call embedder and LLM stubs to simulate a run, and return a test run identifier.
        
        Parameters:
            text (str): Input text to process.
            file_path (str | None): Optional path to the source file associated with the run.
        
        Returns:
            SimpleNamespace: Object with attribute `run_id` containing the test run identifier (for example, "test-run").
        """
        self.run_args = {"text": text, "file_path": file_path}
        # Simulate embedder and LLM usage to exercise client stubs
        self.embedder.embed_query(text)
        self.llm.invoke(text)
        return SimpleNamespace(run_id="test-run")


class FakeDriver:
    def __init__(self) -> None:
        """
        Initialize the fake driver used in tests.
        
        Creates an empty `queries` list to record executed query texts and a `_pool` namespace with a `pool_config`
        containing a `user_agent` attribute initialized to None.
        """
        self.queries: list[str] = []
        self._pool = types.SimpleNamespace(pool_config=types.SimpleNamespace(user_agent=None))

    def __enter__(self) -> "FakeDriver":
        """
        Enter the context manager for the FakeDriver and provide the driver instance.
        
        Returns:
            FakeDriver: The same FakeDriver instance (`self`) for use within the context.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    async def __aenter__(self) -> "FakeDriver":  # pragma: no cover - async compatibility
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        """
        Exit the asynchronous context manager.
        
        Called when leaving an `async with` block; receives exception information if an exception was raised inside the block. This method does not suppress exceptions (it always returns None).
        
        Parameters:
            exc_type (type | None): Exception class if an exception was raised, otherwise None.
            exc (BaseException | None): Exception instance if an exception was raised, otherwise None.
            tb (types.TracebackType | None): Traceback object for the exception, if any.
        """
        return None

    def execute_query(self, query: str, *params, database_: str | None = None, **kwargs):
        """
        Simulate executing a Cypher query against a test Neo4j driver and return canned responses based on the query text.
        
        This test helper records the trimmed query into self.queries and returns a tuple shaped like (records, summary, metadata) where `records` is a list of dicts representing query results. The returned `records` vary deterministically for specific query patterns used in tests (e.g., DETACH DELETE, MERGE for Document nodes, dbms.components, and various MATCH ... RETURN count queries).
        
        Parameters:
            query (str): The Cypher query text to execute; it will be trimmed and recorded.
            *params: Ignored positional parameters for compatibility with call sites.
            database_ (str | None): Ignored database selector; present for compatibility with call sites.
            **kwargs: Ignored keyword parameters for compatibility with call sites.
        
        Returns:
            tuple: A 3-tuple (records, summary, metadata) where `records` is a list of result dictionaries (e.g., [{"value": N}] or [{"versions": [...], "edition": "..."}]) and `summary` and `metadata` are always None in this fake driver.
        """
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
    wheel_dir = "/tmp"
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


def test_run_pipeline_success(tmp_path, monkeypatch: pytest.MonkeyPatch, env) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("sample content", encoding="utf-8")
    log_path = tmp_path / "log.json"

    fake_client = FakeSharedClient()
    captured_pipeline: dict[str, FakePipeline] = {}
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

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda settings: fake_client)
    monkeypatch.setattr(kg, "SimpleKGPipeline", lambda **kwargs: captured_pipeline.setdefault("pipeline", FakePipeline(**kwargs)))

    def driver_factory(*_args, **_kwargs):
        """
        Create a new FakeDriver, record it in `created_drivers`, and return it.
        
        Instantiates a FakeDriver, appends it to the module-level list `created_drivers` for later inspection in tests, and returns the instance. Any positional or keyword arguments are ignored.
        
        Returns:
            FakeDriver: The newly created driver instance.
        """
        driver = FakeDriver()
        created_drivers.append(driver)
        return driver

    _patch_driver(monkeypatch, driver_factory)
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda cls, env=None, actor=None: settings))

    log = kg.run([
        "--source",
        str(source),
        "--log-path",
        str(log_path),
        "--chunk-size",
        "10",
        "--chunk-overlap",
        "2",
    ])

    assert log["status"] == "success"
    assert log["counts"] == {"documents": 2, "chunks": 4, "relationships": 4}
    assert fake_client.embedding_calls
    assert fake_client.chat_calls
    assert captured_pipeline["pipeline"].run_args["text"] == "sample content"
    assert isinstance(captured_pipeline["pipeline"].kg_writer, kg.SanitizingNeo4jWriter)
    assert created_drivers, "Expected GraphDatabase.driver to be invoked"
    assert any("DETACH DELETE" in query for driver in created_drivers for query in driver.queries)
    saved = json.loads(log_path.read_text())
    assert saved["status"] == "success"


def test_run_handles_openai_failure(monkeypatch: pytest.MonkeyPatch, env, tmp_path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("content", encoding="utf-8")

    class FailingClient(FakeSharedClient):
        def embedding(self, *, input_text: str):  # type: ignore[override]
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

    monkeypatch.setattr(kg, "SharedOpenAIClient", lambda settings: FailingClient())
    _patch_driver(monkeypatch, lambda uri, auth=None: FakeDriver())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda cls, env=None, actor=None: settings))
    monkeypatch.setattr(kg, "SimpleKGPipeline", lambda **kwargs: FakePipeline(**kwargs))

    with pytest.raises(RuntimeError) as excinfo:
        kg.run(["--source", str(source), "--chunk-size", "5", "--chunk-overlap", "1"])
    assert "OpenAI request failed" in str(excinfo.value)


def test_missing_file_raises(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """
    Verifies that running the pipeline with a non-existent source file raises FileNotFoundError.

    Replaces the GraphDatabase driver with FakeDriver to isolate the test from external services before invoking kg.run with a missing source path.

    Parameters:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch AsyncGraphDatabase.driver for the duration of the test.
    """
    _patch_driver(monkeypatch, lambda uri, auth=None: FakeDriver())
    with pytest.raises(FileNotFoundError):
        kg.run(["--source", "does-not-exist.txt"])

# --- Additional tests for scripts/kg_build.py (pytest) ---

def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)
    args = kg._parse_args([])
    assert args.source == str(kg.DEFAULT_SOURCE)
    assert args.chunk_size == kg.DEFAULT_CHUNK_SIZE
    assert args.chunk_overlap == kg.DEFAULT_CHUNK_OVERLAP
    assert args.database is None
    assert args.log_path == str(kg.DEFAULT_LOG_PATH)


def test_parse_args_overrides() -> None:
    args = kg._parse_args([
        "--source", "/tmp/foo.txt",
        "--chunk-size", "42",
        "--chunk-overlap", "7",
        "--database", "neo4j",
        "--log-path", "/tmp/log.json",
    ])
    assert args.source == "/tmp/foo.txt"
    assert args.chunk_size == 42
    assert args.chunk_overlap == 7
    assert args.database == "neo4j"
    assert args.log_path == "/tmp/log.json"


def test_ensure_positive_and_non_negative() -> None:
    assert kg._ensure_positive(1, name="x") == 1
    assert kg._ensure_non_negative(0, name="y") == 0
    with pytest.raises(ValueError):
        kg._ensure_positive(0, name="x")
    with pytest.raises(ValueError):
        kg._ensure_non_negative(-1, name="y")


def test_ensure_directory_creates_parents(tmp_path) -> None:
    target = tmp_path / "deep" / "nested" / "log.json"
    kg._ensure_directory(target)
    assert (tmp_path / "deep" / "nested").exists()


def test_extract_content_variants_mapping_and_list() -> None:
    # Simple string content
    raw1 = {"choices": [{"message": {"content": "Hello"}}]}
    assert kg._extract_content(raw1) == "Hello"

    # Content as list of items with { "text": { "value": ... } }
    raw2 = {"choices": [{"message": {"content": [{"text": {"value": "Hi"}}, {"text": {"value": r"\!"}}]}}]}
    assert kg._extract_content(raw2) == r"Hi\!"

    # Missing/empty choices returns empty string
    assert kg._extract_content({}) == ""


def test_sanitize_property_value_serializes_nested_map() -> None:
    value = {"note": "Initial", "meta": {"flag": True, "count": 2}}
    result = kg._sanitize_property_value(value)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["note"] == "Initial"
    assert parsed["meta"]["flag"] is True


def test_sanitize_property_value_preserves_primitive_list() -> None:
    value = [1, 2, 3]
    assert kg._sanitize_property_value(value) == [1, 2, 3]


def test_strip_code_fence_variants() -> None:
    fenced = "```json\npayload\n```"
    assert kg._strip_code_fence(fenced) == "payload"
    single_line = "no fences here"
    assert kg._strip_code_fence(single_line) == "no fences here"


def test_shared_openai_embedder_calls_dimension_enforcer(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"flag": False}
    def _fake_ensure(vec, settings):
        called["flag"] = True
        assert isinstance(vec, list)
        assert len(vec) == 5
        assert settings.actor == "kg_build"

    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=1,
        backoff_seconds=0.0,
        enable_fallback=False,
    )
    client = FakeSharedClient()
    monkeypatch.setattr(kg, "ensure_embedding_dimensions", _fake_ensure)
    embedder = kg.SharedOpenAIEmbedder(client, settings)
    vec = embedder.embed_query("hello")
    assert called["flag"] is True
    assert isinstance(vec, list) and len(vec) == 5


def test_shared_openai_embedder_converts_openai_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailEmbed(FakeSharedClient):
        def embedding(self, *, input_text: str):  # type: ignore[override]
            raise kg.OpenAIClientError("nope", remediation="retry")
    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=1,
        backoff_seconds=0.0,
        enable_fallback=False,
    )
    emb = kg.SharedOpenAIEmbedder(FailEmbed(), settings)
    with pytest.raises(kg.EmbeddingsGenerationError):
        emb.embed_query("x")


def test_shared_openai_llm_invoke_strips_code_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client(FakeSharedClient):
        def chat_completion(self, *, messages, temperature: float):  # type: ignore[override]
            return SimpleNamespace(
                raw_response={
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"text": {"value": "```text\nOK\n```"}}
                                ]
                            }
                        }
                    ]
                }
            )
    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=1,
        backoff_seconds=0.0,
        enable_fallback=False,
    )
    llm = kg.SharedOpenAILLM(Client(), settings)
    out = llm.invoke("prompt")
    assert out.content == "OK"


def test_shared_openai_llm_invoke_empty_raises() -> None:
    class Client(FakeSharedClient):
        def chat_completion(self, *, messages, temperature: float):  # type: ignore[override]
            return SimpleNamespace(
                raw_response={
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"text": {"value": "```\n\n```"}}
                                ]
                            }
                        }
                    ]
                }
            )
    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=1,
        backoff_seconds=0.0,
        enable_fallback=False,
    )
    llm = kg.SharedOpenAILLM(Client(), settings)
    with pytest.raises(kg.LLMGenerationError):
        llm.invoke("prompt")


def test_shared_openai_llm_converts_openai_error_to_llm_error() -> None:
    class Client(FakeSharedClient):
        def chat_completion(self, *, messages, temperature: float):  # type: ignore[override]
            raise kg.OpenAIClientError("failure", remediation="retry")
    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=1,
        backoff_seconds=0.0,
        enable_fallback=False,
    )
    llm = kg.SharedOpenAILLM(Client(), settings)
    with pytest.raises(kg.LLMGenerationError):
        llm.invoke("prompt")


def test_shared_openai_llm_ainvoke(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    class Client(FakeSharedClient):
        def chat_completion(self, *, messages, temperature: float):  # type: ignore[override]
            return SimpleNamespace(raw_response={"choices": [{"message": {"content": "ACK"}}]})
    settings = kg.OpenAISettings(
        chat_model="gpt-4.1-mini",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=5,
        embedding_dimensions_override=None,
        actor="kg_build",
        max_attempts=1,
        backoff_seconds=0.0,
        enable_fallback=False,
    )
    llm = kg.SharedOpenAILLM(Client(), settings)
    res = asyncio.run(llm.ainvoke("asynchronous"))
    assert res.content == "ACK"


def test_collect_counts_handles_neo4j_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyNeo4jError(Exception):
        pass
    monkeypatch.setattr(kg, "Neo4jError", DummyNeo4jError)

    class Driver:
        def execute_query(self, query: str, *, database_: str | None = None):
            if "Chunk" in query and "HAS_CHUNK" not in query:
                raise DummyNeo4jError("boom")
            if "HAS_CHUNK" in query:
                return SimpleNamespace(records=[{"value": 3}])
            if "Document" in query:
                return SimpleNamespace(records=[{"value": 1}])
            return SimpleNamespace(records=[{"value": 2}])

    counts = kg._collect_counts(Driver(), database=None)
    assert counts["documents"] == 1
    assert counts["relationships"] == 3
    assert "chunks" not in counts  # skipped due to error


def test_main_success_and_error_paths(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    # Success path
    monkeypatch.setattr(kg, "run", lambda argv=None: {"status": "ok"})
    assert kg.main(["--source", str(tmp_path / "x.txt")]) == 0

    # Error path: raise RuntimeError -> exit code 1 and stderr message
    def raise_run(argv=None):
        raise RuntimeError("explode")
    monkeypatch.setattr(kg, "run", raise_run)
    rc = kg.main(["--source", str(tmp_path / "x.txt")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err
