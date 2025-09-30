from __future__ import annotations

import json
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    ) -> None:
        self.llm = llm
        self.driver = driver
        self.embedder = embedder
        self.schema = schema
        self.from_pdf = from_pdf
        self.text_splitter = text_splitter
        self.database = neo4j_database
        self.run_args: dict[str, str] = {}

    async def run_async(self, *, text: str = "", file_path: str | None = None):
        self.run_args = {"text": text, "file_path": file_path}
        # Simulate embedder and LLM usage to exercise client stubs
        self.embedder.embed_query(text)
        self.llm.invoke(text)
        return SimpleNamespace(run_id="test-run")


class FakeAsyncDriver:
    async def __aenter__(self) -> "FakeAsyncDriver":
        """Return the driver instance for async context managers."""

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Async context manager exit hook that performs no cleanup."""

        return None

    async def close(self) -> None:  # pragma: no cover - compatibility shim
        """Match the async driver close contract."""

        return None

    async def execute_query(self, query: str, *, database_: str | None = None):
        """Return deterministic counts matching the Cypher query text."""

        if "Document" in query and "HAS_CHUNK" not in query:
            value = 2
        elif "Chunk" in query and "HAS_CHUNK" not in query:
            value = 4
        else:
            value = 4
        return ([{"value": value}], None, None)


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
    monkeypatch.setattr(kg.AsyncGraphDatabase, "driver", lambda uri, auth=None: FakeAsyncDriver())
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
    monkeypatch.setattr(kg.AsyncGraphDatabase, "driver", lambda uri, auth=None: FakeAsyncDriver())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda cls, env=None, actor=None: settings))
    monkeypatch.setattr(kg, "SimpleKGPipeline", lambda **kwargs: FakePipeline(**kwargs))

    with pytest.raises(RuntimeError) as excinfo:
        kg.run(["--source", str(source), "--chunk-size", "5", "--chunk-overlap", "1"])
    assert "OpenAI request failed" in str(excinfo.value)


def test_missing_file_raises(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """
    Verifies that running the pipeline with a non-existent source file raises FileNotFoundError.

    Replaces the AsyncGraphDatabase.driver with FakeAsyncDriver to isolate the test from external services before invoking kg.run with a missing source path.

    Parameters:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch AsyncGraphDatabase.driver for the duration of the test.
    """
    monkeypatch.setattr(kg.AsyncGraphDatabase, "driver", lambda uri, auth=None: FakeAsyncDriver())
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
    raw2 = {"choices": [{"message": {"content": [{"text": {"value": "Hi"}}, {"text": {"value": "\!"}}]}}]}
    assert kg._extract_content(raw2) == "Hi\!"

    # Missing/empty choices returns empty string
    assert kg._extract_content({}) == ""


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
