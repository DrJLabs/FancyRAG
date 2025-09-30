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

import scripts.kg_build as kg
from cli.openai_client import OpenAIClientError


class FakeSharedClient:
    def __init__(self) -> None:
        """
        Initialize the fake client and set up empty call-recording lists.
        
        The instance will track embedding requests in `self.embedding_calls` (list of input texts)
        and chat invocations in `self.chat_calls` (list of message lists).
        """
        self.embedding_calls: list[str] = []
        self.chat_calls: list[list[dict[str, str]]] = []

    def embedding(self, *, input_text: str) -> SimpleNamespace:
        """
        Record the input text and return a stub embedding result.
        
        Appends the provided input_text to the instance's embedding_calls list as a record of the call.
        
        Parameters:
            input_text (str): Text to embed.
        
        Returns:
            SimpleNamespace: An object with two attributes:
                - vector (list[float]): Embedding vector (list of floats).
                - tokens_consumed (int): Number of tokens attributed to the embedding.
        """
        self.embedding_calls.append(input_text)
        vector = [0.0] * 5
        return SimpleNamespace(vector=vector, tokens_consumed=10)

    def chat_completion(self, *, messages, temperature: float) -> SimpleNamespace:
        """
        Produce a stubbed chat completion response containing a single acknowledgement message.
        
        Parameters:
        	messages (list): Sequence of message objects sent to the chat model (each typically contains `role` and `content`).
        	temperature (float): Sampling temperature for the model's response.
        
        Returns:
        	SimpleNamespace: An object with a `raw_response` attribute that mimics an OpenAI chat completion response; `raw_response` contains a single choice whose `message.content` is "Acknowledged".
        """
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
        """
        Initialize the fake pipeline with its LLM, database driver, embedder, schema, source format, and text splitter.
        
        Parameters:
            schema (optional): Graph schema or schema-like object used by the pipeline, if any.
            from_pdf (bool): True if the pipeline input originates from a PDF source.
            neo4j_database (str): Name of the Neo4j database to use for queries and writes.
        """
        self.llm = llm
        self.driver = driver
        self.embedder = embedder
        self.schema = schema
        self.from_pdf = from_pdf
        self.text_splitter = text_splitter
        self.database = neo4j_database
        self.run_args: dict[str, str] = {}

    async def run_async(self, *, text: str = "", file_path: str | None = None):
        """
        Record the provided text and file path, invoke the embedder and LLM with the text to simulate a pipeline run, and return a test run identifier.
        
        Parameters:
            text (str): Text to process (may be empty).
            file_path (str | None): Optional source file path associated with the text.
        
        Returns:
            SimpleNamespace: Object with attribute `run_id` equal to `"test-run"`.
        """
        self.run_args = {"text": text, "file_path": file_path}
        # Simulate embedder and LLM usage to exercise client stubs
        self.embedder.embed_query(text)
        self.llm.invoke(text)
        return SimpleNamespace(run_id="test-run")


class FakeDriver:
    def __enter__(self) -> "FakeDriver":
        """
        Enter the context manager and provide the FakeDriver instance.
        
        Returns:
            The FakeDriver instance.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Context manager exit method that performs no action and does not suppress exceptions.
        
        Parameters:
            exc_type (type | None): Exception type if an exception was raised inside the context, otherwise None.
            exc (BaseException | None): Exception instance if raised inside the context, otherwise None.
            tb (types.TracebackType | None): Traceback object for the exception, otherwise None.
        
        Notes:
            This method intentionally returns None so any exception raised in the with-block is propagated.
        """
        return None

    def execute_query(self, query: str, *, database_: str | None = None):
        """
        Return a fake query result whose reported `value` is determined by keywords found in `query`.
        
        Parameters:
        	query (str): The Cypher-like query string to inspect; the presence of the substrings "Document" or "Chunk" and the absence of "HAS_CHUNK" influence the returned value.
        	database_ (str | None): Optional database name (ignored by this fake driver).
        
        Returns:
        	SimpleNamespace: An object with a `records` attribute containing a single dict `{"value": <int>}`, where the int is:
        	- 2 if "Document" appears in `query` and "HAS_CHUNK" does not,
        	- 4 if "Chunk" appears in `query` and "HAS_CHUNK" does not,
        	- 4 in all other cases.
        """
        if "Document" in query and "HAS_CHUNK" not in query:
            value = 2
        elif "Chunk" in query and "HAS_CHUNK" not in query:
            value = 4
        else:
            value = 4
        return SimpleNamespace(records=[{"value": value}])


@pytest.fixture(autouse=True)
def _ensure_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Add any local Neo4j wheel files found in /tmp to the Python import path using the provided pytest monkeypatch.
    
    Searches /tmp for files ending with `.whl` whose names start with `neo4j_graphrag-` or `neo4j-` and prepends each matching wheel's path to sys.path via monkeypatch.syspath_prepend so tests can import those packages.
    """
    wheel_dir = "/tmp"
    for name in os.listdir(wheel_dir):
        if name.startswith("neo4j_graphrag-") and name.endswith(".whl"):
            monkeypatch.syspath_prepend(os.path.join(wheel_dir, name))
        if name.startswith("neo4j-") and name.endswith(".whl"):
            monkeypatch.syspath_prepend(os.path.join(wheel_dir, name))


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Configure environment variables required by the tests.
    
    Sets OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD to deterministic test values.
    """
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
    monkeypatch.setattr(kg.GraphDatabase, "driver", lambda uri, auth: FakeDriver())
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
            """
            Simulates a failing embedding request by always raising an OpenAIClientError.
            
            Parameters:
                input_text (str): Text that would be embedded.
            
            Raises:
                OpenAIClientError: Always raised with message "boom" and remediation "retry later".
            """
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
    monkeypatch.setattr(kg.GraphDatabase, "driver", lambda uri, auth: FakeDriver())
    monkeypatch.setattr(kg.OpenAISettings, "load", classmethod(lambda cls, env=None, actor=None: settings))
    monkeypatch.setattr(kg, "SimpleKGPipeline", lambda **kwargs: FakePipeline(**kwargs))

    with pytest.raises(RuntimeError) as excinfo:
        kg.run(["--source", str(source), "--chunk-size", "5", "--chunk-overlap", "1"])
    assert "OpenAI request failed" in str(excinfo.value)


def test_missing_file_raises(monkeypatch: pytest.MonkeyPatch, env) -> None:
    """
    Verifies that running the pipeline with a non-existent source file raises FileNotFoundError.
    
    Replaces the GraphDatabase.driver with a FakeDriver to isolate the test from external services before invoking kg.run with a missing source path.
    
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch GraphDatabase.driver for the duration of the test.
    """
    monkeypatch.setattr(kg.GraphDatabase, "driver", lambda uri, auth: FakeDriver())
    with pytest.raises(FileNotFoundError):
        kg.run(["--source", "does-not-exist.txt"])
