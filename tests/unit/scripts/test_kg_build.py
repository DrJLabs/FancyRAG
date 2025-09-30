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


class FakeDriver:
    def __enter__(self) -> "FakeDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute_query(self, query: str, *, database_: str | None = None):
        if "Document" in query and "HAS_CHUNK" not in query:
            value = 2
        elif "Chunk" in query and "HAS_CHUNK" not in query:
            value = 4
        else:
            value = 4
        return SimpleNamespace(records=[{"value": value}])


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
    monkeypatch.setattr(kg.GraphDatabase, "driver", lambda uri, auth: FakeDriver())
    with pytest.raises(FileNotFoundError):
        kg.run(["--source", "does-not-exist.txt"])
