# Testing library/framework: pytest (with monkeypatch and capsys fixtures). These tests follow existing project conventions.
from __future__ import annotations

import json
import sys
from importlib import util
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

existing_neo4j = sys.modules.get("neo4j")
if existing_neo4j is None:
    stub_neo4j = ModuleType("neo4j")
    stub_neo4j.GraphDatabase = SimpleNamespace(driver=lambda *_, **__: None)
    stub_neo4j.RoutingControl = SimpleNamespace(READ="READ")
    stub_neo4j.__spec__ = ModuleSpec("neo4j", loader=None)
    sys.modules.setdefault("neo4j", stub_neo4j)
else:
    if getattr(existing_neo4j, "__spec__", None) is None:
        existing_neo4j.__spec__ = ModuleSpec("neo4j", loader=None)
    if not hasattr(existing_neo4j, "RoutingControl"):
        existing_neo4j.RoutingControl = SimpleNamespace(READ="READ")
    stub_neo4j = existing_neo4j

stub_neo4j_exceptions = ModuleType("neo4j.exceptions")
stub_neo4j_exceptions.Neo4jError = Exception
stub_neo4j_exceptions.__spec__ = ModuleSpec("neo4j.exceptions", loader=None)
if "neo4j.exceptions" not in sys.modules:
    sys.modules.setdefault("neo4j.exceptions", stub_neo4j_exceptions)

stub_qdrant = ModuleType("qdrant_client")


class _StubQdrantClient:  # pragma: no cover - import stub
    def __init__(self, *_, **__):
        pass


stub_qdrant.QdrantClient = _StubQdrantClient
stub_qdrant_http = ModuleType("qdrant_client.http")
stub_qdrant_exceptions = ModuleType("qdrant_client.http.exceptions")
stub_qdrant_exceptions.ApiException = Exception
stub_qdrant_exceptions.ResponseHandlingException = Exception
stub_qdrant_http.exceptions = stub_qdrant_exceptions
stub_qdrant.http = SimpleNamespace(exceptions=stub_qdrant_exceptions)
if util.find_spec("qdrant_client") is None:
    sys.modules.setdefault("qdrant_client", stub_qdrant)
    sys.modules.setdefault("qdrant_client.http", stub_qdrant_http)
    sys.modules.setdefault("qdrant_client.http.exceptions", stub_qdrant_exceptions)

stub_openai_client = ModuleType("cli.openai_client")


class _StubOpenAIClientError(Exception):
    def __init__(self, message, remediation=None):
        super().__init__(message)
        self.remediation = remediation


class _StubSharedOpenAIClient:  # pragma: no cover - import stub
    def __init__(self, *_, **__):
        raise NotImplementedError("Stub SharedOpenAIClient should be patched in tests")


stub_openai_client.OpenAIClientError = _StubOpenAIClientError
stub_openai_client.SharedOpenAIClient = _StubSharedOpenAIClient
if util.find_spec("cli.openai_client") is None:
    sys.modules.setdefault("cli.openai_client", stub_openai_client)

stub_graphrag = ModuleType("neo4j_graphrag")
stub_graphrag.__spec__ = ModuleSpec("neo4j_graphrag", loader=None)
sys.modules.setdefault("neo4j_graphrag", stub_graphrag)

stub_graphrag_exceptions = ModuleType("neo4j_graphrag.exceptions")
stub_graphrag_exceptions.__spec__ = ModuleSpec("neo4j_graphrag.exceptions", loader=None)
stub_graphrag_exceptions.RetrieverInitializationError = Exception
stub_graphrag_exceptions.SearchValidationError = Exception
sys.modules.setdefault("neo4j_graphrag.exceptions", stub_graphrag_exceptions)

stub_graphrag_retrievers = ModuleType("neo4j_graphrag.retrievers")
stub_graphrag_retrievers.__spec__ = ModuleSpec("neo4j_graphrag.retrievers", loader=None)


class _StubRetriever:  # pragma: no cover - import stub
    def __init__(self, *_, **__):
        raise NotImplementedError("Stub retriever should be patched in tests")


stub_graphrag_retrievers.QdrantNeo4jRetriever = _StubRetriever
sys.modules.setdefault("neo4j_graphrag.retrievers", stub_graphrag_retrievers)

MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "ask_qdrant.py"
SPEC = util.spec_from_file_location("ask_qdrant", MODULE_PATH)
ask = util.module_from_spec(SPEC)  # type: ignore[arg-type]
assert SPEC and SPEC.loader  # pragma: no cover - sanity check
SPEC.loader.exec_module(ask)  # type: ignore[union-attr]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)


def _setup_shared_client(monkeypatch, *, vector):
    class FakeClient:
        def __init__(self, settings):
            self.settings = settings

        def embedding(self, *, input_text: str):  # noqa: ARG002 - signature parity
            return SimpleNamespace(vector=vector)

    monkeypatch.setattr(ask, "SharedOpenAIClient", lambda settings: FakeClient(settings))


def _setup_driver(monkeypatch):
    class FakeDriverCtx:
        def __enter__(self):
            return object()

        def __exit__(self, _exc_type, _exc, _tb):
            return False

    monkeypatch.setattr(ask.GraphDatabase, "driver", lambda *_, **__: FakeDriverCtx())


def _setup_retriever(monkeypatch, *, records, capture):
    class FakeRecord:
        def __init__(self, payload):
            self._payload = payload

        def data(self):
            return dict(self._payload)

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            capture["args"] = args
            capture["kwargs"] = kwargs
            self._records = [FakeRecord(item) for item in records]

        def get_search_results(self, *, query_vector, top_k):  # noqa: ARG002
            capture["top_k"] = top_k
            capture["query_vector"] = query_vector
            return SimpleNamespace(records=self._records, metadata=None)

    monkeypatch.setattr(ask, "QdrantNeo4jRetriever", FakeRetriever)


def _configure_identity_scrubber(monkeypatch):
    monkeypatch.setattr(ask, "scrub_object", lambda payload: payload)


def test_main_success_creates_artifact(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "How?", "--top-k", "2"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1, 0.2, 0.3])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(
        monkeypatch,
        records=[
            {
                "chunk_id": "1",
                "text": "example chunk",
                "source_path": "doc.txt",
                "document_name": "Doc",
                "document_source_path": "doc.txt",
                "score": 0.99,
            },
            {
                "chunk_id": "2",
                "text": "other chunk",
                "source_path": "other.txt",
                "document_name": None,
                "document_source_path": None,
                "score": 0.42,
            },
        ],
        capture=capture,
    )

    ask.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["status"] == "success"
    assert payload["message"] == "Retrieved 2 matches"
    artifact_path = tmp_path / "artifacts" / "local_stack" / "ask_qdrant.json"
    saved = json.loads(artifact_path.read_text())
    assert saved["matches"][0]["chunk_id"] == "1"
    assert saved["matches"][0]["score"] == pytest.approx(0.99)
    assert saved["matches"][1]["chunk_id"] == "2"
    assert capture["top_k"] == 2
    kwargs = capture["kwargs"]
    assert kwargs["id_property_external"] == "chunk_id"
    assert "OPTIONAL MATCH (doc:Document)-[:HAS_CHUNK]->(node)" in kwargs["retrieval_query"]


def test_main_handles_openai_error(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Oops"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)

    class FakeOpenAIError(Exception):
        def __init__(self, message, remediation=None):
            super().__init__(message)
            self.remediation = remediation

    monkeypatch.setattr(ask, "OpenAIClientError", FakeOpenAIError)

    class FailingClient:
        def __init__(self, settings):
            pass

        def embedding(self, *, input_text: str):  # noqa: ARG002
            raise FakeOpenAIError("boom", remediation="try again")

    monkeypatch.setattr(ask, "SharedOpenAIClient", lambda settings: FailingClient(settings))

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["status"] == "error"
    assert payload["message"] == "try again"
    artifact_path = tmp_path / "artifacts" / "local_stack" / "ask_qdrant.json"
    saved = json.loads(artifact_path.read_text())
    assert saved["status"] == "error"


def test_retriever_initialization_error(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "How?", "--top-k", "2"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1, 0.2])
    _setup_driver(monkeypatch)

    class Boom(Exception):
        pass

    def _boom(*_args, **_kwargs):
        raise Boom("init failed")

    monkeypatch.setattr(ask, "QdrantNeo4jRetriever", _boom)
    monkeypatch.setattr(ask, "RetrieverInitializationError", Boom)

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["status"] == "error"
    assert "init failed" in payload["message"]
    artifact_path = tmp_path / "artifacts" / "local_stack" / "ask_qdrant.json"
    saved = json.loads(artifact_path.read_text())
    assert saved["status"] == "error"
