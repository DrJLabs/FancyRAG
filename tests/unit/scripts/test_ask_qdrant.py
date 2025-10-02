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
        """
        Constructor placeholder for the stub retriever that indicates it must be patched in tests.
        
        Raises:
            NotImplementedError: Always raised with the message "Stub retriever should be patched in tests".
        """
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
    """
    Patch ask.SharedOpenAIClient with a fake client whose embedding method returns a preset vector.
    
    This helper installs a FakeClient via monkeypatch that captures the provided settings and whose embedding(...) call returns a SimpleNamespace with attribute `vector` set to the given `vector`. Use in tests to deterministically control embedding output.
    
    Parameters:
        monkeypatch: pytest.MonkeyPatch — fixture used to apply the monkeypatch.
        vector (Sequence[float]): The embedding vector that FakeClient.embedding will return.
    """
    class FakeClient:
        def __init__(self, settings):
            self.settings = settings

        def embedding(self, *, input_text: str):  # noqa: ARG002 - signature parity
            return SimpleNamespace(vector=vector)

    monkeypatch.setattr(ask, "SharedOpenAIClient", lambda settings: FakeClient(settings))


def _setup_driver(monkeypatch):
    """
    Patch ask.GraphDatabase.driver to return a simple context manager that simulates a Neo4j driver.
    
    The provided fake context manager yields a plain object on enter and does not suppress exceptions on exit, allowing tests to run code that uses `with GraphDatabase.driver(...) as ...:` without creating a real driver.
    
    Parameters:
        monkeypatch: The pytest monkeypatch fixture used to apply the attribute patch.
    """
    class FakeDriverCtx:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

        def execute_query(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(ask.GraphDatabase, "driver", lambda *_, **__: FakeDriverCtx())


def _setup_retriever(monkeypatch, *, records, capture):
    """
    Register a test fake retriever implementation on ask.QdrantNeo4jRetriever and capture invocation details.
    
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Fixture used to set the attribute on the ask module.
        records (Iterable[dict]): Sequence of payload dicts that each fake record's .data() will return.
        capture (dict): Mutable mapping where the fake retriever stores captured information:
            - "args": positional arguments passed to the retriever constructor
            - "kwargs": keyword arguments passed to the retriever constructor
            - "top_k": value passed to get_search_results
            - "query_vector": query_vector passed to get_search_results
    """
    class FakeRecord:
        def __init__(self, payload):
            self._payload = payload

        def data(self):
            return dict(self._payload)

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            """
            Initialize the fake retriever while recording the initialization arguments.
            
            Stores the passed positional and keyword arguments into the shared `capture` mapping under keys `"args"` and `"kwargs"`, and creates `FakeRecord` instances from the surrounding `records` sequence, assigning them to `self._records`.
            
            Parameters:
                *args: Positional arguments supplied to the retriever constructor; captured in `capture["args"]`.
                **kwargs: Keyword arguments supplied to the retriever constructor; captured in `capture["kwargs"]`.
            """
            capture["args"] = args
            capture["kwargs"] = kwargs
            self._records = [FakeRecord(item) for item in records]

        def get_search_results(self, *, query_vector, top_k):
            """
            Record the provided `query_vector` and `top_k` in the shared `capture` dictionary and return the preconfigured search results.
            
            Parameters:
            	query_vector (Sequence[float]): Embedding vector used for the search query.
            	top_k (int): Number of top matches requested.
            
            Returns:
            	SimpleNamespace: An object with `records` set to the retriever's stored records and `metadata` set to None.
            """
            capture["top_k"] = top_k
            capture["query_vector"] = query_vector
            return SimpleNamespace(records=self._records, metadata=None)

    monkeypatch.setattr(ask, "QdrantNeo4jRetriever", FakeRetriever)


def _configure_identity_scrubber(monkeypatch):
    """
    Replace ask.scrub_object with an identity function so payloads are not altered during tests.
    """
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


def test_main_includes_semantic_context(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ask_qdrant.py",
            "--question",
            "How?",
            "--top-k",
            "1",
            "--include-semantic",
        ],
    )
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
            }
        ],
        capture=capture,
    )
    semantic_payload = {
        "1": {
            "nodes": [
                {
                    "id": "1:Person",
                    "labels": ["Person", "__Entity__"],
                    "properties": {"name": "Alice"},
                }
            ],
            "relationships": [
                {
                    "type": "RELATED_TO",
                    "start": "1:Person",
                    "end": "1:Company",
                    "properties": {"weight": 0.8},
                }
            ],
        }
    }
    monkeypatch.setattr(ask, "_fetch_semantic_context", lambda *_, **__: semantic_payload)

    ask.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["matches"][0]["semantic"] == semantic_payload["1"]
    artifact_path = tmp_path / "artifacts" / "local_stack" / "ask_qdrant.json"
    saved = json.loads(artifact_path.read_text())
    assert saved["matches"][0]["semantic"] == semantic_payload["1"]


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
        """
        Helper function that always raises a retriever initialization error.
        
        Raises:
            Boom: Always raised with the message "init failed".
        """
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
