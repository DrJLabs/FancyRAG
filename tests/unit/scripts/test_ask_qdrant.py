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
                "chunk_uid": "chunk-1",
                "text": "example chunk",
                "source_path": "doc.txt",
                "document_name": "Doc",
                "document_source_path": "doc.txt",
                "score": 0.99,
            },
            {
                "chunk_id": "2",
                "chunk_uid": "chunk-2",
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
    assert saved["matches"][0]["chunk_uid"] == "chunk-1"
    assert saved["matches"][0]["score"] == pytest.approx(0.99)
    assert saved["matches"][1]["chunk_id"] == "2"
    assert saved["matches"][1]["chunk_uid"] == "chunk-2"
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
                "chunk_uid": "chunk-1",
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
        "chunk-1": {
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
    assert payload["matches"][0]["semantic"] == semantic_payload["chunk-1"]
    artifact_path = tmp_path / "artifacts" / "local_stack" / "ask_qdrant.json"
    saved = json.loads(artifact_path.read_text())
    assert saved["matches"][0]["semantic"] == semantic_payload["chunk-1"]


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
def test_main_with_default_top_k(monkeypatch, tmp_path, capsys):
    """Test that default top_k value (5) is used when not specified."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "What is AI?"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.5, 0.5])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    ask.main()

    assert capture["top_k"] == 5


def test_main_with_custom_database(monkeypatch, tmp_path, capsys):
    """Test that custom NEO4J_DATABASE environment variable is respected."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEO4J_DATABASE", "custom_db")
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    ask.main()

    kwargs = capture["kwargs"]
    assert kwargs.get("neo4j_database") == "custom_db"


def test_main_with_qdrant_api_key(monkeypatch, tmp_path, capsys):
    """Test that QDRANT_API_KEY is passed to QdrantClient."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_API_KEY", "test-api-key")
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    qdrant_init: dict = {}

    class FakeQdrantClient:
        def __init__(self, *, url, api_key=None, **_):
            qdrant_init["url"] = url
            qdrant_init["api_key"] = api_key

    monkeypatch.setattr(ask, "QdrantClient", FakeQdrantClient)

    ask.main()

    assert qdrant_init.get("api_key") == "test-api-key"


def test_main_empty_results(monkeypatch, tmp_path, capsys):
    """Test handling of empty search results."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Nonexistent"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1, 0.2])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    ask.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["status"] == "skipped"
    assert payload["message"] == "Qdrant returned no matches"
    artifact_path = tmp_path / "artifacts" / "local_stack" / "ask_qdrant.json"
    saved = json.loads(artifact_path.read_text())
    assert saved["matches"] == []
    assert saved["status"] == "skipped"


def test_main_search_validation_error(monkeypatch, tmp_path, capsys):
    """Test handling of search validation errors from retriever."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)

    class ValidationError(Exception):
        pass

    class FailingRetriever:
        def __init__(self, *args, **kwargs):
            pass

        def get_search_results(self, **kwargs):
            raise ValidationError("Invalid search parameters")

    monkeypatch.setattr(ask, "QdrantNeo4jRetriever", FailingRetriever)
    monkeypatch.setattr(ask, "SearchValidationError", ValidationError)

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "error"
    assert "Invalid search parameters" in payload["message"]


def test_main_qdrant_api_exception(monkeypatch, tmp_path, capsys):
    """Test handling of Qdrant API exceptions."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)

    class QdrantAPIError(Exception):
        pass

    class FailingRetriever:
        def __init__(self, *args, **kwargs):
            pass

        def get_search_results(self, **kwargs):
            raise QdrantAPIError("Connection to Qdrant failed")

    monkeypatch.setattr(ask, "QdrantNeo4jRetriever", FailingRetriever)
    monkeypatch.setattr(ask, "ApiException", QdrantAPIError)

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "error"
    assert "Qdrant" in payload["message"] or "Connection" in payload["message"]


def test_main_neo4j_connection_error(monkeypatch, tmp_path, capsys):
    """Test handling of Neo4j connection failures."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test", "--include-semantic"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])

    class Neo4jConnError(Exception):
        pass

    class FailingDriver:
        def __enter__(self):
            raise Neo4jConnError("Failed to connect to Neo4j")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(ask.GraphDatabase, "driver", lambda *_, **__: FailingDriver())
    monkeypatch.setattr(ask, "Neo4jError", Neo4jConnError)
    capture: dict[str, object] = {}
    _setup_retriever(
        monkeypatch,
        records=[{"chunk_id": "1", "chunk_uid": "uid-1", "text": "test", "score": 0.9}],
        capture=capture,
    )

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "error"


def test_main_with_missing_question_argument(monkeypatch, tmp_path, capsys):
    """Test that missing --question argument is handled appropriately."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py"])

    with pytest.raises(SystemExit):
        ask.main()


def test_main_with_invalid_top_k_value(monkeypatch, tmp_path, capsys):
    """Top-k <= 0 should be clamped to 1."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test", "--top-k", "0"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    ask.main()

    assert capture["top_k"] == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] in {"skipped", "success"}


def test_main_with_large_top_k_value(monkeypatch, tmp_path, capsys):
    """Test handling of very large top_k values."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test", "--top-k", "1000"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    ask.main()

    assert capture["top_k"] == 1000


def test_main_with_empty_question(monkeypatch, tmp_path, capsys):
    """Test handling of empty question string."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", ""])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.0])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    ask.main()

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] in {"skipped", "success"}


def test_main_with_unicode_question(monkeypatch, tmp_path, capsys):
    """Test handling of Unicode characters in questions."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "What is AI? 你好 🤖"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1, 0.2])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    ask.main()

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] in {"skipped", "success"}


def test_main_result_with_missing_optional_fields(monkeypatch, tmp_path, capsys):
    """Test handling of results with missing optional fields like document_name."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test", "--top-k", "1"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(
        monkeypatch,
        records=[
            {
                "chunk_id": "1",
                "chunk_uid": "chunk-1",
                "text": "minimal",
                "score": 0.5,
            }
        ],
        capture=capture,
    )

    ask.main()

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] in {"skipped", "success"}
    assert payload["matches"][0]["chunk_id"] == "1"


def test_main_semantic_context_with_empty_result(monkeypatch, tmp_path, capsys):
    """Test that empty semantic context is handled gracefully."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["ask_qdrant.py", "--question", "Test", "--include-semantic"]
    )
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(
        monkeypatch,
        records=[{"chunk_id": "1", "chunk_uid": "uid-1", "text": "test", "score": 0.9}],
        capture=capture,
    )
    monkeypatch.setattr(ask, "_fetch_semantic_context", lambda *_, **__: {})

    ask.main()

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "success"


def test_main_semantic_context_fetch_error(monkeypatch, tmp_path, capsys):
    """Test handling of errors when fetching semantic context."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["ask_qdrant.py", "--question", "Test", "--include-semantic"]
    )
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(
        monkeypatch,
        records=[{"chunk_id": "1", "chunk_uid": "uid-1", "text": "test", "score": 0.9}],
        capture=capture,
    )

    def failing_semantic_fetch(*args, **kwargs):
        raise Exception("Semantic fetch failed")

    monkeypatch.setattr(ask, "_fetch_semantic_context", failing_semantic_fetch)

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "error"


def test_main_artifact_directory_creation(monkeypatch, tmp_path, capsys):
    """Test that artifact directory is created if it doesn't exist."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    artifact_dir = tmp_path / "artifacts" / "local_stack"
    assert not artifact_dir.exists()

    ask.main()

    assert artifact_dir.exists()
    assert (artifact_dir / "ask_qdrant.json").exists()


def test_main_with_very_long_question(monkeypatch, tmp_path, capsys):
    """Test handling of very long question strings."""
    monkeypatch.chdir(tmp_path)
    long_question = "What is " + "very " * 1000 + "important?"
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", long_question])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1] * 100)
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    ask.main()

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] in {"skipped", "success"}


def test_main_multiple_results_with_same_score(monkeypatch, tmp_path, capsys):
    """Test handling of multiple results with identical scores."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test", "--top-k", "3"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(
        monkeypatch,
        records=[
            {"chunk_id": "1", "chunk_uid": "uid-1", "text": "a", "score": 0.9},
            {"chunk_id": "2", "chunk_uid": "uid-2", "text": "b", "score": 0.9},
            {"chunk_id": "3", "chunk_uid": "uid-3", "text": "c", "score": 0.9},
        ],
        capture=capture,
    )

    ask.main()

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "success"
    assert len(payload["matches"]) == 3
    assert all(m["score"] == pytest.approx(0.9) for m in payload["matches"])


def test_main_result_with_special_characters_in_text(monkeypatch, tmp_path, capsys):
    """Test handling of special characters in chunk text."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(
        monkeypatch,
        records=[
            {
                "chunk_id": "1",
                "chunk_uid": "uid-1",
                "text": 'Special: <>"\\n\\t\\/\b\f\r',
                "score": 0.9,
            }
        ],
        capture=capture,
    )

    ask.main()

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] in {"skipped", "success"}
    artifact_path = tmp_path / "artifacts" / "local_stack" / "ask_qdrant.json"
    saved = json.loads(artifact_path.read_text())
    assert "Special:" in saved["matches"][0]["text"]


def test_main_with_openai_rate_limit_error(monkeypatch, tmp_path, capsys):
    """Test handling of OpenAI rate limit errors."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)

    class RateLimitError(Exception):
        def __init__(self, message, remediation=None):
            super().__init__(message)
            self.remediation = remediation

    monkeypatch.setattr(ask, "OpenAIClientError", RateLimitError)

    class RateLimitedClient:
        def __init__(self, settings):
            pass

        def embedding(self, *, input_text: str):
            raise RateLimitError(
                "Rate limit exceeded", remediation="Wait and retry after some time"
            )

    monkeypatch.setattr(ask, "SharedOpenAIClient", lambda settings: RateLimitedClient(settings))

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "error"
    assert "retry" in payload["message"].lower() or "rate" in payload["message"].lower()


def test_embedding_vector_passed_correctly(monkeypatch, tmp_path, capsys):
    """Test that the embedding vector is correctly passed to the retriever."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    test_vector = [0.123, 0.456, 0.789]
    _setup_shared_client(monkeypatch, vector=test_vector)
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    ask.main()

    assert capture["query_vector"] == test_vector


def test_retrieval_query_structure(monkeypatch, tmp_path, capsys):
    """Test that the retrieval query includes expected Cypher patterns."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    ask.main()

    kwargs = capture["kwargs"]
    query = kwargs["retrieval_query"]
    assert "OPTIONAL MATCH" in query
    assert "Document" in query
    assert "HAS_CHUNK" in query


def test_main_with_json_output_format(monkeypatch, tmp_path, capsys):
    """Test that output is valid JSON in all cases."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(monkeypatch, records=[], capture=capture)

    ask.main()

    payload = json.loads(capsys.readouterr().out.strip())
    assert isinstance(payload, dict)
    assert "status" in payload
    assert "message" in payload
    assert "matches" in payload


def test_scrub_object_called_on_results(monkeypatch, tmp_path, capsys):
    """Test that scrub_object is called on result payloads."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(
        monkeypatch,
        records=[{"chunk_id": "1", "chunk_uid": "uid-1", "text": "test", "score": 0.9}],
        capture=capture,
    )

    scrub_calls = []

    def tracking_scrub(payload):
        scrub_calls.append(payload)
        return payload

    monkeypatch.setattr(ask, "scrub_object", tracking_scrub)

    ask.main()

    assert len(scrub_calls) > 0


def test_main_result_ordering_by_score(monkeypatch, tmp_path, capsys):
    """Test that results maintain score-based ordering from retriever."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test", "--top-k", "3"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(
        monkeypatch,
        records=[
            {"chunk_id": "1", "chunk_uid": "uid-1", "text": "high", "score": 0.95},
            {"chunk_id": "2", "chunk_uid": "uid-2", "text": "medium", "score": 0.75},
            {"chunk_id": "3", "chunk_uid": "uid-3", "text": "low", "score": 0.45},
        ],
        capture=capture,
    )

    ask.main()

    payload = json.loads(capsys.readouterr().out.strip())
    scores = [m["score"] for m in payload["matches"]]
    assert scores == [0.95, 0.75, 0.45]


def test_main_unexpected_exception_handling(monkeypatch, tmp_path, capsys):
    """Test that unexpected exceptions are caught and reported properly."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)

    class UnexpectedError(Exception):
        pass

    class FailingRetriever:
        def __init__(self, *args, **kwargs):
            pass

        def get_search_results(self, **kwargs):
            raise UnexpectedError("Something went wrong unexpectedly")

    monkeypatch.setattr(ask, "QdrantNeo4jRetriever", FailingRetriever)

    with pytest.raises(SystemExit) as excinfo:
        ask.main()

    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "error"


def test_record_to_match_from_data(monkeypatch):
    """_record_to_match handles objects with data() and coerces IDs to strings."""
    class R:
        def data(self):
            return {"chunk_id": 123, "chunk_uid": 456, "text": "t"}

    out = ask._record_to_match(R())
    assert out["chunk_id"] == "123"
    assert out["chunk_uid"] == "456"
    assert out["text"] == "t"


def test_record_to_match_from_dict_with_id_fallback():
    """_record_to_match uses 'id' when 'chunk_id' is absent."""
    out = ask._record_to_match({"id": 99, "score": 0.7})
    assert out["chunk_id"] == "99"
    assert out["score"] == 0.7


def test_record_to_match_from_sequence_mapping():
    """_record_to_match accepts mapping-like sequences."""
    seq = [("chunk_id", 1), ("chunk_uid", "u1")]
    out = ask._record_to_match(seq)
    assert out["chunk_id"] == "1"
    assert out["chunk_uid"] == "u1"


def test_main_score_non_numeric_remains_unchanged(monkeypatch, tmp_path, capsys):
    """Non-numeric score should remain unchanged after normalization attempt."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Test", "--top-k", "1"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1])
    _setup_driver(monkeypatch)
    capture: dict[str, object] = {}
    _setup_retriever(
        monkeypatch,
        records=[{"chunk_id": "1", "chunk_uid": "u1", "text": "x", "score": "oops"}],
        capture=capture,
    )

    ask.main()

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["matches"][0]["score"] == "oops"


def test_fetch_semantic_context_happy_path(monkeypatch):
    """Unit test for _fetch_semantic_context node and relationship shaping."""
    _configure_identity_scrubber(monkeypatch)

    class FakeDriver:
        def execute_query(self, *args, **kwargs):
            return [
                {
                    "chunk_uid": "U1",
                    "nodes": [
                        {"id": "1:Person", "labels": ["Person"], "properties": {"name": "Alice"}}
                    ],
                    "relationships": [
                        None,
                        {
                            "type": "KNOWS",
                            "start": "1:Person",
                            "end": "2:Person",
                            "properties": {"w": 1.0},
                        },
                    ],
                }
            ]

    out = ask._fetch_semantic_context(FakeDriver(), database="db", chunk_uids=["U1"])
    assert "U1" in out
    assert out["U1"]["nodes"][0]["id"] == "1:Person"
    assert out["U1"]["nodes"][0]["labels"] == ["Person"]
    assert out["U1"]["nodes"][0]["properties"]["name"] == "Alice"
    assert out["U1"]["relationships"][0]["type"] == "KNOWS"


def test_fetch_semantic_context_no_ids(monkeypatch):
    """_fetch_semantic_context returns empty dict for empty input."""
    _configure_identity_scrubber(monkeypatch)
    class FakeDriver:
        def execute_query(self, *args, **kwargs):
            raise AssertionError("execute_query should not be called when no chunk_uids")
    out = ask._fetch_semantic_context(FakeDriver(), database=None, chunk_uids=[])
    assert out == {}
