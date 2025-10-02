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
    stub_neo4j.__spec__ = ModuleSpec("neo4j", loader=None)
    sys.modules.setdefault("neo4j", stub_neo4j)
else:
    if getattr(existing_neo4j, "__spec__", None) is None:
        existing_neo4j.__spec__ = ModuleSpec("neo4j", loader=None)
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


def _setup_driver(monkeypatch, record):
    class FakeDriver:
        def execute_query(self, query, params, database_=None):  # noqa: ARG002
            return ([{**record, "chunk_id": params["chunk_id"]}], None, None)

    class FakeDriverCtx:
        def __enter__(self):
            return FakeDriver()

        def __exit__(self, _exc_type, _exc, _tb):
            return False

    monkeypatch.setattr(ask.GraphDatabase, "driver", lambda *_, **__: FakeDriverCtx())


def _setup_qdrant(monkeypatch, points):
    monkeypatch.setattr(ask, "_query_qdrant", lambda *_, **__: points)


def _configure_identity_scrubber(monkeypatch):
    monkeypatch.setattr(ask, "scrub_object", lambda payload: payload)


def test_main_success_creates_artifact(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "How?", "--top-k", "2"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.1, 0.2, 0.3])
    _setup_qdrant(
        monkeypatch,
        [
            SimpleNamespace(payload={"chunk_id": "1"}, id="1", score=0.99),
            SimpleNamespace(payload=None, id="2", score=0.42),
        ],
    )
    _setup_driver(
        monkeypatch,
        {
            "text": "example chunk",
            "source_path": "doc.txt",
            "document_name": "Doc",
            "document_source_path": "doc.txt",
        },
    )

    ask.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["status"] == "success"
    assert payload["message"] == "Retrieved 2 matches"
    artifact_path = tmp_path / "artifacts" / "local_stack" / "ask_qdrant.json"
    saved = json.loads(artifact_path.read_text())
    assert saved["matches"][0]["chunk_id"] == "1"
    assert saved["matches"][1]["chunk_id"] == "2"


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


def test_query_qdrant_fallback_to_search():
    """_query_qdrant should fallback to client.search when query_points is unavailable/legacy."""
    class LegacyClient:
        def query_points(self, *_, **__):
            raise AttributeError
        def search(self, *_, **__):
            return [SimpleNamespace(payload={"chunk_id": "x"}, id="x", score=0.1)]
    results = ask._query_qdrant(LegacyClient(), collection="coll", vector=[0.1], limit=1)
    assert isinstance(results, list)
    assert results and results[0].payload["chunk_id"] == "x"


def test_query_qdrant_query_points_modern_api():
    """_query_qdrant should use modern query_points API when available."""
    class ModernClient:
        def query_points(self, *_, **__):
            return SimpleNamespace(points=[SimpleNamespace(payload={"chunk_id": "y"}, id="y", score=0.2)])
    results = ask._query_qdrant(ModernClient(), collection="coll", vector=[0.2], limit=1)
    assert len(results) == 1
    assert results[0].payload["chunk_id"] == "y"


def test_main_with_custom_collection_populates_log(monkeypatch, tmp_path, capsys):
    """Ensure --collection is propagated to the output log and artifact."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ask_qdrant.py", "--question", "Which?", "--collection", "alt_collection"])
    monkeypatch.setattr(ask, "_load_settings", lambda: object())
    _configure_identity_scrubber(monkeypatch)
    _setup_shared_client(monkeypatch, vector=[0.33, 0.44])
    _setup_qdrant(monkeypatch, [SimpleNamespace(payload={"chunk_id": "c1"}, id="c1", score=0.88)])
    _setup_driver(monkeypatch, {"text": "ctx", "source_path": "doc.txt", "document_name": "Doc", "document_source_path": "doc.txt"})

    ask.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["status"] == "success"
    assert payload["collection"] == "alt_collection"

    artifact_path = tmp_path / "artifacts" / "local_stack" / "ask_qdrant.json"
    saved = json.loads(artifact_path.read_text())
    assert saved["collection"] == "alt_collection"
