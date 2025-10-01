from __future__ import annotations

import json
import sys
from importlib import util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

stub_neo4j = ModuleType("neo4j")
stub_neo4j.GraphDatabase = SimpleNamespace(driver=lambda *_, **__: None)
sys.modules.setdefault("neo4j", stub_neo4j)
stub_neo4j_exceptions = ModuleType("neo4j.exceptions")
stub_neo4j_exceptions.Neo4jError = Exception
sys.modules.setdefault("neo4j.exceptions", stub_neo4j_exceptions)

stub_qdrant = ModuleType("qdrant_client")


class _StubQdrantClient:  # pragma: no cover - import stub
    def __init__(self, *_, **__):
        pass


stub_qdrant.QdrantClient = _StubQdrantClient
sys.modules.setdefault("qdrant_client", stub_qdrant)

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
            self.inputs: list[str] = []

        def embedding(self, *, input_text: str):
            self.inputs.append(input_text)
            return SimpleNamespace(vector=vector)

    monkeypatch.setattr(ask, "SharedOpenAIClient", lambda settings: FakeClient(settings))


def _setup_driver(monkeypatch, record):
    class FakeDriver:
        def execute_query(self, query, params, database_=None):
            _ = database_
            return ([{**record, "chunk_id": params["chunk_id"]}], None, None)

    class FakeDriverCtx:
        def __enter__(self):
            return FakeDriver()

        def __exit__(self, _exc_type, _exc, _tb):
            return False

    monkeypatch.setattr(ask.GraphDatabase, "driver", lambda uri, auth: FakeDriverCtx())


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
            self.settings = settings

        def embedding(self, *, input_text: str):
            self.last_question = input_text
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
