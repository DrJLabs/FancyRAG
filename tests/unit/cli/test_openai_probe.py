import json
from pathlib import Path
from types import SimpleNamespace

from cli import diagnostics


class FakeChatResponse(SimpleNamespace):
    pass


class FakeEmbeddingResponse(SimpleNamespace):
    pass


class HappyClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embedding)
        self.calls = {"chat": 0, "embedding": 0}

    def _chat(self, **kwargs):
        self.calls["chat"] += 1
        assert kwargs["model"]
        return FakeChatResponse(
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
            choices=[SimpleNamespace(finish_reason="stop")],
        )

    def _embedding(self, **kwargs):
        self.calls["embedding"] += 1
        assert kwargs["model"]
        return FakeEmbeddingResponse(
            data=[SimpleNamespace(embedding=[0.0] * 1536)],
            usage=SimpleNamespace(total_tokens=8),
        )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_openai_probe_success(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_ACTOR", "pytest-actor")
    client = HappyClient()
    root = tmp_path / "repo"
    root.mkdir()

    artifacts_dir = root / "artifacts" / "openai"

    exit_code = diagnostics.run_openai_probe(
        root,
        artifacts_dir=artifacts_dir,
        skip_live=False,
        max_attempts=3,
        base_delay=0.01,
        sleep_fn=lambda *_: None,
        client_factory=lambda: client,
    )

    assert exit_code == 0
    report = _read_json(artifacts_dir / "probe.json")
    metrics_text = (artifacts_dir / "metrics.prom").read_text(encoding="utf-8")

    assert report["status"] == "success"
    assert report["chat"]["finish_reason"] == "stop"
    assert report["embedding"]["vector_length"] == 1536
    assert report["settings"]["chat_override"] is False
    assert 'graphrag_openai_chat_latency_ms_bucket' in metrics_text
    assert 'le="100.0"' in metrics_text
    assert 'le="5000.0"' in metrics_text


def test_openai_probe_identifies_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_ACTOR", "pytest-actor")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    client = HappyClient()
    root = tmp_path / "repo"
    root.mkdir()

    exit_code = diagnostics.run_openai_probe(
        root,
        artifacts_dir=root / "artifacts" / "openai",
        skip_live=False,
        max_attempts=2,
        base_delay=0.01,
        sleep_fn=lambda *_: None,
        client_factory=lambda: client,
    )

    assert exit_code == 0
    report = _read_json(root / "artifacts" / "openai" / "probe.json")
    assert report["settings"]["chat_override"] is True
    assert report["chat"]["fallback_used"] is True


def test_openai_probe_rate_limit_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_ACTOR", "pytest-actor")

    class FakeRateLimit(Exception):
        pass

    class FlakyClient(HappyClient):
        def __init__(self):
            super().__init__()
            self.chat_attempts = 0

        def _chat(self, **kwargs):
            self.chat_attempts += 1
            if self.chat_attempts < 3:
                raise FakeRateLimit("slow down")
            return super()._chat(**kwargs)

    sleeps: list[float] = []
    root = tmp_path / "repo"
    root.mkdir()

    monkeypatch.setattr(diagnostics, "_is_rate_limit_error", lambda exc: isinstance(exc, FakeRateLimit))

    exit_code = diagnostics.run_openai_probe(
        root,
        artifacts_dir=root / "artifacts" / "openai",
        skip_live=False,
        max_attempts=3,
        base_delay=0.01,
        sleep_fn=lambda duration: sleeps.append(duration),
        client_factory=FlakyClient,
    )

    assert exit_code == 0
    assert sleeps == [0.01, 0.02]
    report = _read_json(root / "artifacts" / "openai" / "probe.json")
    assert report["chat"]["status"] == "success"


def test_openai_probe_rate_limit_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_ACTOR", "pytest-actor")

    class FakeRateLimit(Exception):
        pass

    class AlwaysRateLimited(HappyClient):
        def _chat(self, **kwargs):
            raise FakeRateLimit("token budget exceeded")

    root = tmp_path / "repo"
    root.mkdir()

    monkeypatch.setattr(diagnostics, "_is_rate_limit_error", lambda exc: isinstance(exc, FakeRateLimit))

    exit_code = diagnostics.run_openai_probe(
        root,
        artifacts_dir=root / "artifacts" / "openai",
        skip_live=False,
        max_attempts=2,
        base_delay=0.01,
        sleep_fn=lambda *_: None,
        client_factory=AlwaysRateLimited,
    )

    assert exit_code == 1
    report = _read_json(root / "artifacts" / "openai" / "probe.json")
    assert report["status"] == "failed"
    assert "token budgets" in report["error"]["remediation"].lower()


def test_openai_probe_skip_live(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_ACTOR", "pytest-actor")
    root = tmp_path / "repo"
    root.mkdir()

    exit_code = diagnostics.run_openai_probe(
        root,
        artifacts_dir=root / "artifacts" / "openai",
        skip_live=True,
        max_attempts=1,
        base_delay=0.01,
        sleep_fn=lambda *_: None,
    )

    assert exit_code == 0
    report = _read_json(root / "artifacts" / "openai" / "probe.json")
    assert report["status"] == "skipped"
