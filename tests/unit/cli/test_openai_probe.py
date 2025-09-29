import json
from pathlib import Path
from types import SimpleNamespace

from cli import diagnostics, openai_client


class FakeChatResponse(SimpleNamespace):
    pass


class FakeEmbeddingResponse(SimpleNamespace):
    pass


class HappyClient:
    def __init__(self):
        """
        Initialize a simulated OpenAI-like client used by tests.
        
        Creates a `chat` namespace exposing `completions.create` and an `embeddings` namespace exposing `create`, both bound to the instance's internal handlers, and initializes call counters under `self.calls` for "chat" and "embedding".
        """
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embedding)
        self.calls = {"chat": 0, "embedding": 0}

    def _chat(self, **kwargs):
        """
        Simulate a chat API call: increments the chat call counter and returns a canned successful chat response.
        
        Parameters:
            kwargs: Must include a 'model' key identifying the model to use; other keys are ignored.
        
        Returns:
            FakeChatResponse: A mock response with `usage` (prompt_tokens=12, completion_tokens=3) and a single choice whose `finish_reason` is `"stop"`.
        """
        self.calls["chat"] += 1
        assert kwargs["model"]
        return FakeChatResponse(
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
            choices=[SimpleNamespace(finish_reason="stop")],
        )

    def _embedding(self, **kwargs):
        """
        Create a fake embedding response and record that an embedding call occurred.
        
        Parameters:
            kwargs: Must include a "model" key; an AssertionError is raised if it is missing.
        
        Returns:
            FakeEmbeddingResponse: Contains `data` with one embedding vector of length 1536 and `usage.total_tokens` equal to 8.
        """
        self.calls["embedding"] += 1
        assert kwargs["model"]
        return FakeEmbeddingResponse(
            data=[SimpleNamespace(embedding=[0.0] * 1536)],
            usage=SimpleNamespace(total_tokens=8),
        )


def _read_json(path: Path) -> dict:
    """
    Read and parse JSON from the given path.
    
    Parameters:
        path (Path): Path to a UTF-8 encoded JSON file.
    
    Returns:
        dict: Parsed JSON content.
    """
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

    class FlakyClient(HappyClient):
        def __init__(self):
            """
            Initialize the client and reset the chat attempts counter.
            
            Calls the superclass constructor and sets `chat_attempts` to 0 to track how many chat calls have been made.
            """
            super().__init__()
            self.chat_attempts = 0

        def _chat(self, **kwargs):
            """
            Simulates intermittent rate limiting by raising FakeRateLimit for the first two invocations, then delegates to the superclass chat implementation.
            
            Raises:
                FakeRateLimit: for the first two calls to simulate a rate-limit error.
            
            Returns:
                The chat response returned by the superclass `_chat`.
            """
            self.chat_attempts += 1
            if self.chat_attempts < 3:
                response = SimpleNamespace(
                    request=SimpleNamespace(),
                    status_code=429,
                    headers={"Retry-After": "1", "retry-after": "1"},
                )
                raise openai_client.RateLimitError("slow down", response=response, body=None)
            return super()._chat(**kwargs)

    sleeps: list[float] = []
    root = tmp_path / "repo"
    root.mkdir()

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
    assert sleeps == [1.0, 1.0]
    report = _read_json(root / "artifacts" / "openai" / "probe.json")
    assert report["chat"]["status"] == "success"


def test_openai_probe_rate_limit_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_ACTOR", "pytest-actor")

    class AlwaysRateLimited(HappyClient):
        def _chat(self, **kwargs):
            """
            Simulate a chat API call that always fails due to a rate limit.
            
            All keyword arguments are accepted and ignored.
            
            Raises:
                openai_client.RateLimitError: always raised with the message "token budget exceeded".
            """
            response = SimpleNamespace(
                request=SimpleNamespace(),
                status_code=429,
                headers={},
            )
            raise openai_client.RateLimitError("token budget exceeded", response=response, body=None)

    root = tmp_path / "repo"
    root.mkdir()

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
    assert report["error"]["details"]["reason"] == "rate_limit"


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
