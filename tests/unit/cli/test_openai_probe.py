import json
import httpx
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
        
        Creates a `responses` namespace exposing `create` and an `embeddings` namespace exposing `create`, both bound to the instance's internal handlers, and initializes call counters under `self.calls` for "responses" and "embedding".
        """
        self.responses = SimpleNamespace(create=self._response)
        self.embeddings = SimpleNamespace(create=self._embedding)
        self.calls = {"responses": 0, "embedding": 0}

    def _response(self, **kwargs):
        """
        Simulate a responses API call: increments the responses call counter and returns a canned successful response.
        
        Parameters:
            kwargs: Must include a 'model' key identifying the model to use; other keys are ignored.
        
        Returns:
            FakeChatResponse: A mock response with `usage` (input_tokens=12, output_tokens=3) and an `output_text` payload.
        """
        self.calls["responses"] += 1
        assert kwargs["model"]
        return FakeChatResponse(
            usage=SimpleNamespace(input_tokens=12, output_tokens=3),
            output_text="Probe acknowledgement",
        )

    def _embedding(self, **kwargs):
        """
        Create a fake embedding response and record that an embedding call occurred.
        
        Parameters:
            kwargs: Must include a "model" key; an AssertionError is raised if it is missing.
        
        Returns:
            FakeEmbeddingResponse: Contains `data` with one embedding vector of length 1024 and `usage.total_tokens` equal to 8.
        """
        self.calls["embedding"] += 1
        assert kwargs["model"]
        return FakeEmbeddingResponse(
            data=[SimpleNamespace(embedding=[0.0] * 1024)],
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


def _make_httpx_response(*, status_code: int = 429, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )


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
    assert report["chat"]["finish_reason"] is None
    assert report["embedding"]["vector_length"] == 1024
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


def test_openai_probe_records_base_url_override(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_ACTOR", "probe-actor")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")

    root = tmp_path / "repo"
    root.mkdir()

    exit_code = diagnostics.run_openai_probe(
        root,
        artifacts_dir=root / "artifacts" / "openai",
        skip_live=True,
        max_attempts=1,
        base_delay=0.1,
    )

    assert exit_code == 0
    report = _read_json(root / "artifacts" / "openai" / "probe.json")
    settings = report["settings"]
    assert settings["base_url_override"] is True
    assert settings["base_url_masked"] == "https://***/v1"
    # raw value is sanitized in report but should not leak hostname
    assert "gateway.example.com" not in json.dumps(report)


def test_openai_probe_rejects_http_base_url_without_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_ACTOR", "probe-actor")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gateway.example.com/v1")

    root = tmp_path / "repo"
    root.mkdir()

    exit_code = diagnostics.run_openai_probe(
        root,
        artifacts_dir=root / "artifacts" / "openai",
        skip_live=True,
        max_attempts=1,
        base_delay=0.1,
    )

    assert exit_code == 1
    report = _read_json(root / "artifacts" / "openai" / "probe.json")
    assert report["status"] == "failed"
    assert report["error"]["details"]["message"].startswith("OPENAI_BASE_URL must use https")


def test_openai_probe_allows_http_base_url_with_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_ACTOR", "probe-actor")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gateway.example.com/v1")
    monkeypatch.setenv("OPENAI_ALLOW_INSECURE_BASE_URL", "true")

    root = tmp_path / "repo"
    root.mkdir()

    exit_code = diagnostics.run_openai_probe(
        root,
        artifacts_dir=root / "artifacts" / "openai",
        skip_live=True,
        max_attempts=1,
        base_delay=0.1,
    )

    assert exit_code == 0
    report = _read_json(root / "artifacts" / "openai" / "probe.json")
    assert report["settings"]["base_url_override"] is True
    assert report["settings"]["allow_insecure_base_url"] is True


def test_openai_probe_rate_limit_retries(tmp_path, monkeypatch):
    """
    Verify that the OpenAI probe retries on rate-limit errors and succeeds after the specified attempts.
    
    Uses a FlakyClient that raises RateLimitError for the first two chat calls with Retry-After headers, runs the probe with retry settings, and asserts that the probe exits successfully, the recorded sleep durations match the retry backoffs ([1.0, 1.0]), and the generated probe report marks the chat status as "success".
    """
    monkeypatch.setenv("GRAPH_RAG_ACTOR", "pytest-actor")

    class FlakyClient(HappyClient):
        def __init__(self):
            """
            Initialize the client and reset the chat attempts counter.
            
            Calls the superclass constructor and sets `chat_attempts` to 0 to track how many chat calls have been made.
            """
            super().__init__()
            self.chat_attempts = 0

        def _response(self, **kwargs):
            """
            Simulates intermittent rate limiting for tests by raising a RateLimitError on the first two calls.
            
            On the first two invocations this method raises openai_client.RateLimitError configured to mimic an HTTP 429 response (including a Retry-After header). On subsequent invocations it delegates to and returns the superclass `_response` result.
            
            Returns:
                The chat response returned by the superclass `_chat`.
            
            Raises:
                openai_client.RateLimitError: Raised for the first two invocations to simulate an HTTP 429 rate-limit response.
            """
            self.chat_attempts += 1
            if self.chat_attempts < 3:
                response = _make_httpx_response(headers={"Retry-After": "1", "retry-after": "1"})
                raise openai_client.RateLimitError("slow down", response=response, body={})
            return super()._response(**kwargs)

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
    """
    Verify that run_openai_probe reports a rate-limit failure when every chat attempt raises a rate-limit error.
    
    This test creates a client whose chat method always raises openai_client.RateLimitError, runs the probe with limited retries, and asserts the process exits with code 1 and that the generated probe.json records a failure with error details reason "rate_limit".
    """
    monkeypatch.setenv("GRAPH_RAG_ACTOR", "pytest-actor")

    class AlwaysRateLimited(HappyClient):
        def _response(self, **kwargs):
            """
            Simulate a chat API call that always raises an OpenAI rate-limit error.
            
            All keyword arguments are accepted and ignored.
            
            Raises:
                openai_client.RateLimitError: Raised with message "token budget exceeded"; the attached response has status_code 429 and empty headers, and an empty `body` dict.
            """
            response = _make_httpx_response(headers={})
            raise openai_client.RateLimitError("token budget exceeded", response=response, body={})

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
