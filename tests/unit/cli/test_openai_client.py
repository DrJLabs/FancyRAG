import pytest
from types import SimpleNamespace

from _compat.structlog import capture_logs

from cli.openai_client import OpenAIClientError, RateLimitError, SharedOpenAIClient
from cli.telemetry import create_metrics
from config.settings import OpenAISettings

SLOW_DOWN_MESSAGE = "slow down"


class FakeClock:
    def __init__(self, step: float = 0.005) -> None:
        self._value = 0.0
        self._step = step

    def __call__(self) -> float:
        current = self._value
        self._value += self._step
        return current


def _stub_response(*, status_code: int = 429, headers: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=status_code,
        headers=headers or {},
        request=SimpleNamespace(),
    )


class StubChatResponse(SimpleNamespace):
    pass


class StubEmbeddingResponse(SimpleNamespace):
    pass


class StubOpenAIClient:
    def __init__(self) -> None:
        self.chat_calls: list[dict] = []
        self.embedding_calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.responses = SimpleNamespace(create=self._responses)
        self.embeddings = SimpleNamespace(create=self._embedding)

    def _chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return StubChatResponse(
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
            choices=[SimpleNamespace(finish_reason="stop")],
        )

    def _responses(self, **kwargs):
        self.chat_calls.append(kwargs)
        messages = kwargs.get("input") or []
        last = messages[-1].get("content", "") if messages else ""
        content = f"Stubbed response for: {last}".strip()
        usage = SimpleNamespace(prompt_tokens=4, completion_tokens=2)
        return StubChatResponse(
            usage=usage,
            output_text=content,
            output=[SimpleNamespace(content=[SimpleNamespace(text=content)])],
            choices=[SimpleNamespace(finish_reason="stop")],
        )

    def _embedding(self, **kwargs):
        self.embedding_calls.append(kwargs)
        return StubEmbeddingResponse(
            data=[SimpleNamespace(embedding=[0.0] * 1536)],
            usage=SimpleNamespace(total_tokens=6),
        )


class DimensionMismatchClient(StubOpenAIClient):
    def _embedding(self, **kwargs):
        self.embedding_calls.append(kwargs)
        return StubEmbeddingResponse(
            data=[SimpleNamespace(embedding=[0.0] * 10)],
            usage=SimpleNamespace(total_tokens=1),
        )


class FlakyChatClient(StubOpenAIClient):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self._failures = failures

    def _responses(self, **kwargs):
        """
        Simulate a chat request that may fail with rate limiting for a configured number of initial calls.

        If configured failures remain, this method raises a RateLimitError containing a 429 status and Retry-After headers; otherwise it returns the underlying client's chat response.

        Returns:
            The chat response object produced by the underlying client.

        Raises:
            RateLimitError: If the client is within its configured failure period; the error includes a 429 status and Retry-After headers.
        """
        if self._failures > 0:
            self._failures -= 1
            headers = {"Retry-After": "1", "retry-after": "1"}
            response = SimpleNamespace(headers=headers, request=SimpleNamespace(), status_code=429)
            raise RateLimitError(SLOW_DOWN_MESSAGE, response=response, body={})
        return super()._responses(**kwargs)


class SequencedRateLimitClient(StubOpenAIClient):
    def __init__(self, retry_after: list[str], *, always_fail: bool = False) -> None:
        """
        Initialize a SequencedRateLimitClient that simulates rate-limited responses with a configurable sequence of `Retry-After` header values.

        Parameters:
            retry_after (list[str]): Sequence of `Retry-After` header values to use for successive rate-limit responses.
            always_fail (bool): If `True`, the client will always raise rate-limit errors; if `False`, it will stop raising after the sequence is exhausted.

        Notes:
            Initializes an internal attempt counter to track how many times the client has been invoked.
        """
        super().__init__()
        self._retry_after = retry_after
        self._always_fail = always_fail
        self._attempts = 0

    def _responses(self, **kwargs):
        """
        Simulate a chat call that enforces rate-limit responses based on a configured Retry-After sequence.

        When configured to fail for the current attempt, raises a RateLimitError whose response includes "Retry-After" and "retry-after" headers and has status_code 429; otherwise returns the superclass chat response.

        Returns:
            The chat response returned by the superclass `_chat` when no rate-limit is applied.

        Raises:
            RateLimitError: if this call is simulated as rate-limited; the exception's `response` contains `Retry-After` and `retry-after` headers and `status_code` 429.
        """
        header_value = "1"
        if self._retry_after:
            index = min(self._attempts, len(self._retry_after) - 1)
            header_value = self._retry_after[index]
        self._attempts += 1
        if self._always_fail or self._attempts <= len(self._retry_after):
            response = SimpleNamespace(
                headers={"Retry-After": header_value, "retry-after": header_value},
                request=SimpleNamespace(),
                status_code=429,
            )
            raise RateLimitError(SLOW_DOWN_MESSAGE, response=response, body={})
        return super()._responses(**kwargs)


def _make_client(env: dict[str, str], *, client) -> SharedOpenAIClient:
    configured = env.copy()
    configured.setdefault("OPENAI_EMBEDDING_DIMENSIONS", "1536")
    settings = OpenAISettings.load(configured, actor="pytest")
    return SharedOpenAIClient(
        settings,
        client=client,
        embedding_client=client,
        metrics=create_metrics(),
        sleep_fn=lambda *_: None,
        clock=FakeClock(),
    )


def test_shared_client_initializes_openai_with_base_url(monkeypatch):
    captured: dict[str, object] = {}
    stub = StubOpenAIClient()

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return stub

    monkeypatch.setattr("cli.openai_client.OpenAI", fake_openai)
    env = {"OPENAI_BASE_URL": "https://gateway.example.com/v1"}
    settings = OpenAISettings.load(env, actor="pytest")

    with capture_logs() as logs:
        client = SharedOpenAIClient(
            settings,
            metrics=create_metrics(),
            sleep_fn=lambda *_: None,
            clock=FakeClock(),
        )

    assert captured["base_url"] == "https://gateway.example.com/v1"
    assert client._chat_client is stub
    assert any(
        entry["event"] == "openai.client.base_url_override"
        and entry.get("base_url") == "https://***/v1"
        for entry in logs
    )


def test_shared_client_omits_base_url_when_not_configured(monkeypatch):
    captured: dict[str, object] = {}
    stub = StubOpenAIClient()

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return stub

    monkeypatch.setattr("cli.openai_client.OpenAI", fake_openai)
    settings = OpenAISettings.load({}, actor="pytest")

    SharedOpenAIClient(
        settings,
        metrics=create_metrics(),
        sleep_fn=lambda *_: None,
        clock=FakeClock(),
    )

    assert "base_url" not in captured


def test_chat_completion_uses_default_model():
    stub = StubOpenAIClient()
    client = _make_client({}, client=stub)

    result = client.chat_completion(messages=[{"role": "user", "content": "ping"}])

    assert result.model == OpenAISettings.load({}, actor="pytest").chat_model
    assert result.fallback_used is False
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 2
    assert stub.chat_calls[0]["model"] == result.model


def test_chat_completion_marks_fallback_when_override():
    stub = StubOpenAIClient()
    client = _make_client({"OPENAI_MODEL": "gpt-4o-mini"}, client=stub)

    result = client.chat_completion(messages=[{"role": "user", "content": "ping"}])

    assert result.fallback_used is True
    assert stub.chat_calls[0]["model"] == "gpt-4o-mini"


def test_embedding_dimension_mismatch_raises():
    client = _make_client({}, client=DimensionMismatchClient())

    with pytest.raises(OpenAIClientError) as exc:
        client.embedding(input_text="check")
    assert "Embedding length" in str(exc.value)


def test_embedding_requires_base_url_when_not_configured():
    stub = StubOpenAIClient()
    settings = OpenAISettings.load({}, actor="pytest")
    client = SharedOpenAIClient(
        settings,
        client=stub,
        metrics=create_metrics(),
        sleep_fn=lambda *_: None,
        clock=FakeClock(),
    )

    with pytest.raises(OpenAIClientError) as exc:
        client.embedding(input_text="test")
    assert exc.value.details["reason"] == "embedding_base_url_missing"


def test_retry_after_header_controls_backoff():
    sleeps: list[float] = []
    stub = FlakyChatClient(failures=2)
    settings = OpenAISettings.load({}, actor="pytest")
    shared = SharedOpenAIClient(
        settings,
        client=stub,
        metrics=create_metrics(),
        sleep_fn=lambda duration: sleeps.append(round(duration, 2)),
        clock=FakeClock(),
    )

    result = shared.chat_completion(messages=[{"role": "user", "content": "ping"}])

    assert result.prompt_tokens == 4
    assert sleeps == [1.0, 1.0]


def test_retry_after_sequence_uses_largest_header():
    sleeps: list[float] = []
    stub = SequencedRateLimitClient(["1", "2"])
    shared = SharedOpenAIClient(
        OpenAISettings.load({}, actor="pytest"),
        client=stub,
        metrics=create_metrics(),
        sleep_fn=lambda duration: sleeps.append(round(duration, 1)),
        clock=FakeClock(),
    )

    shared.chat_completion(messages=[{"role": "user", "content": "ping"}])

    assert sleeps == [1.0, 2.0]


def test_retry_after_exhaustion_raises_client_error():
    stub = SequencedRateLimitClient(["1"], always_fail=True)
    shared = SharedOpenAIClient(
        OpenAISettings.load({"OPENAI_MAX_ATTEMPTS": "3"}, actor="pytest"),
        client=stub,
        metrics=create_metrics(),
        sleep_fn=lambda *_: None,
        clock=FakeClock(),
    )

    with pytest.raises(OpenAIClientError) as exc:
        shared.chat_completion(messages=[{"role": "user", "content": "ping"}])

    assert exc.value.details["reason"] == "rate_limit"


def test_shared_client_uses_default_metrics():
    """Test SharedOpenAIClient creates default metrics when not provided."""
    stub = StubOpenAIClient()
    settings = OpenAISettings.load({}, actor="pytest")
    
    client = SharedOpenAIClient(
        settings,
        client=stub,
        sleep_fn=lambda *_: None,
        clock=FakeClock(),
    )
    
    assert client._metrics is not None


def test_shared_client_respects_custom_clock():
    """Test SharedOpenAIClient uses provided clock function."""
    stub = StubOpenAIClient()
    settings = OpenAISettings.load({}, actor="pytest")
    
    clock_calls = []
    def custom_clock():
        val = len(clock_calls) * 0.01
        clock_calls.append(val)
        return val
    
    client = SharedOpenAIClient(
        settings,
        client=stub,
        metrics=create_metrics(),
        sleep_fn=lambda *_: None,
        clock=custom_clock,
    )
    
    client.chat_completion(messages=[{"role": "user", "content": "test"}])
    assert len(clock_calls) > 0


def test_embedding_returns_expected_result_fields():
    """Test embedding returns EmbeddingResult with all fields."""
    stub = StubOpenAIClient()
    client = _make_client({}, client=stub)
    
    result = client.embedding(input_text="test embedding")
    
    assert result.model == "text-embedding-3-small"
    assert isinstance(result.latency_ms, float)
    assert len(result.vector) == 1536
    assert result.tokens_consumed == 6
    assert result.raw_response is not None


def test_embedding_with_override_dimensions():
    """Test embedding accepts override_dimensions parameter."""
    stub = StubOpenAIClient()
    client = _make_client({}, client=stub)
    
    # Should work with matching dimensions
    result = client.embedding(input_text="test", override_dimensions=1536)
    assert len(result.vector) == 1536


def test_embedding_dimension_mismatch_with_override():
    """Test embedding raises when override doesn't match actual."""
    client = _make_client({}, client=DimensionMismatchClient())
    
    with pytest.raises(OpenAIClientError) as exc:
        client.embedding(input_text="test", override_dimensions=1536)
    
    assert "Embedding length" in str(exc.value)


def test_embedding_no_data_raises_error():
    """Test embedding raises when response has no data."""
    class NoDataClient(StubOpenAIClient):
        def _embedding(self, **kwargs):
            self.embedding_calls.append(kwargs)
            return StubEmbeddingResponse(data=[], usage=SimpleNamespace(total_tokens=0))
    
    client = _make_client({}, client=NoDataClient())
    
    with pytest.raises(OpenAIClientError) as exc:
        client.embedding(input_text="test")
    
    assert "did not include any vector payload" in str(exc.value)
    assert exc.value.remediation


def test_chat_completion_passes_extra_params():
    """Test chat_completion forwards extra parameters."""
    stub = StubOpenAIClient()
    client = _make_client({"OPENAI_MODEL": "gpt-4o-mini"}, client=stub)
    
    client.chat_completion(
        messages=[{"role": "user", "content": "test"}],
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        custom_param="value"
    )
    
    assert stub.chat_calls[0]["temperature"] == 0.7
    assert stub.chat_calls[0]["max_output_tokens"] == 100
    assert stub.chat_calls[0]["top_p"] == 0.9
    assert stub.chat_calls[0]["custom_param"] == "value"


def test_chat_completion_omits_temperature_for_gpt5():
    stub = StubOpenAIClient()
    client = _make_client({}, client=stub)

    client.chat_completion(
        messages=[{"role": "user", "content": "test"}],
        temperature=0.7,
        max_tokens=100,
    )

    assert "temperature" not in stub.chat_calls[0]


def test_embedding_passes_extra_params():
    """Test embedding forwards extra parameters."""
    stub = StubOpenAIClient()
    client = _make_client({}, client=stub)
    
    client.embedding(input_text="test", encoding_format="float", user="test-user")
    
    assert stub.embedding_calls[0]["encoding_format"] == "float"
    assert stub.embedding_calls[0]["user"] == "test-user"


def test_chat_fallback_only_when_default_model_rate_limited():
    """Test fallback only triggers for default model rate limits."""
    stub = FlakyChatClient(failures=10)
    settings = OpenAISettings.load(
        {"OPENAI_MODEL": "gpt-4o-mini"},  # Non-default model
        actor="pytest"
    )
    client = SharedOpenAIClient(
        settings,
        client=stub,
        metrics=create_metrics(),
        sleep_fn=lambda *_: None,
        clock=FakeClock(),
    )
    
    with pytest.raises(OpenAIClientError):
        client.chat_completion(messages=[{"role": "user", "content": "test"}])


def test_chat_fallback_disabled_prevents_fallback():
    """Test fallback doesn't trigger when disabled."""
    stub = FlakyChatClient(failures=10)
    settings = OpenAISettings.load(
        {"OPENAI_ENABLE_FALLBACK": "false"},
        actor="pytest"
    )
    client = SharedOpenAIClient(
        settings,
        client=stub,
        metrics=create_metrics(),
        sleep_fn=lambda *_: None,
        clock=FakeClock(),
    )
    
    with pytest.raises(OpenAIClientError) as exc:
        client.chat_completion(messages=[{"role": "user", "content": "test"}])
    
    assert exc.value.details["reason"] == "rate_limit"


def test_is_retryable_recognizes_rate_limit():
    """Test _is_retryable identifies rate limit errors."""
    from cli.openai_client import SharedOpenAIClient

    error = RateLimitError("rate limited", response=_stub_response(), body={})
    assert SharedOpenAIClient._is_retryable(error) is True


def test_is_retryable_recognizes_retryable_status_codes():
    """Test _is_retryable identifies retryable HTTP status codes."""
    from cli.openai_client import APIStatusError, SharedOpenAIClient

    for code in [408, 409, 425, 429, 500, 502, 503, 504]:
        error = APIStatusError(
            "error",
            response=_stub_response(status_code=code),
            body={},
        )
        assert SharedOpenAIClient._is_retryable(error) is True


def test_is_retryable_rejects_non_retryable_codes():
    """Test _is_retryable rejects non-retryable status codes."""
    from cli.openai_client import APIStatusError, SharedOpenAIClient

    for code in [400, 401, 403, 404]:
        error = APIStatusError(
            "error",
            response=_stub_response(status_code=code),
            body={},
        )
        assert SharedOpenAIClient._is_retryable(error) is False


def test_next_delay_uses_retry_after_header():
    """Test _next_delay extracts and uses Retry-After header."""
    settings = OpenAISettings.load({}, actor="pytest")
    client = SharedOpenAIClient(
        settings,
        client=StubOpenAIClient(),
        metrics=create_metrics(),
        sleep_fn=lambda *_: None,
        clock=FakeClock(),
    )
    
    error = RateLimitError(
        "rate limited",
        response=_stub_response(headers={"retry-after": "5"}),
        body={},
    )
    
    sleep_for, _ = client._next_delay(1.0, error)
    assert sleep_for == 5.0


def test_next_delay_with_http_date_header():
    """Test _next_delay handles HTTP date format in Retry-After."""
    from cli.openai_client import _extract_retry_after_seconds

    # Future date (5 seconds from now)
    from datetime import datetime, timezone, timedelta
    future = datetime.now(timezone.utc) + timedelta(seconds=5)
    date_str = future.strftime("%a, %d %b %Y %H:%M:%S GMT")

    error = RateLimitError(
        "rate limited",
        response=_stub_response(headers={"retry-after": date_str}),
        body={},
    )

    seconds = _extract_retry_after_seconds(error)
    assert seconds is not None
    assert 4 <= seconds <= 6  # Allow some timing variance


def test_next_delay_doubles_when_no_retry_after():
    """Test _next_delay doubles delay when no Retry-After header."""
    settings = OpenAISettings.load({}, actor="pytest")
    client = SharedOpenAIClient(
        settings,
        client=StubOpenAIClient(),
        metrics=create_metrics(),
        sleep_fn=lambda *_: None,
        clock=FakeClock(),
    )
    
    error = RateLimitError("rate limited", response=_stub_response(headers={}), body={})

    sleep_for, next_delay = client._next_delay(1.0, error)
    assert sleep_for == 1.0
    assert next_delay == 2.0


def test_openai_client_error_fields():
    """Test OpenAIClientError includes remediation and details."""
    error = OpenAIClientError(
        "Something failed",
        remediation="Try again later",
        details={"code": 429, "reason": "rate_limit"}
    )

    assert str(error) == "Something failed"
    assert error.remediation == "Try again later"
    assert error.details["code"] == 429
    assert error.details["reason"] == "rate_limit"


def test_openai_client_error_defaults_empty_details():
    """Test OpenAIClientError defaults to empty details."""
    error = OpenAIClientError("Error", remediation="Fix it")
    assert error.details == {}


def test_mask_base_url_in_openai_client():
    """Test mask_base_url helper used by openai_client module."""
    from cli.sanitizer import mask_base_url

    assert mask_base_url("https://api.openai.com/v1") == "https://***/v1"
    assert mask_base_url("http://localhost") == "http://***"


def test_chat_result_dataclass_frozen():
    """Test ChatResult is immutable."""
    from cli.openai_client import ChatResult

    result = ChatResult(
        model="gpt-4",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=20,
        finish_reason="stop",
        fallback_used=False,
        raw_response=None
    )

    with pytest.raises(AttributeError):
        result.model = "different"


def test_embedding_result_dataclass_frozen():
    """Test EmbeddingResult is immutable."""
    from cli.openai_client import EmbeddingResult

    result = EmbeddingResult(
        model="text-embedding",
        latency_ms=50.0,
        vector=[0.1, 0.2],
        tokens_consumed=5,
        raw_response=None
    )

    with pytest.raises(AttributeError):
        result.model = "different"


def test_extract_retry_after_invalid_date():
    """Test _extract_retry_after_seconds handles invalid date strings."""
    from cli.openai_client import _extract_retry_after_seconds

    error = RateLimitError(
        "error",
        response=_stub_response(headers={"retry-after": "not a valid date"}),
        body={},
    )

    result = _extract_retry_after_seconds(error)
    assert result is None


def test_extract_retry_after_missing_header():
    """Test _extract_retry_after_seconds returns None when header missing."""
    from cli.openai_client import _extract_retry_after_seconds

    error = RateLimitError("error", response=_stub_response(headers={}), body={})

    result = _extract_retry_after_seconds(error)
    assert result is None


def test_extract_retry_after_empty_value():
    """Test _extract_retry_after_seconds handles empty header value."""
    from cli.openai_client import _extract_retry_after_seconds

    error = RateLimitError("error", response=_stub_response(headers={"retry-after": "   "}), body={})

    result = _extract_retry_after_seconds(error)
    assert result is None


def test_extract_retry_after_negative_seconds():
    """Test _extract_retry_after_seconds clamps negative to zero."""
    from cli.openai_client import _extract_retry_after_seconds
    from datetime import datetime, timezone, timedelta

    past = datetime.now(timezone.utc) - timedelta(seconds=10)
    date_str = past.strftime("%a, %d %b %Y %H:%M:%S GMT")

    error = RateLimitError(
        "error",
        response=_stub_response(headers={"retry-after": date_str}),
        body={},
    )

    result = _extract_retry_after_seconds(error)
    assert result == 0.0
