import pytest
from types import SimpleNamespace

from cli.openai_client import OpenAIClientError, RateLimitError, SharedOpenAIClient
from cli.telemetry import create_metrics
from config.settings import OpenAISettings


class FakeClock:
    def __init__(self, step: float = 0.005) -> None:
        self._value = 0.0
        self._step = step

    def __call__(self) -> float:
        current = self._value
        self._value += self._step
        return current


class StubChatResponse(SimpleNamespace):
    pass


class StubEmbeddingResponse(SimpleNamespace):
    pass


class StubOpenAIClient:
    def __init__(self) -> None:
        self.chat_calls: list[dict] = []
        self.embedding_calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embedding)

    def _chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return StubChatResponse(
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
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

    def _chat(self, **kwargs):
        """
        Simulate a chat request that may raise a rate-limit error for a configured number of initial calls.
        
        When the client is configured to fail, this method raises a RateLimitError containing a 429 status and Retry-After headers; otherwise it delegates to the underlying client's chat implementation and returns its response.
        
        Returns:
            The chat response object produced by the underlying client.
        
        Raises:
            RateLimitError: If the client is still in its configured failure period; the error includes a 429 status and Retry-After headers.
        """
        if self._failures > 0:
            self._failures -= 1
            headers = {"Retry-After": "1", "retry-after": "1"}
            response = SimpleNamespace(headers=headers, request=SimpleNamespace(), status_code=429)
            raise RateLimitError("slow down", response=response, body=None)
        return super()._chat(**kwargs)


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

    def _chat(self, **kwargs):
        """
        Simulates a chat call that enforces rate-limit behavior using a sequence of `Retry-After` header values.
        
        When the configured retry sequence or the `always_fail` flag indicates a failure for the current attempt, raises a RateLimitError whose `response` includes `Retry-After` and `retry-after` headers and has HTTP status 429; otherwise delegates to and returns the superclass `_chat` result.
        
        Returns:
            The chat response returned by the superclass `_chat` when no rate-limit is applied.
        
        Raises:
            RateLimitError: if this call is simulated as rate-limited (response.headers contains `Retry-After` and status_code is 429).
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
            raise RateLimitError("slow down", response=response, body=None)
        return super()._chat(**kwargs)


def _make_client(env: dict[str, str], *, client) -> SharedOpenAIClient:
    settings = OpenAISettings.load(env, actor="pytest")
    return SharedOpenAIClient(
        settings,
        client=client,
        metrics=create_metrics(),
        sleep_fn=lambda *_: None,
        clock=FakeClock(),
    )


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
