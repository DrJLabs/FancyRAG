"""Minimal OpenAI SDK stub used for offline integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

_DEFAULT_EMBEDDING_DIMENSIONS = 1024


class APIError(Exception):
    """Base error for stubbed OpenAI client."""


class APIConnectionError(APIError):
    """Raised when a network connection would fail."""


class APIStatusError(APIError):
    """Raised when the API would return a non-success status code."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response: Any = None,
        body: Any = None,
        headers: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response
        self.body = body
        self.headers = headers


class RateLimitError(APIStatusError):
    """Raised when the API would return a rate-limit response."""

    def __init__(
        self,
        message: str = "rate limited",
        *,
        response: Any = None,
        body: Any = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code or 429, response=response, body=body)


@dataclass
class _ChatUsage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class _EmbeddingUsage:
    total_tokens: int


@dataclass
class _ResponseUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class _ChatCompletions:
    """Stubbed chat completions interface."""

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: int,
        **_: Any,
    ) -> Any:
        last = messages[-1].get("content", "") if messages else ""
        content = f"Stubbed response for: {last}".strip()
        usage = _ChatUsage(prompt_tokens=max(len(messages) * 10, 5), completion_tokens=min(max_tokens, 12))
        return SimpleNamespace(
            id="stub-chat-completion",
            model=model,
            usage=usage,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason="stop",
                )
            ],
        )


class _Chat:
    def __init__(self) -> None:
        self.completions = _ChatCompletions()


class _Responses:
    """Stubbed responses interface."""

    def create(
        self,
        *,
        model: str,
        input: Any,
        temperature: float,
        max_output_tokens: int,
        **_: Any,
    ) -> Any:
        if isinstance(input, Sequence) and not isinstance(input, str):
            last = input[-1]
            if isinstance(last, Mapping):
                last_content = last.get("content", "")
            else:
                last_content = getattr(last, "content", "")
        else:
            last_content = str(input)
        content = f"Stubbed response for: {last_content}".strip()
        usage = _ResponseUsage(
            input_tokens=max(len(str(input).split()), 1),
            output_tokens=min(max_output_tokens, 12),
        )
        return SimpleNamespace(
            id="stub-response",
            model=model,
            usage=usage,
            output_text=content,
        )


class _Embeddings:
    """Stubbed embeddings interface."""

    def create(self, *, model: str, input: str, **_: Any) -> Any:
        vector = [float((i % 17) - 8) for i in range(_DEFAULT_EMBEDDING_DIMENSIONS)]
        usage = _EmbeddingUsage(total_tokens=max(len(input.split()), 1))
        return SimpleNamespace(
            model=model,
            data=[SimpleNamespace(embedding=vector)],
            usage=usage,
        )


class OpenAI:
    """Minimal stub compatible with the real OpenAI SDK surface used in the project."""

    def __init__(self, *, base_url: str | None = None, **_: Any) -> None:
        self.base_url = base_url
        self.chat = _Chat()
        self.responses = _Responses()
        self.embeddings = _Embeddings()


__all__ = [
    "OpenAI",
    "APIError",
    "APIConnectionError",
    "APIStatusError",
    "RateLimitError",
]
