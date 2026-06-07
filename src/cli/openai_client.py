"""Shared OpenAI client abstraction with guardrails and telemetry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from _compat.structlog import get_logger

from config.settings import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_RETRY_ATTEMPTS,
    FALLBACK_CHAT_MODELS,
    OpenAISettings,
)
from cli.telemetry import OpenAIMetrics, get_metrics
from cli.utils import ensure_embedding_dimensions
from cli.sanitizer import mask_base_url

logger = get_logger(__name__)
try:  # pragma: no cover - exercised in integration tests
    from openai import APIConnectionError, APIError, APIStatusError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - handled in tests without openai installed
    class APIError(Exception):
        """Fallback APIError used when the real openai package is unavailable."""

    class APIConnectionError(APIError):
        """Fallback connection error."""

    class APIStatusError(APIError):
        """Fallback status error."""

        def __init__(self, message: str, *, status_code: Optional[int] = None, response: Any = None) -> None:
            super().__init__(message)
            self.status_code = status_code
            self.response = response

    class RateLimitError(APIStatusError):
        """Fallback rate limit error."""

        def __init__(self, message: str, *, response: Any = None, body: Any = None) -> None:
            super().__init__(message, status_code=429, response=response)
            self.body = body

    class OpenAI:  # type: ignore[no-redef]
        """Fallback OpenAI client raising ImportError on use."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError(
                "The openai package is required for OpenAI operations; install it with the project extras."
            )


_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRY_AFTER_HEADER = "retry-after"


@dataclass(frozen=True)
class ChatResult:
    """Result payload for a chat completion request."""

    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Optional[str]
    fallback_used: bool
    raw_response: Any


@dataclass(frozen=True)
class EmbeddingResult:
    """Result payload for an embedding request."""

    model: str
    latency_ms: float
    vector: Sequence[float]
    tokens_consumed: int
    raw_response: Any


class OpenAIClientError(RuntimeError):
    """Error raised when the shared OpenAI client cannot complete a request."""

    def __init__(self, message: str, *, remediation: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.remediation = remediation
        self.details = details or {}


class SharedOpenAIClient:
    """Shared OpenAI client abstraction wiring guardrails, retries, and telemetry."""

    def __init__(
        self,
        settings: OpenAISettings,
        *,
        client: Optional[OpenAI] = None,
        embedding_client: Optional[OpenAI] = None,
        metrics: Optional[OpenAIMetrics] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._settings = settings
        client_kwargs: Dict[str, Any] = {}
        if settings.api_base_url:
            client_kwargs["base_url"] = settings.api_base_url
            logger.info(
                "openai.client.base_url_override",
                actor=settings.actor,
                base_url=mask_base_url(settings.api_base_url),
            )
        if settings.api_key is not None:
            client_kwargs["api_key"] = settings.api_key.get_secret_value()
        self._chat_client = client or OpenAI(**client_kwargs)
        self._embedding_client = embedding_client
        if self._embedding_client is None and settings.embedding_api_base_url:
            embedding_kwargs: Dict[str, Any] = {
                "base_url": settings.embedding_api_base_url,
            }
            if settings.embedding_api_key is not None:
                embedding_kwargs["api_key"] = settings.embedding_api_key.get_secret_value()
            self._embedding_client = OpenAI(**embedding_kwargs)
            logger.info(
                "openai.client.embedding_base_url_override",
                actor=settings.actor,
                base_url=mask_base_url(settings.embedding_api_base_url),
            )
        self._metrics = metrics or get_metrics()
        self._sleep_fn = sleep_fn
        self._clock = clock
        # Select a deterministic fallback model to ensure stable telemetry and behavior
        self._fallback_model = (
            sorted(FALLBACK_CHAT_MODELS)[0]
            if FALLBACK_CHAT_MODELS
            else DEFAULT_CHAT_MODEL
        )

    # ------------------------------------------------------------------
    # Public API

    def chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        **extra_params: Any,
    ) -> ChatResult:
        """Execute a responses API request with retry and fallback guardrails.

        Defaults `reasoning` to minimal effort when not explicitly provided.
        """

        model = self._settings.chat_model
        fallback_used = model in FALLBACK_CHAT_MODELS

        def _run(model_name: str) -> tuple[Any, float]:
            params: Dict[str, Any] = {
                "model": model_name,
                "input": list(messages),
                "max_output_tokens": max_tokens,
            }
            params.update(extra_params)
            if temperature is not None and not model_name.startswith("gpt-5"):
                params["temperature"] = temperature
            if "reasoning" not in params:
                params["reasoning"] = {"effort": "minimal"}
            return self._execute_with_backoff(
                lambda: self._chat_client.responses.create(**params),
                description=f"responses create ({model_name})",
            )

        try:
            response, latency_ms = _run(model)
        except OpenAIClientError as exc:
            if (
                self._settings.enable_fallback
                and model == DEFAULT_CHAT_MODEL
                and self._fallback_model != model
                and isinstance(exc.details, Mapping)
                and exc.details.get("reason") == "rate_limit"
            ):
                logger.warning(
                    "openai.retry.fallback",
                    actor=self._settings.actor,
                    from_model=model,
                    to_model=self._fallback_model,
                )
                model = self._fallback_model
                fallback_used = True
                response, latency_ms = _run(model)
            else:
                raise
        except (APIConnectionError, APIError) as exc:
            raise self._wrap_error(
                exc,
                remediation=(
                    "Verify OpenAI connectivity, model availability, and retry after reducing concurrent requests."
                ),
                details={"reason": "chat", "model": model},
            ) from exc

        usage = getattr(response, "usage", None)
        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")
        finish_reason = _first_choice_value(response, "finish_reason")

        self._metrics.observe_chat(
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            actor=self._settings.actor,
        )

        logger.info(
            "openai.chat.success",
            actor=self._settings.actor,
            model=model,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            fallback_used=fallback_used,
        )

        return ChatResult(
            model=model,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            fallback_used=fallback_used,
            raw_response=response,
        )

    def embedding(
        self,
        *,
        input_text: str,
        override_dimensions: Optional[int] = None,
        **extra_params: Any,
    ) -> EmbeddingResult:
        """Execute an embedding request enforcing dimension guardrails."""

        try:
            if self._embedding_client is None:
                raise OpenAIClientError(
                    "Embedding client is not configured.",
                    remediation="Set EMBEDDING_API_BASE_URL (and EMBEDDING_API_KEY if required) for local embeddings.",
                    details={"reason": "embedding_base_url_missing"},
                )
            response, latency_ms = self._execute_with_backoff(
                lambda: self._embedding_client.embeddings.create(
                    model=self._settings.embedding_model,
                    input=input_text,
                    **extra_params,
                ),
                description=f"embedding request ({self._settings.embedding_model})",
            )
        except (APIConnectionError, APIError) as exc:
            raise self._wrap_error(
                exc,
                remediation="Verify OpenAI embeddings availability and retry after checking service status.",
                details={"reason": "embedding", "model": self._settings.embedding_model},
            ) from exc

        data = getattr(response, "data", None) or []
        if not data:
            raise OpenAIClientError(
                "Embedding response did not include any vector payload.",
                remediation="Inspect OpenAI response structure or retry after confirming service health.",
                details={"reason": "no_embedding"},
            )

        vector = _extract_embedding_vector(data[0])
        try:
            ensure_embedding_dimensions(
                vector,
                settings=self._settings,
                override_dimensions=override_dimensions,
            )
        except ValueError as exc:
            raise OpenAIClientError(
                str(exc),
                remediation="Align OPENAI_EMBEDDING_DIMENSIONS with the provider response or adjust overrides.",
                details={"reason": "dimension_mismatch"},
            ) from exc

        tokens_consumed = _usage_value(getattr(response, "usage", None), "total_tokens")
        self._metrics.observe_embedding(
            model=self._settings.embedding_model,
            latency_ms=latency_ms,
            vector_length=len(vector),
            tokens_consumed=tokens_consumed,
            actor=self._settings.actor,
        )

        logger.info(
            "openai.embedding.success",
            actor=self._settings.actor,
            model=self._settings.embedding_model,
            latency_ms=round(latency_ms, 2),
            vector_length=len(vector),
            tokens_consumed=tokens_consumed,
        )

        return EmbeddingResult(
            model=self._settings.embedding_model,
            latency_ms=round(latency_ms, 2),
            vector=vector,
            tokens_consumed=tokens_consumed,
            raw_response=response,
        )

    # ------------------------------------------------------------------
    # Internal helpers

    def _execute_with_backoff(
        self,
        operation: Callable[[], Any],
        *,
        description: str,
    ) -> tuple[Any, float]:
        attempts = 0
        delay = self._settings.backoff_seconds or DEFAULT_BACKOFF_SECONDS
        max_attempts = self._settings.max_attempts or DEFAULT_MAX_RETRY_ATTEMPTS
        errors: list[str] = []

        while attempts < max_attempts:
            attempts += 1
            start = self._clock()
            try:
                result = operation()
                latency_ms = (self._clock() - start) * 1000.0
                return result, latency_ms
            except (
                RateLimitError,
                APIStatusError,
                APIConnectionError,
                APIError,
            ) as exc:
                if self._is_retryable(exc):
                    errors.append(repr(exc))
                    if attempts >= max_attempts:
                        raise OpenAIClientError(
                            f"Rate limit or transient failure for {description} after {max_attempts} attempts.",
                            remediation="Reduce concurrent usage or retry later with lower load.",
                            details={"errors": errors, "reason": "rate_limit"},
                        ) from exc
                    sleep_duration, delay = self._next_delay(delay, exc)
                    logger.warning(
                        "openai.retry.backoff",
                        actor=self._settings.actor,
                        description=description,
                        attempt=attempts,
                        sleep_seconds=round(sleep_duration, 3),
                    )
                    self._sleep_fn(sleep_duration)
                    continue
                raise

        raise OpenAIClientError(
            f"Exceeded retry attempts for {description}.",
            remediation="Investigate OpenAI availability and retry with adjusted settings.",
            details={"reason": "retry_exhausted", "errors": errors},
        )

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, APIStatusError) and getattr(error, "status_code", None) in _RETRYABLE_STATUS_CODES:
            return True
        return False

    def _next_delay(self, current_delay: float, error: Exception) -> tuple[float, float]:
        retry_after_seconds = _extract_retry_after_seconds(error)
        if retry_after_seconds is not None:
            sleep_for = max(retry_after_seconds, current_delay)
            next_delay = max(sleep_for, current_delay * 2)
            return sleep_for, next_delay
        return current_delay, current_delay * 2

    @staticmethod
    def _wrap_error(
        error: Exception,
        *,
        remediation: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> OpenAIClientError:
        return OpenAIClientError(str(error), remediation=remediation, details=details)


def _usage_value(usage: Any, attr: str) -> int:
    if usage is None:
        return 0
    fallback = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens"}.get(attr)
    value = _lookup_usage_attr(usage, attr, fallback=fallback)
    return int(value or 0)


def _lookup_usage_attr(usage: Any, attr: str, *, fallback: Optional[str] = None) -> Any:
    if isinstance(usage, Mapping):
        value = usage.get(attr)
        if value is None and fallback:
            return usage.get(fallback)
        return value
    value = getattr(usage, attr, None)
    if value is None and fallback:
        return getattr(usage, fallback, None)
    return value


def _first_choice_value(response: Any, field: str) -> Optional[str]:
    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, Mapping):
        choices = response.get("choices")
    if not choices:
        return None
    choice = choices[0]
    if isinstance(choice, Mapping):
        return choice.get(field)
    return getattr(choice, field, None)


def _extract_embedding_vector(payload: Any) -> Sequence[float]:
    if isinstance(payload, Mapping):
        vector = payload.get("embedding")
    else:
        vector = getattr(payload, "embedding", None)
    if vector is None:
        raise OpenAIClientError(
            "Embedding vector missing from response payload.",
            remediation="Upgrade the openai SDK or inspect API responses for schema changes.",
            details={"reason": "missing_embedding"},
        )
    return vector


def _extract_retry_after_seconds(error: Exception) -> Optional[float]:
    def _calculate_delta(parsed: datetime) -> float:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta = (parsed - datetime.now(timezone.utc)).total_seconds()
        return max(delta, 0.0)

    response = getattr(error, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    header_value = headers.get(_RETRY_AFTER_HEADER)
    if not header_value:
        return None
    header_value = str(header_value).strip()
    if not header_value:
        return None
    if header_value.isdigit():
        return float(header_value)
    try:
        parsed = parsedate_to_datetime(header_value)
    except (TypeError, ValueError):
        parsed = None
    if parsed is not None:
        return _calculate_delta(parsed)
    candidates = [value.strip() for value in header_value.split(",") if value.strip()]
    for value in candidates:
        if value.isdigit():
            return float(value)
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            continue
        return _calculate_delta(parsed)
    return None


__all__ = [
    "ChatResult",
    "EmbeddingResult",
    "OpenAIClientError",
    "RateLimitError",
    "SharedOpenAIClient",
]
