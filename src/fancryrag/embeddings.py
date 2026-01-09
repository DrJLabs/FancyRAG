"""Embedding utilities with retry and latency instrumentation."""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import OpenAIError
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

_EMBEDDING_FAILURE_MSG = "Embedding request failed without raising an exception"


logger = logging.getLogger(__name__)


class RetryingOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAI embeddings client with retry and latency logging."""

    def __init__(
        self,
        *,
        model: str,
        max_retries: int = 3,
        timeout_seconds: float = 10.0,
        backoff_seconds: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._max_retries = max_retries
        self._timeout = timeout_seconds
        self._backoff = backoff_seconds

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:  # type: ignore[override]
        last_error: Exception | None = None
        attempt = 0
        start_overall = time.perf_counter()

        while attempt < self._max_retries:
            attempt += 1
            start = time.perf_counter()
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model,
                    timeout=self._timeout,
                    **kwargs,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    "embedding.success",
                    extra={
                        "attempt": attempt,
                        "latency_ms": round(latency_ms, 2),
                    },
                )
                if not getattr(response, "data", None):
                    raise OpenAIError("OpenAI API returned empty data list")
                embedding: list[float] = response.data[0].embedding
                return embedding
            except OpenAIError as error:  # pragma: no cover - depends on network failures
                last_error = error
                latency_ms = (time.perf_counter() - start) * 1000
                logger.warning(
                    "embedding.retry",
                    extra={
                        "attempt": attempt,
                        "latency_ms": round(latency_ms, 2),
                        "error": type(error).__name__,
                    },
                )
                sleep_for = self._backoff * (2 ** (attempt - 1))
                time.sleep(min(sleep_for, 5.0))

        total_ms = (time.perf_counter() - start_overall) * 1000
        logger.error(
            "embedding.failed",
            extra={
                "attempts": self._max_retries,
                "latency_ms": round(total_ms, 2),
                "error": type(last_error).__name__ if last_error else "unknown",
            },
        )
        if last_error:
            raise last_error
        raise RuntimeError(_EMBEDDING_FAILURE_MSG)


__all__ = ["RetryingOpenAIEmbeddings"]
