"""Prometheus-compatible telemetry primitives for OpenAI interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import structlog
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

logger = structlog.get_logger(__name__)


_LATENCY_BUCKETS_MS = (50, 100, 250, 500, 1000, 2000, 5000, 10000)


def _redact_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()
        if key_lower in {"api_key", "authorization", "bearer"}:
            safe[key] = "***"
        else:
            safe[key] = value
    return safe


@dataclass
class OpenAIMetrics:
    registry: CollectorRegistry
    chat_latency: Histogram
    chat_tokens: Counter
    embedding_latency: Histogram
    embedding_tokens: Counter

    def observe_chat(
        self,
        *,
        model: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        actor: Optional[str] = None,
    ) -> None:
        self.chat_latency.labels(model=model).observe(latency_ms)
        self.chat_tokens.labels(model=model, token_type="prompt").inc(prompt_tokens)
        self.chat_tokens.labels(model=model, token_type="completion").inc(completion_tokens)
        logger.info(
            "openai.telemetry.chat",
            **_redact_payload(
                {
                    "actor": actor or "unknown",
                    "model": model,
                    "latency_ms": latency_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            ),
        )

    def observe_embedding(
        self,
        *,
        model: str,
        latency_ms: float,
        vector_length: int,
        tokens_consumed: int,
        actor: Optional[str] = None,
    ) -> None:
        self.embedding_latency.labels(model=model).observe(latency_ms)
        self.embedding_tokens.labels(model=model, token_type="input").inc(tokens_consumed)
        logger.info(
            "openai.telemetry.embedding",
            **_redact_payload(
                {
                    "actor": actor or "unknown",
                    "model": model,
                    "latency_ms": latency_ms,
                    "vector_length": vector_length,
                    "tokens_consumed": tokens_consumed,
                }
            ),
        )

    def export(self) -> str:
        return generate_latest(self.registry).decode("utf-8")


def _build_metrics(registry: Optional[CollectorRegistry] = None) -> OpenAIMetrics:
    reg = registry or CollectorRegistry()
    chat_latency = Histogram(
        "graphrag_openai_chat_latency_ms",
        "Latency distribution for OpenAI chat completions (ms).",
        labelnames=("model",),
        buckets=_LATENCY_BUCKETS_MS,
        registry=reg,
    )
    chat_tokens = Counter(
        "graphrag_openai_chat_tokens_total",
        "Token usage for OpenAI chat completions.",
        labelnames=("model", "token_type"),
        registry=reg,
    )
    embedding_latency = Histogram(
        "graphrag_openai_embedding_latency_ms",
        "Latency distribution for OpenAI embedding requests (ms).",
        labelnames=("model",),
        buckets=_LATENCY_BUCKETS_MS,
        registry=reg,
    )
    embedding_tokens = Counter(
        "graphrag_openai_embedding_tokens_total",
        "Token usage for OpenAI embedding requests.",
        labelnames=("model", "token_type"),
        registry=reg,
    )
    return OpenAIMetrics(
        registry=reg,
        chat_latency=chat_latency,
        chat_tokens=chat_tokens,
        embedding_latency=embedding_latency,
        embedding_tokens=embedding_tokens,
    )


_DEFAULT_METRICS = _build_metrics()


def get_metrics() -> OpenAIMetrics:
    """Return the shared metrics aggregator for CLI consumption."""

    return _DEFAULT_METRICS


def create_metrics(registry: Optional[CollectorRegistry] = None) -> OpenAIMetrics:
    """Factory for isolated metrics registries, useful in tests."""

    return _build_metrics(registry)


__all__ = [
    "OpenAIMetrics",
    "get_metrics",
    "create_metrics",
]
