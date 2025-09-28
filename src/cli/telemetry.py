"""Prometheus-compatible telemetry primitives for OpenAI interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, cast

from _compat.structlog import get_logger
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from cli.sanitizer import scrub_object

logger = get_logger(__name__)


_LATENCY_BUCKETS_MS = (100, 250, 500, 1000, 2000, 5000)


def _redact_payload(data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of the input mapping with sensitive values redacted.
    
    Parameters:
        data (Mapping[str, Any]): Mapping to sanitize; may contain nested mappings and values of arbitrary types.
    
    Returns:
        Dict[str, Any]: A redacted copy of `data` with sensitive fields masked while preserving the original structure and value types where possible.
    """
    return cast(Dict[str, Any], scrub_object(data))


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
        """
        Record telemetry for a chat completion and emit a redacted structured log entry.
        
        Records the observed latency (milliseconds) and increments prompt/completion token counters for the given model, then logs a redacted payload containing actor, model, latency_ms, prompt_tokens, and completion_tokens.
        
        Parameters:
            actor (Optional[str]): Optional identifier for the actor to include in the log; when omitted, "unknown" is used.
        """
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
        """
        Record embedding request telemetry and emit a redacted structured log.
        
        Records the embedding request latency (milliseconds) for the given model and
        increments the input token counter by `tokens_consumed`. Emits a structured
        log entry "openai.telemetry.embedding" containing a redacted payload with
        the actor, model, latency_ms, vector_length, and tokens_consumed.
        
        Parameters:
            model (str): Model identifier used for the embedding request.
            latency_ms (float): Observed latency in milliseconds.
            vector_length (int): Length of the returned embedding vector.
            tokens_consumed (int): Number of input tokens consumed by the request.
            actor (Optional[str]): Optional identifier for the actor that initiated
                the request; defaults to "unknown" when not provided.
        """
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
        """
        Return the current metrics from the registry in Prometheus text exposition format.
        
        Returns:
            metrics_text (str): Metrics data serialized in Prometheus text format.
        """
        return generate_latest(self.registry).decode("utf-8")


def _build_metrics(registry: Optional[CollectorRegistry] = None) -> OpenAIMetrics:
    """
    Create an OpenAIMetrics container with Prometheus metrics registered to a CollectorRegistry.
    
    Parameters:
        registry (Optional[CollectorRegistry]): CollectorRegistry to register metrics into. If omitted, a new registry is created.
    
    Returns:
        OpenAIMetrics: Instance containing chat and embedding histograms for latency and counters for token usage, all registered to the provided registry.
    """
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
    """
    Get the shared OpenAIMetrics instance used by CLI tools.
    
    Returns:
        OpenAIMetrics: The shared metrics aggregator instance.
    """

    return _DEFAULT_METRICS


def create_metrics(registry: Optional[CollectorRegistry] = None) -> OpenAIMetrics:
    """Factory for isolated metrics registries, useful in tests."""

    return _build_metrics(registry)


__all__ = [
    "OpenAIMetrics",
    "get_metrics",
    "create_metrics",
]
