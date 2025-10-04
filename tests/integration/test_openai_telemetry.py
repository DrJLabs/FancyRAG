import pytest

pytest.importorskip(
    "prometheus_client",
    reason="prometheus_client is required to exercise telemetry metrics",
)

from prometheus_client import CollectorRegistry
from _compat.structlog import capture_logs

from cli import telemetry


def test_chat_metrics_record_and_export():
    registry = CollectorRegistry()
    metrics = telemetry.create_metrics(registry)
    with capture_logs() as logs:
        metrics.observe_chat(
            model="gpt-4.1-mini",
            latency_ms=120.5,
            prompt_tokens=512,
            completion_tokens=256,
            actor="pytest",
        )
    count_value = registry.get_sample_value(
        "graphrag_openai_chat_latency_ms_count",
        labels={"model": "gpt-4.1-mini"},
    )
    assert count_value == 1
    prompt_total = registry.get_sample_value(
        "graphrag_openai_chat_tokens_total",
        labels={"model": "gpt-4.1-mini", "token_type": "prompt"},
    )
    completion_total = registry.get_sample_value(
        "graphrag_openai_chat_tokens_total",
        labels={"model": "gpt-4.1-mini", "token_type": "completion"},
    )
    assert prompt_total == 512
    assert completion_total == 256
    assert any(entry["event"] == "openai.telemetry.chat" for entry in logs)
    exported = metrics.export()
    assert "graphrag_openai_chat_latency_ms" in exported


def test_embedding_metrics_and_redaction():
    registry = CollectorRegistry()
    metrics = telemetry.create_metrics(registry)
    with capture_logs() as logs:
        metrics.observe_embedding(
            model="text-embedding-3-small",
            latency_ms=80.0,
            vector_length=1536,
            tokens_consumed=128,
            actor="pytest",
        )
    latency_count = registry.get_sample_value(
        "graphrag_openai_embedding_latency_ms_count",
        labels={"model": "text-embedding-3-small"},
    )
    assert latency_count == 1
    tokens_total = registry.get_sample_value(
        "graphrag_openai_embedding_tokens_total",
        labels={"model": "text-embedding-3-small", "token_type": "input"},
    )
    assert tokens_total == 128
    secret_payload = telemetry._redact_payload(  # pylint: disable=protected-access
        {"api_key": "sk-test", "authorization": "Bearer abc", "bearer": "token"}
    )
    assert all(value == "***" for value in secret_payload.values())
    assert any(entry["event"] == "openai.telemetry.embedding" for entry in logs)
