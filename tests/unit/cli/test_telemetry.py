from src.cli.telemetry import create_metrics


def test_latency_histogram_uses_expected_buckets():
    metrics = create_metrics()
    reg = metrics.registry
    # Inspect bucket boundaries via Prometheus samples (exclude +Inf)
    chat_metric = next(m for m in reg.collect() if m.name == "graphrag_openai_chat_latency_ms")
    bounds = sorted(
        {int(float(s.labels["le"])) for s in chat_metric.samples if s.name.endswith("_bucket") and s.labels["le"] != "+Inf"}
    )
    assert bounds[0] == 100
    assert 250 in bounds and 2000 in bounds
    assert bounds[-1] == 5000
