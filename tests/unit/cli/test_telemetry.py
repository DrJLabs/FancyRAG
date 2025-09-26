from cli.telemetry import create_metrics


def test_latency_histogram_uses_expected_buckets():
    metrics = create_metrics()
    histogram = metrics.chat_latency
    # prom-client stores buckets under ._buckets with sorted boundaries
    buckets = histogram._upper_bounds  # type: ignore[attr-defined]
    assert buckets[0] == 100
    assert 250 in buckets and 2000 in buckets
    assert buckets[-2] == 5000
