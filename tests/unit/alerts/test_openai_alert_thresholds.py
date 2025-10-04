from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

pytest.importorskip(
    "yaml",
    reason="PyYAML is required to validate the OpenAI telemetry playbook",
)

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAYBOOK_PATH = PROJECT_ROOT / "docs" / "alerts" / "openai-telemetry.yml"
BASELINE_MODEL = "gpt-4.1-mini"
FALLBACK_MODEL = "gpt-4o-mini"


def test_playbook_exists_and_has_expected_models():
    """
    Validate the OpenAI telemetry playbook exists and that its models and alerts conform to expected thresholds and metadata.
    
    Checks performed:
    - PLAYBOOK_PATH exists and loads as YAML with schema == 1.
    - metadata.last_reviewed is present and not older than 183 days.
    - Both BASELINE_MODEL and FALLBACK_MODEL are defined under "models" and include numeric fields:
      latency_ms_p95, token_prompt_per_minute, token_completion_per_minute, cost_usd_per_1k_prompt_tokens, cost_usd_per_1k_completion_tokens.
    - alerts.latency_p95 exists and its threshold_ms entries for the baseline and fallback models match each model's latency_ms_p95.
    - alerts.token_usage_spike exists and threshold_percent_over_baseline is a positive number.
    - latency_p95 alert includes a panel_uid for Grafana linkage.
    """
    assert PLAYBOOK_PATH.exists(), "OpenAI telemetry playbook missing; alerts cannot be automated."
    data = yaml.safe_load(PLAYBOOK_PATH.read_text(encoding="utf-8"))

    assert data.get("schema") == 1, "Playbook schema must be set to 1."

    metadata = data.get("metadata", {})
    last_reviewed = metadata.get("last_reviewed")
    assert last_reviewed, "metadata.last_reviewed is required"
    parsed = dt.date.fromisoformat(str(last_reviewed))
    age = dt.date.today() - parsed
    assert age.days <= 183, "Alert thresholds must be reviewed within the last 6 months."

    models = data.get("models", {})
    for name in (BASELINE_MODEL, FALLBACK_MODEL):
        assert name in models, f"Model {name} missing from alert thresholds."
        model_block = models[name]
        for field in (
            "latency_ms_p95",
            "token_prompt_per_minute",
            "token_completion_per_minute",
            "cost_usd_per_1k_prompt_tokens",
            "cost_usd_per_1k_completion_tokens",
        ):
            assert field in model_block, f"{field} missing for model {name}"
            assert isinstance(model_block[field], (int, float)), f"{field} for {name} must be numeric"

    alerts = data.get("alerts", {})
    latency_alert = alerts.get("latency_p95")
    assert latency_alert, "latency_p95 alert missing"
    thresholds = latency_alert.get("threshold_ms", {})
    assert thresholds.get(BASELINE_MODEL) == models[BASELINE_MODEL]["latency_ms_p95"], (
        "Baseline latency threshold must match documented p95 value."
    )
    assert thresholds.get(FALLBACK_MODEL) == models[FALLBACK_MODEL]["latency_ms_p95"], (
        "Fallback latency threshold must match documented p95 value."
    )

    token_alert = alerts.get("token_usage_spike")
    assert token_alert, "token_usage_spike alert missing"
    threshold_percent = token_alert.get("threshold_percent_over_baseline")
    assert isinstance(threshold_percent, (int, float)) and threshold_percent > 0

    panel_uid = latency_alert.get("panel_uid")
    assert panel_uid, "Grafana panel UID must be defined for latency alert"
