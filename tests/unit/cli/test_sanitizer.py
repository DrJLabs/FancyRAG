from src.cli import sanitizer


def test_sanitize_text_redacts_env_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abc123")
    text = "API key sk-test-abc123 should never leak"
    sanitized = sanitizer.sanitize_text(text)
    assert "sk-test-abc123" not in sanitized
    assert "***" in sanitized


def test_scrub_object_scrubs_sensitive_keys(monkeypatch):
    monkeypatch.setenv("NEO4J_PASSWORD", "secret123")
    payload = {
        "authorization": "Bearer some-token",
        "nested": {"notes": "Use secret123 for auth"},
        "list": ["sk-test-value", {"api_key": "abc"}],
    }
    sanitized = sanitizer.scrub_object(payload)
    assert sanitized["authorization"] == "***"
    assert "***" in sanitized["nested"]["notes"]
    assert "***" in sanitized["list"][0]
    assert sanitized["list"][1]["api_key"] == "***"


def test_scrub_object_handles_mixed_structures(monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret")
    data = {
        "message": "Token qdrant-secret should hide",
        "tuple": ("sk-live", {"password": "hunter2"}),
    }
    scrubbed = sanitizer.scrub_object(data)
    assert "qdrant-secret" not in scrubbed["message"]
    assert scrubbed["tuple"][0] == "***"
    assert scrubbed["tuple"][1]["password"] == "***"


def test_scrub_object_sanitizes_tuples_in_lists():
    payload = {"items": [("password", "sk-secret"), ("note", "contains sk-secret")]}
    scrubbed = sanitizer.scrub_object(payload)
    assert scrubbed["items"][0][1] == "***"
    assert "***" in scrubbed["items"][1][1]
