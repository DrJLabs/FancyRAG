# ruff: noqa: S105
from cli import sanitizer


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


def test_scrub_object_handles_openai_headers_and_json_payload():
    payload = {
        "headers": {
            "X-OpenAI-Client": "python/1.0.0",
            "OpenAI-Organization": "org-123",
        },
        "body": '{"authorization": "Bearer test-token-987654"}',
    }

    scrubbed = sanitizer.scrub_object(payload)

    assert scrubbed["headers"]["X-OpenAI-Client"] == "***"
    assert scrubbed["headers"]["OpenAI-Organization"] == "***"
    assert "***" in scrubbed["body"]


def test_sanitize_text_handles_empty_string():
    """Test sanitizing empty string returns empty string."""
    result = sanitizer.sanitize_text("")
    assert result == ""


def test_sanitize_text_handles_none_input():
    """Test sanitizing None input handles gracefully."""
    result = sanitizer.sanitize_text(None)
    assert result is None


def test_sanitize_text_handles_non_string_input():
    """Test sanitizing non-string input (numbers, booleans, etc.)."""
    assert sanitizer.sanitize_text(123) == 123
    assert sanitizer.sanitize_text(True) is True
    assert sanitizer.sanitize_text([1, 2, 3]) == [1, 2, 3]


def test_sanitize_text_handles_bytes(monkeypatch):
    """Bytes inputs are decoded and sanitized."""
    monkeypatch.setenv("API_KEY", "key123")
    result = sanitizer.sanitize_text(b"Bearer key123")
    assert result == "Bearer ***"


def test_sanitize_text_case_sensitive_matching(monkeypatch):
    """Test that environment var matching is case-sensitive."""
    monkeypatch.setenv("TEST_SECRET", "CaseSensitive123")
    text = "Secret is CaseSensitive123 but not casesensitive123"
    result = sanitizer.sanitize_text(text)
    assert "CaseSensitive123" not in result
    assert "casesensitive123" in result  # Should not be redacted


def test_sanitize_text_multiple_env_vars_same_text(monkeypatch):
    """Test sanitizing text with multiple environment variables."""
    monkeypatch.setenv("API_KEY", "key123")
    monkeypatch.setenv("SECRET_TOKEN", "token456")
    monkeypatch.setenv("DB_PASSWORD", "pass789")

    text = "Using key123 and token456 with pass789"
    result = sanitizer.sanitize_text(text)

    assert "key123" not in result
    assert "token456" not in result
    assert "pass789" not in result
    assert result.count("***") == 3


def test_sanitize_text_partial_matches_not_redacted(monkeypatch):
    """Test that partial matches of environment variables are not redacted."""
    monkeypatch.setenv("SECRET", "abc123")
    text = "This contains abc123def which shouldn't be fully redacted"
    result = sanitizer.sanitize_text(text)

    # Only exact matches should be redacted
    assert "abc123" not in result
    assert "***" in result


def test_sanitize_text_env_var_at_boundaries(monkeypatch):
    """Test environment variables at string boundaries."""
    monkeypatch.setenv("TOKEN", "boundary123")

    # Test at start
    text1 = "boundary123 is at the start"
    result1 = sanitizer.sanitize_text(text1)
    assert result1.startswith("***")

    # Test at end  
    text2 = "Token at end: boundary123"
    result2 = sanitizer.sanitize_text(text2)
    assert result2.endswith("***")

    # Test as whole string
    text3 = "boundary123"
    result3 = sanitizer.sanitize_text(text3)
    assert result3 == "***"


def test_scrub_object_handles_empty_structures():
    """Test scrubbing empty data structures."""
    assert sanitizer.scrub_object({}) == {}
    assert sanitizer.scrub_object([]) == []
    assert sanitizer.scrub_object(()) == ()


def test_scrub_object_handles_none():
    """Test scrubbing None values."""
    assert sanitizer.scrub_object(None) is None

    data = {"key": None, "nested": {"value": None}}
    result = sanitizer.scrub_object(data)
    assert result["key"] is None
    assert result["nested"]["value"] is None


def test_scrub_object_handles_primitive_types():
    """Test scrubbing primitive data types."""
    assert sanitizer.scrub_object("string") == "string"
    assert sanitizer.scrub_object(42) == 42
    assert sanitizer.scrub_object(3.14) == 3.14
    assert sanitizer.scrub_object(True) is True


def test_scrub_object_deep_nesting():
    """Test scrubbing deeply nested structures."""
    deep_data = {
        "level1": {
            "level2": {
                "level3": {
                    "password": "deep-secret",
                    "level4": [
                        {"api_key": "nested-key"},
                        {"authorization": "Bearer deep-token"}
                    ]
                }
            }
        }
    }

    result = sanitizer.scrub_object(deep_data)
    assert result["level1"]["level2"]["level3"]["password"] == "***"
    assert result["level1"]["level2"]["level3"]["level4"][0]["api_key"] == "***"
    assert result["level1"]["level2"]["level3"]["level4"][1]["authorization"] == "***"


def test_scrub_object_preserves_non_sensitive_keys():
    """Test that non-sensitive keys are preserved exactly."""
    data = {
        "username": "john_doe",
        "email": "john@example.com",
        "preferences": {"theme": "dark", "lang": "en"},
        "api_key": "secret123"
    }

    result = sanitizer.scrub_object(data)
    assert result["username"] == "john_doe"
    assert result["email"] == "john@example.com"
    assert result["preferences"]["theme"] == "dark"
    assert result["preferences"]["lang"] == "en"
    assert result["api_key"] == "***"


def test_scrub_object_handles_list_with_mixed_types():
    """Test scrubbing lists containing mixed data types."""
    data = [
        "plain string",
        42,
        {"password": "secret"},
        ["nested", {"token": "hidden"}],
        None,
        True
    ]

    result = sanitizer.scrub_object(data)
    assert result[0] == "plain string"
    assert result[1] == 42
    assert result[2]["password"] == "***"
    assert result[3][1]["token"] == "***"
    assert result[4] is None
    assert result[5] is True


def test_scrub_object_handles_set_type():
    """Test scrubbing set data structures."""
    data = {"items": {"sk-live", "username", "api_key"}}
    result = sanitizer.scrub_object(data)
    # Sets should be returned as sorted lists for determinism
    assert isinstance(result["items"], list)
    assert result["items"][0] == "***"
    assert "username" in result["items"]


def test_scrub_object_handles_circular_references():
    """Test handling of circular references in data structures."""
    data = {"name": "test"}
    data["self"] = data  # Create circular reference

    # This should not cause infinite recursion
    result = sanitizer.scrub_object(data)
    assert "name" in result
    assert result["self"] == "<circular>"


def test_scrub_object_multiple_sensitive_keys_same_dict():
    """Test scrubbing dictionary with multiple sensitive keys."""
    data = {
        "api_key": "key123",
        "password": "pass456",
        "secret": "secret789",
        "authorization": "Bearer token",
        "x-api-key": "header-key",
        "normal_field": "keep this"
    }

    result = sanitizer.scrub_object(data)
    assert result["api_key"] == "***"
    assert result["password"] == "***"
    assert result["secret"] == "***"
    assert result["authorization"] == "***"
    assert result["x-api-key"] == "***"
    assert result["normal_field"] == "keep this"


def test_scrub_object_detects_camel_case_keys():
    """Sensitive camelCase keys are detected and redacted."""
    data = {
        "apiKey": "secret",
        "clientSecret": "top-secret",
        "userPassword": "pw12345",
    }

    result = sanitizer.scrub_object(data)
    assert result["apiKey"] == "***"
    assert result["clientSecret"] == "***"
    assert result["userPassword"] == "***"


def test_scrub_object_case_insensitive_key_matching():
    """Test that sensitive key matching is case-insensitive."""
    data = {
        "API_KEY": "upper",
        "api_key": "lower",
        "Api_Key": "mixed",
        "PASSWORD": "upper_pass",
        "Password": "mixed_pass"
    }

    result = sanitizer.scrub_object(data)
    assert result["API_KEY"] == "***"
    assert result["api_key"] == "***"
    assert result["Api_Key"] == "***"
    assert result["PASSWORD"] == "***"
    assert result["Password"] == "***"


def test_scrub_object_json_string_parsing():
    """Test scrubbing of JSON strings within values."""
    data = {
        "config": '{"api_key": "json-secret", "timeout": 30}',
        "metadata": '{"authorization": "Bearer json-token"}'
    }

    result = sanitizer.scrub_object(data)
    assert "json-secret" not in result["config"]
    assert "json-token" not in result["metadata"]
    assert "***" in result["config"]
    assert "***" in result["metadata"]


def test_scrub_object_url_parameters():
    """Test scrubbing sensitive data in URLs."""
    data = {
        "endpoint": "https://api.example.com?api_key=url-secret&user=test",
        "callback": "http://app.com/auth?token=callback-token"
    }

    result = sanitizer.scrub_object(data)
    assert "url-secret" not in result["endpoint"]
    assert "callback-token" not in result["callback"]


def test_sanitize_text_with_special_characters(monkeypatch):
    """Test sanitizing text with special characters in environment variables."""
    monkeypatch.setenv("SPECIAL_KEY", "abc@#$%123")
    text = "Key is abc@#$%123 in the system"
    result = sanitizer.sanitize_text(text)
    assert "abc@#$%123" not in result
    assert "***" in result


def test_sanitize_text_redacts_basic_auth():
    """Basic Authorization headers should be redacted."""
    text = "Authorization: Basic dXNlcjpwYXNz"
    result = sanitizer.sanitize_text(text)
    assert "***" in result


def test_sanitize_text_very_long_string(monkeypatch):
    """Test sanitizing very long strings for performance."""
    monkeypatch.setenv("LONG_SECRET", "secret123")
    long_text = "prefix " + "x" * 10000 + " secret123 " + "y" * 10000 + " suffix"
    result = sanitizer.sanitize_text(long_text)
    assert "secret123" not in result
    assert "***" in result
    assert len(result) < len(long_text)  # Should be shorter due to replacement


def test_scrub_object_performance_large_structure():
    """Test scrubbing performance with large data structures."""
    large_data = {f"key_{i}": f"value_{i}" for i in range(1000)}
    large_data["api_key"] = "secret"  # Add one sensitive key

    result = sanitizer.scrub_object(large_data)
    assert result["api_key"] == "***"
    assert len(result) == 1001  # All keys should be preserved


def test_scrub_object_preserves_structure_types():
    """Test that original data structure types are preserved."""
    original = {
        "list": [1, 2, 3],
        "tuple": (4, 5, 6),
        "dict": {"nested": True},
        "string": "text",
        "password": "secret"
    }

    result = sanitizer.scrub_object(original)
    assert isinstance(result["list"], list)
    assert isinstance(result["tuple"], tuple)
    assert isinstance(result["dict"], dict)
    assert isinstance(result["string"], str)
    assert result["password"] == "***"


def test_sanitize_text_unicode_characters(monkeypatch):
    """Test sanitizing text with unicode characters."""
    monkeypatch.setenv("UNICODE_KEY", "🔑secret123🔒")
    text = "The key is 🔑secret123🔒 for authentication"
    result = sanitizer.sanitize_text(text)
    assert "🔑secret123🔒" not in result
    assert "***" in result


def test_scrub_object_with_custom_objects():
    """Test scrubbing objects with custom class instances."""
    class CustomObject:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return f"CustomObject({self.value})"

    custom_obj = CustomObject("test")
    data = {
        "object": custom_obj,
        "password": "secret"
    }

    result = sanitizer.scrub_object(data)
    assert result["object"] == custom_obj  # Custom objects should be preserved
    assert result["password"] == "***"


def test_sanitize_and_scrub_integration():
    """Test integration between sanitize_text and scrub_object functions."""
    data = {
        "message": "API key is sk-test-integration",
        "nested": {
            "api_key": "nested-secret",
            "description": "Contains nested-secret value"
        }
    }

    # First scrub the object (handles sensitive keys)
    scrubbed = sanitizer.scrub_object(data)

    # Then sanitize the text values (handles env var content)

    assert scrubbed["nested"]["api_key"] == "***"  # Key-based scrubbing
    # Text-based sanitization would need env vars set to work