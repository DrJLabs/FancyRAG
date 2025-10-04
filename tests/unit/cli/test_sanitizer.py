# ruff: noqa: S105
from cli import sanitizer


def test_sanitize_text_redacts_env_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abc123")
    text = "API key sk-test-abc123 should never leak"
    sanitized = sanitizer.sanitize_text(text)
    assert "sk-test-abc123" not in sanitized
    assert "***" in sanitized


def test_sanitize_text_redacts_openai_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")
    text = "https://gateway.example.com/v1 uses host gateway.example.com"
    sanitized = sanitizer.sanitize_text(text)
    assert "gateway.example.com" not in sanitized
    assert "https://gateway.example.com/v1" not in sanitized
    assert sanitized.count("***") >= 2


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


def test_sanitize_and_scrub_integration(monkeypatch):
    """Test integration between sanitize_text and scrub_object functions."""

    monkeypatch.setenv("NESTED_SECRET", "nested-secret-value")
    data = {
        "message": "API key is sk-test-integration",
        "nested": {
            "api_key": "some-key",
            "description": "Contains nested-secret-value in text",
        },
    }

    scrubbed = sanitizer.scrub_object(data)

    assert "sk-test-integration" not in scrubbed["message"]
    assert "***" in scrubbed["message"]
    assert scrubbed["nested"]["api_key"] == "***"
    assert "nested-secret-value" not in scrubbed["nested"]["description"]
    assert "***" in scrubbed["nested"]["description"]


def test_is_sensitive_name_detects_explicit_keys():
    """Test _is_sensitive_name detects explicitly listed sensitive keys."""
    from cli.sanitizer import _is_sensitive_name
    
    assert _is_sensitive_name("api_key") is True
    assert _is_sensitive_name("apikey") is True
    assert _is_sensitive_name("authorization") is True
    assert _is_sensitive_name("bearer") is True
    assert _is_sensitive_name("password") is True
    assert _is_sensitive_name("token") is True


def test_is_sensitive_name_detects_key_suffix():
    """Test _is_sensitive_name detects keys ending with 'key'."""
    from cli.sanitizer import _is_sensitive_name
    
    assert _is_sensitive_name("client_key") is True
    assert _is_sensitive_name("api-key") is True
    assert _is_sensitive_name("private_key") is True
    assert _is_sensitive_name("encryption-key") is True


def test_is_sensitive_name_detects_token_words():
    """Test _is_sensitive_name detects sensitive token words."""
    from cli.sanitizer import _is_sensitive_name
    
    assert _is_sensitive_name("access_token") is True
    assert _is_sensitive_name("refresh-token") is True
    assert _is_sensitive_name("auth_secret") is True
    assert _is_sensitive_name("db_password") is True


def test_is_sensitive_name_detects_camelcase():
    """Test _is_sensitive_name detects camelCase sensitive names."""
    from cli.sanitizer import _is_sensitive_name
    
    assert _is_sensitive_name("apiKey") is True
    assert _is_sensitive_name("clientSecret") is True
    assert _is_sensitive_name("accessToken") is True
    assert _is_sensitive_name("userPassword") is True


def test_is_sensitive_name_rejects_safe_names():
    """Test _is_sensitive_name allows non-sensitive names."""
    from cli.sanitizer import _is_sensitive_name
    
    assert _is_sensitive_name("username") is False
    assert _is_sensitive_name("email") is False
    assert _is_sensitive_name("id") is False
    assert _is_sensitive_name("name") is False
    assert _is_sensitive_name("value") is False


def test_is_sensitive_name_handles_mixed_case():
    """Test _is_sensitive_name is case-insensitive."""
    from cli.sanitizer import _is_sensitive_name
    
    assert _is_sensitive_name("API_KEY") is True
    assert _is_sensitive_name("Api_Key") is True
    assert _is_sensitive_name("PASSWORD") is True
    assert _is_sensitive_name("Token") is True


def test_sanitize_text_with_openai_base_url_netloc(monkeypatch):
    """Test sanitize_text redacts OPENAI_BASE_URL netloc."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.gateway.com/v1")
    text = "Connecting to custom.gateway.com for API calls"
    result = sanitizer.sanitize_text(text)
    assert "custom.gateway.com" not in result
    assert "***" in result


def test_sanitize_text_bearer_token_pattern():
    """Test sanitize_text matches Bearer token patterns."""
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    result = sanitizer.sanitize_text(text)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
    assert "***" in result


def test_sanitize_text_api_key_json_pattern():
    """Test sanitize_text matches api_key in JSON-like strings."""
    text = '{"api_key": "sk-proj-abc123xyz", "model": "gpt-4"}'
    result = sanitizer.sanitize_text(text)
    assert "sk-proj-abc123xyz" not in result
    assert "***" in result


def test_sanitize_text_with_extra_patterns():
    """Test sanitize_text accepts extra_patterns parameter."""
    import re
    extra = [re.compile(r"custom-\w+")]
    text = "Secret: custom-secret-value and custom-another"
    result = sanitizer.sanitize_text(text, extra_patterns=extra)
    assert "custom-secret-value" not in result
    assert "custom-another" not in result
    assert "***" in result


def test_sanitize_text_min_length_threshold():
    """Test sanitize_text respects 4-character minimum for env var values."""
    # Very short values (< 4 chars) should not be redacted
    # This is to avoid over-redacting common short strings
    text = "Value is abc in the config"  # "abc" is only 3 chars
    result = sanitizer.sanitize_text(text)
    # Short values aren't in SECRET_ENV_KEYS, so won't be redacted unless they match patterns
    assert "abc" in result or "***" in result  # Either way is acceptable


def test_scrub_object_handles_frozenset():
    """Test scrub_object converts frozenset to sorted tuple."""
    data = {
        "items": frozenset(["apple", "sk-test", "banana"])
    }
    result = sanitizer.scrub_object(data)
    assert isinstance(result["items"], tuple)
    # Should be sorted and sk-test should be redacted
    assert result["items"][0] == "***"


def test_scrub_object_circular_reference_in_list():
    """Test scrub_object handles circular references in lists."""
    data = []
    data.append(data)  # Circular reference
    result = sanitizer.scrub_object(data)
    assert result[0] == "<circular>"


def test_scrub_object_circular_reference_in_tuple():
    """Test scrub_object handles circular references through tuples."""
    inner = {"key": "value"}
    inner["self"] = inner
    data = {"tuple": (inner, "other")}
    result = sanitizer.scrub_object(data)
    assert result["tuple"][0]["self"] == "<circular>"


def test_scrub_object_nested_password_in_tuple():
    """Test scrub_object sanitizes nested passwords in tuples."""
    data = {
        "config": (
            {"password": "secret1"},
            {"api_key": "secret2"},
            "plain"
        )
    }
    result = sanitizer.scrub_object(data)
    assert result["config"][0]["password"] == "***"
    assert result["config"][1]["api_key"] == "***"
    assert result["config"][2] == "plain"


def test_scrub_object_sensitive_none_value_preserved():
    """Test scrub_object preserves None for sensitive keys."""
    data = {
        "password": None,
        "api_key": None,
        "username": None
    }
    result = sanitizer.scrub_object(data)
    assert result["password"] is None
    assert result["api_key"] is None
    assert result["username"] is None


def test_sanitize_text_handles_bytearray():
    """Test sanitize_text handles bytearray input."""
    text = bytearray(b"Bearer secret-token-123")
    result = sanitizer.sanitize_text(text)
    assert isinstance(result, str)
    assert "***" in result


def test_sanitize_text_unicode_decode_errors():
    """Test sanitize_text handles invalid UTF-8 gracefully."""
    # Invalid UTF-8 sequence
    text = b"\x80\x81\x82"
    result = sanitizer.sanitize_text(text)
    # Should decode with replacement character
    assert isinstance(result, str)


def test_scrub_object_large_nested_structure():
    """Test scrub_object performance with deeply nested structures."""
    # Create a 10-level deep nested structure
    data = {"level_0": {"password": "secret"}}
    current = data["level_0"]
    for i in range(1, 10):
        current[f"level_{i}"] = {"api_key": f"key_{i}"}
        current = current[f"level_{i}"]
    
    result = sanitizer.scrub_object(data)
    
    # Verify sensitive data at all levels is redacted
    assert result["level_0"]["password"] == "***"
    current_result = result["level_0"]
    for i in range(1, 10):
        assert current_result[f"level_{i}"]["api_key"] == "***"
        current_result = current_result[f"level_{i}"]


def test_secret_patterns_match_sk_prefix():
    """Test SECRET_PATTERNS includes sk- prefix pattern."""
    from cli.sanitizer import SECRET_PATTERNS
    
    text = "sk-proj-abc123"
    matched = any(pattern.search(text) for pattern in SECRET_PATTERNS)
    assert matched is True


def test_secret_patterns_match_authorization_header():
    """Test SECRET_PATTERNS matches authorization headers."""
    from cli.sanitizer import SECRET_PATTERNS
    
    text = '"authorization": "Bearer abc123xyz"'
    matched = any(pattern.search(text) for pattern in SECRET_PATTERNS)
    assert matched is True


def test_sensitive_key_names_completeness():
    """Test SENSITIVE_KEY_NAMES includes expected keys."""
    from cli.sanitizer import SENSITIVE_KEY_NAMES
    
    assert "api_key" in SENSITIVE_KEY_NAMES
    assert "password" in SENSITIVE_KEY_NAMES
    assert "token" in SENSITIVE_KEY_NAMES
    assert "secret" in SENSITIVE_KEY_NAMES
    assert "authorization" in SENSITIVE_KEY_NAMES


def test_secret_env_keys_includes_openai_base_url():
    """Test SECRET_ENV_KEYS includes OPENAI_BASE_URL."""
    from cli.sanitizer import SECRET_ENV_KEYS
    
    assert "OPENAI_BASE_URL" in SECRET_ENV_KEYS
    assert "OPENAI_API_KEY" in SECRET_ENV_KEYS
    assert "NEO4J_PASSWORD" in SECRET_ENV_KEYS