import pytest
from dataclasses import FrozenInstanceError

from _compat.structlog import capture_logs

from pydantic import ValidationError

from config.settings import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_RETRY_ATTEMPTS,
    FALLBACK_CHAT_MODELS,
    OpenAISettings,
)


@pytest.mark.parametrize(
    "env,expected_model,is_override",
    [
        ({}, DEFAULT_CHAT_MODEL, False),
        ({"OPENAI_MODEL": "gpt-4.1-mini"}, "gpt-4.1-mini", False),
        ({"OPENAI_MODEL": "gpt-4o-mini"}, "gpt-4o-mini", True),
    ],
)
def test_chat_model_allowlist(env, expected_model, is_override):
    with capture_logs() as logs:
        settings = OpenAISettings.load(env, actor="pytest")
    assert settings.chat_model == expected_model
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.embedding_dimensions == DEFAULT_EMBEDDING_DIMENSIONS
    assert settings.embedding_dimensions_override is None
    assert settings.actor == "pytest"
    if is_override:
        assert any(entry["event"] == "openai.chat.override" for entry in logs)
    else:
        assert all(entry["event"] != "openai.chat.override" for entry in logs)


def test_invalid_chat_model_logs_and_raises():
    env = {"OPENAI_MODEL": "gpt-nonsense"}
    with capture_logs() as logs:
        with pytest.raises(ValueError) as exc:
            OpenAISettings.load(env, actor="pytest")
    assert "Unsupported OpenAI chat model" in str(exc.value)
    assert any(entry["event"] == "openai.chat.invalid_model" for entry in logs)


@pytest.mark.parametrize(
    "override_value,expect_success",
    [
        ("2048", True),
        (" 3072 ", True),
        ("-1", False),
        ("abc", False),
    ],
)
def test_embedding_override_validation(override_value, expect_success):
    env = {"OPENAI_EMBEDDING_DIMENSIONS": override_value}
    with capture_logs() as logs:
        if expect_success:
            settings = OpenAISettings.load(env, actor="pytest")
            assert settings.embedding_dimensions_override == int(override_value)
            assert any(entry["event"] == "openai.embedding.override" for entry in logs)
        else:
            with pytest.raises(ValueError):
                OpenAISettings.load(env, actor="pytest")
            assert any(
                entry["event"] in {"openai.embedding.invalid_override", "openai.embedding.non_positive_override"}
                for entry in logs
            )


def test_actor_fallback_uses_getpass(monkeypatch):
    monkeypatch.delenv("GRAPH_RAG_ACTOR", raising=False)
    monkeypatch.setattr("config.settings.getpass.getuser", lambda: "cli-user")
    settings = OpenAISettings.load({}, actor=None)
    assert settings.actor == "cli-user"


def test_allowed_chat_models_exposed():
    settings = OpenAISettings.load({}, actor="pytest")
    assert settings.allowed_chat_models == FALLBACK_CHAT_MODELS | {DEFAULT_CHAT_MODEL}


def test_max_attempts_override_logs_and_applies():
    env = {"OPENAI_MAX_ATTEMPTS": "5"}
    with capture_logs() as logs:
        settings = OpenAISettings.load(env, actor="pytest")
    assert settings.max_attempts == 5
    assert any(entry["event"] == "openai.settings.max_attempts_override" for entry in logs)


def test_backoff_override_validation():
    env = {"OPENAI_BACKOFF_SECONDS": "0.75"}
    with capture_logs() as logs:
        settings = OpenAISettings.load(env, actor="pytest")
    assert settings.backoff_seconds == pytest.approx(0.75)
    assert any(entry["event"] == "openai.settings.backoff_override" for entry in logs)


def test_invalid_backoff_raises():
    env = {"OPENAI_BACKOFF_SECONDS": "zero"}
    with pytest.raises(ValueError):
        OpenAISettings.load(env, actor="pytest")


def test_disable_fallback_blocks_override():
    env = {"OPENAI_MODEL": "gpt-4o-mini", "OPENAI_ENABLE_FALLBACK": "false"}
    with pytest.raises(ValueError):
        OpenAISettings.load(env, actor="pytest")


def test_disable_fallback_keeps_default():
    env = {"OPENAI_ENABLE_FALLBACK": "false"}
    settings = OpenAISettings.load(env, actor="pytest")
    assert settings.enable_fallback is False
    assert settings.chat_model == DEFAULT_CHAT_MODEL
    assert settings.max_attempts == DEFAULT_MAX_RETRY_ATTEMPTS
    assert settings.backoff_seconds == DEFAULT_BACKOFF_SECONDS


def test_base_url_override_logs_and_applies():
    env = {"OPENAI_BASE_URL": "https://gateway.example.com/v1"}
    with capture_logs() as logs:
        settings = OpenAISettings.load(env, actor="pytest")
    assert settings.api_base_url == "https://gateway.example.com/v1"
    assert any(
        entry["event"] == "openai.settings.base_url_override"
        and entry.get("base_url") == "https://***/v1"
        for entry in logs
    )


@pytest.mark.parametrize("value", ["ftp://example.com", "https://"])
def test_invalid_base_url_raises(value):
    env = {"OPENAI_BASE_URL": value}
    with capture_logs() as logs:
        with pytest.raises(ValueError):
            OpenAISettings.load(env, actor="pytest")
    assert any(entry["event"] == "openai.settings.invalid_base_url" for entry in logs)


def test_http_base_url_requires_explicit_opt_in():
    env = {"OPENAI_BASE_URL": "http://gateway.example.com/v1"}
    with capture_logs() as logs:
        with pytest.raises(ValueError):
            OpenAISettings.load(env, actor="pytest")
    assert any(entry["event"] == "openai.settings.insecure_base_url" for entry in logs)


def test_http_base_url_allowed_with_flag():
    env = {
        "OPENAI_BASE_URL": "http://gateway.example.com/v1",
        "OPENAI_ALLOW_INSECURE_BASE_URL": "true",
    }
    with capture_logs() as logs:
        settings = OpenAISettings.load(env, actor="pytest")
    assert settings.api_base_url == "http://gateway.example.com/v1"
    assert settings.allow_insecure_base_url is True
    assert any(entry["event"] == "openai.settings.insecure_base_url_override" for entry in logs)


def test_embedding_base_url_override_logs_and_applies():
    env = {
        "EMBEDDING_API_BASE_URL": "http://localhost:20010/v1",
        "OPENAI_ALLOW_INSECURE_BASE_URL": "true",
    }
    with capture_logs() as logs:
        settings = OpenAISettings.load(env, actor="pytest")
    assert settings.embedding_api_base_url == "http://localhost:20010/v1"
    assert any(
        entry["event"] == "openai.settings.embedding_base_url_override"
        and entry.get("base_url") == "http://***/v1"
        for entry in logs
    )


def test_embedding_http_base_url_requires_explicit_opt_in():
    env = {"EMBEDDING_API_BASE_URL": "http://localhost:20010/v1"}
    with capture_logs() as logs:
        with pytest.raises(ValueError):
            OpenAISettings.load(env, actor="pytest")
    assert any(entry["event"] == "openai.settings.insecure_embedding_base_url" for entry in logs)


@pytest.mark.parametrize("value", ["ftp://example.com", "https://"])
def test_invalid_embedding_base_url_raises(value):
    env = {"EMBEDDING_API_BASE_URL": value}
    with capture_logs() as logs:
        with pytest.raises(ValueError):
            OpenAISettings.load(env, actor="pytest")
    assert any(entry["event"] == "openai.settings.invalid_embedding_base_url" for entry in logs)


def test_invalid_insecure_flag_value_logs_and_raises():
    env = {
        "OPENAI_BASE_URL": "https://gateway.example.com/v1",
        "OPENAI_ALLOW_INSECURE_BASE_URL": "maybe",
    }
    with capture_logs() as logs:
        with pytest.raises(ValueError):
            OpenAISettings.load(env, actor="pytest")
    assert any(entry["event"] == "openai.settings.invalid_insecure_flag" for entry in logs)


def test_settings_expected_embedding_dimensions_uses_override():
    """Test expected_embedding_dimensions returns override when set."""
    settings = OpenAISettings.load(
        {"OPENAI_EMBEDDING_DIMENSIONS": "2048"},
        actor="pytest"
    )
    assert settings.expected_embedding_dimensions() == 2048


def test_settings_expected_embedding_dimensions_defaults():
    """Test expected_embedding_dimensions returns default when no override."""
    settings = OpenAISettings.load({}, actor="pytest")
    assert settings.expected_embedding_dimensions() == DEFAULT_EMBEDDING_DIMENSIONS


def test_settings_is_chat_override_property():
    """Test is_chat_override property correctly identifies overrides."""
    default_settings = OpenAISettings.load({}, actor="pytest")
    assert default_settings.is_chat_override is False

    override_settings = OpenAISettings.load(
        {"OPENAI_MODEL": "gpt-4o-mini"},
        actor="pytest"
    )
    assert override_settings.is_chat_override is True


def test_settings_allowed_chat_models_property():
    """Test allowed_chat_models property returns correct set."""
    settings = OpenAISettings.load({}, actor="pytest")
    allowed = settings.allowed_chat_models
    assert DEFAULT_CHAT_MODEL in allowed
    assert all(model in allowed for model in FALLBACK_CHAT_MODELS)


def test_settings_load_with_all_defaults():
    """Test OpenAISettings.load uses all default values correctly."""
    settings = OpenAISettings.load({}, actor="test-actor")

    assert settings.chat_model == DEFAULT_CHAT_MODEL
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.embedding_dimensions == DEFAULT_EMBEDDING_DIMENSIONS
    assert settings.embedding_dimensions_override is None
    assert settings.embedding_api_base_url is None
    assert settings.embedding_api_key is None
    assert settings.actor == "test-actor"
    assert settings.max_attempts == DEFAULT_MAX_RETRY_ATTEMPTS
    assert settings.backoff_seconds == DEFAULT_BACKOFF_SECONDS
    assert settings.enable_fallback is True
    assert settings.api_base_url is None
    assert settings.allow_insecure_base_url is False


def test_settings_max_attempts_zero_rejected():
    """Test zero max_attempts is rejected."""
    env = {"OPENAI_MAX_ATTEMPTS": "0"}
    with pytest.raises(ValueError, match="must be a positive integer"):
        OpenAISettings.load(env, actor="pytest")


def test_settings_backoff_seconds_zero_rejected():
    """Test zero backoff_seconds is rejected."""
    env = {"OPENAI_BACKOFF_SECONDS": "0"}
    with pytest.raises(ValueError, match="must be greater than zero"):
        OpenAISettings.load(env, actor="pytest")


def test_settings_backoff_seconds_negative_rejected():
    """Test negative backoff_seconds is rejected."""
    env = {"OPENAI_BACKOFF_SECONDS": "-0.5"}
    with pytest.raises(ValueError, match="must be greater than zero"):
        OpenAISettings.load(env, actor="pytest")


def test_settings_enable_fallback_accepts_variants():
    """Test OPENAI_ENABLE_FALLBACK accepts various true/false variants."""
    for true_val in ["1", "true", "TRUE", "yes", "YES", "on", "ON"]:
        settings = OpenAISettings.load(
            {"OPENAI_ENABLE_FALLBACK": true_val},
            actor="pytest"
        )
        assert settings.enable_fallback is True

    for false_val in ["0", "false", "FALSE", "no", "NO", "off", "OFF"]:
        settings = OpenAISettings.load(
            {"OPENAI_ENABLE_FALLBACK": false_val},
            actor="pytest"
        )
        assert settings.enable_fallback is False


def test_settings_enable_fallback_rejects_invalid():
    """Test OPENAI_ENABLE_FALLBACK rejects invalid values."""
    env = {"OPENAI_ENABLE_FALLBACK": "maybe"}
    with pytest.raises(ValueError, match="Use one of"):
        OpenAISettings.load(env, actor="pytest")


def test_settings_insecure_flag_accepts_variants():
    """Test OPENAI_ALLOW_INSECURE_BASE_URL accepts standard true/false variants."""
    for true_val in ["true", "TRUE", "1", "yes", "YES", "on", "ON"]:
        env = {
            "OPENAI_BASE_URL": "http://localhost:8000",
            "OPENAI_ALLOW_INSECURE_BASE_URL": true_val,
        }
        settings = OpenAISettings.load(env, actor="pytest")
        assert settings.allow_insecure_base_url is True

    for false_val in ["false", "FALSE", "0", "no", "NO", "off", "OFF"]:
        env = {
            "OPENAI_BASE_URL": "https://gateway.example.com",
            "OPENAI_ALLOW_INSECURE_BASE_URL": false_val,
        }
        settings = OpenAISettings.load(env, actor="pytest")
        assert settings.allow_insecure_base_url is False


def test_settings_base_url_without_netloc_rejected():
    """Test base URL without netloc is rejected."""
    env = {"OPENAI_BASE_URL": "https://"}
    with pytest.raises(ValueError, match="Provide an absolute http"):
        OpenAISettings.load(env, actor="pytest")


def test_settings_base_url_with_ftp_scheme_rejected():
    """Test base URL with unsupported scheme is rejected."""
    env = {"OPENAI_BASE_URL": "ftp://example.com/api"}
    with pytest.raises(ValueError, match="Provide an absolute http"):
        OpenAISettings.load(env, actor="pytest")


def test_settings_base_url_preserves_path():
    """Test base URL path is preserved in settings."""
    env = {"OPENAI_BASE_URL": "https://api.example.com/v1/custom"}
    settings = OpenAISettings.load(env, actor="pytest")
    assert settings.api_base_url == "https://api.example.com/v1/custom"


def test_settings_embedding_model_override():
    """Test OPENAI_EMBEDDING_MODEL can be overridden."""
    env = {"OPENAI_EMBEDDING_MODEL": "text-embedding-ada-002"}
    settings = OpenAISettings.load(env, actor="pytest")
    assert settings.embedding_model == "text-embedding-ada-002"


def test_settings_embedding_model_prefers_embedding_env():
    env = {
        "EMBEDDING_MODEL": "local-embed",
        "OPENAI_EMBEDDING_MODEL": "fallback-embed",
    }
    settings = OpenAISettings.load(env, actor="pytest")
    assert settings.embedding_model == "local-embed"


def test_settings_embedding_model_strips_whitespace():
    """Test embedding model value is stripped of whitespace."""
    env = {"OPENAI_EMBEDDING_MODEL": "  text-embedding-3-large  "}
    settings = OpenAISettings.load(env, actor="pytest")
    assert settings.embedding_model == "text-embedding-3-large"


def test_settings_chat_model_strips_whitespace():
    """Test chat model value is stripped of whitespace."""
    env = {"OPENAI_MODEL": "  gpt-4.1-mini  "}
    settings = OpenAISettings.load(env, actor="pytest")
    assert settings.chat_model == "gpt-4.1-mini"


def test_settings_actor_from_env_variable():
    """Test actor can be set via GRAPH_RAG_ACTOR env variable."""
    env = {"GRAPH_RAG_ACTOR": "ci-pipeline"}
    settings = OpenAISettings.load(env, actor=None)
    assert settings.actor == "ci-pipeline"


def test_settings_actor_explicit_overrides_env():
    """Test explicit actor parameter overrides environment."""
    env = {"GRAPH_RAG_ACTOR": "ci-pipeline"}
    settings = OpenAISettings.load(env, actor="explicit-actor")
    assert settings.actor == "explicit-actor"


def test_settings_frozen_dataclass():
    """Test OpenAISettings is immutable (frozen)."""
    settings = OpenAISettings.load({}, actor="pytest")
    with pytest.raises((FrozenInstanceError, AttributeError, ValidationError)):
        settings.chat_model = "different-model"


def test_settings_embedding_dimensions_very_large():
    """Test very large embedding dimensions are accepted."""
    env = {"OPENAI_EMBEDDING_DIMENSIONS": "4096"}
    settings = OpenAISettings.load(env, actor="pytest")
    assert settings.embedding_dimensions_override == 4096


def test_settings_max_attempts_very_large():
    """Test very large max_attempts value is accepted."""
    env = {"OPENAI_MAX_ATTEMPTS": "100"}
    settings = OpenAISettings.load(env, actor="pytest")
    assert settings.max_attempts == 100


def test_settings_backoff_fractional_seconds():
    """Test fractional backoff_seconds are handled correctly."""
    env = {"OPENAI_BACKOFF_SECONDS": "0.123"}
    settings = OpenAISettings.load(env, actor="pytest")
    assert settings.backoff_seconds == pytest.approx(0.123)


def test_mask_base_url_utility():
    """Test shared mask_base_url helper redacts host information."""
    from cli.sanitizer import mask_base_url

    assert mask_base_url("https://api.example.com/v1") == "https://***/v1"
    assert mask_base_url("http://localhost:8080") == "http://***"
    assert mask_base_url("invalid url") == "***"


def test_settings_logging_on_successful_load():
    """Test settings load logs appropriate messages."""
    with capture_logs() as logs:
        OpenAISettings.load({}, actor="pytest")

    # Should not log errors for default settings
    assert not any(entry.get("level") == "error" for entry in logs)


def test_settings_fallback_disabled_with_default_model():
    """Test fallback can be disabled with default model."""
    env = {"OPENAI_ENABLE_FALLBACK": "false"}
    settings = OpenAISettings.load(env, actor="pytest")
    assert settings.enable_fallback is False
    assert settings.chat_model == DEFAULT_CHAT_MODEL


def test_settings_http_without_opt_in_logs_error():
    """Test http base URL without opt-in logs error."""
    env = {"OPENAI_BASE_URL": "http://example.com"}
    with capture_logs() as logs:
        with pytest.raises(ValueError):
            OpenAISettings.load(env, actor="pytest")
    assert any(
        entry["event"] == "openai.settings.insecure_base_url"
        for entry in logs
    )


def test_settings_insecure_flag_false_explicit():
    """Test explicit false for insecure flag."""
    env = {
        "OPENAI_BASE_URL": "https://example.com",
        "OPENAI_ALLOW_INSECURE_BASE_URL": "false"
    }
    with capture_logs() as logs:
        settings = OpenAISettings.load(env, actor="pytest")
    assert settings.allow_insecure_base_url is False
    assert any(
        entry["event"] == "openai.settings.insecure_flag_disabled"
        for entry in logs
    )
