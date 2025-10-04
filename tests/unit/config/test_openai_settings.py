import pytest

from _compat.structlog import capture_logs

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


def test_invalid_insecure_flag_value_logs_and_raises():
    env = {
        "OPENAI_BASE_URL": "https://gateway.example.com/v1",
        "OPENAI_ALLOW_INSECURE_BASE_URL": "maybe",
    }
    with capture_logs() as logs:
        with pytest.raises(ValueError):
            OpenAISettings.load(env, actor="pytest")
    assert any(entry["event"] == "openai.settings.invalid_insecure_flag" for entry in logs)
