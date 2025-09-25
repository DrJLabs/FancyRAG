import pytest
from structlog.testing import capture_logs

from config.settings import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
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


def test_actor_fallback_to_env_user(monkeypatch):
    monkeypatch.setenv("USER", "cli-user")
    settings = OpenAISettings.load({}, actor=None)
    assert settings.actor == "cli-user"


def test_allowed_chat_models_exposed():
    settings = OpenAISettings.load({}, actor="pytest")
    assert settings.allowed_chat_models == FALLBACK_CHAT_MODELS | {DEFAULT_CHAT_MODEL}
