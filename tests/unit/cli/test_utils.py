import pytest

from _compat.structlog import capture_logs

from config.settings import OpenAISettings
from cli.utils import ensure_embedding_dimensions


def _settings(override: int | None = None) -> OpenAISettings:
    """
    Construct OpenAISettings for pytest, optionally overriding the embedding dimensions.
    
    If `override` is provided, the environment used to load settings will include
    OPENAI_EMBEDDING_DIMENSIONS set to that value.
    
    Parameters:
        override (int | None): Embedding dimension to inject into the environment; when
            None the environment is unchanged.
    
    Returns:
        OpenAISettings: Settings loaded from the constructed environment.
    """
    env = {}
    if override is not None:
        env["OPENAI_EMBEDDING_DIMENSIONS"] = str(override)
    return OpenAISettings.load(env, actor="pytest")


def test_embedding_happy_path():
    settings = _settings()
    vector = [0.0] * settings.embedding_dimensions
    assert ensure_embedding_dimensions(vector, settings=settings) is vector


def test_embedding_override_accepts_custom_dimension():
    settings = _settings(3072)
    vector = [0.0] * 3072
    with capture_logs() as logs:
        ensure_embedding_dimensions(vector, settings=settings)
    assert any(entry["event"] == "openai.embedding.override_applied" for entry in logs)


def test_embedding_mismatch_without_override_raises():
    settings = _settings()
    vector = [0.0] * 1400
    with capture_logs() as logs:
        with pytest.raises(ValueError):
            ensure_embedding_dimensions(vector, settings=settings)
    assert any(entry["event"] == "openai.embedding.dimension_mismatch" for entry in logs)


def test_embedding_override_mismatch_raises():
    settings = _settings(3072)
    vector = [0.0] * 1024
    with capture_logs() as logs:
        with pytest.raises(ValueError):
            ensure_embedding_dimensions(vector, settings=settings)
    assert any(entry["event"] == "openai.embedding.override_mismatch" for entry in logs)


def test_explicit_override_argument_takes_precedence():
    base_settings = _settings()
    vector = [0.0] * 2048
    with capture_logs() as logs:
        with pytest.raises(ValueError):
            ensure_embedding_dimensions(vector, settings=base_settings)
    assert any(entry["event"] == "openai.embedding.dimension_mismatch" for entry in logs)
    # Provide explicit override to accept the vector length
    ensure_embedding_dimensions(vector, settings=base_settings, override_dimensions=2048)
