from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def reset_module_cache():
    """
    Autouse pytest fixture that reloads the fancyrag.utils.env module after each test to reset its state.
    
    This fixture imports the module before the test runs and reloads it after the test completes so that changes to module-level state do not leak between tests.
    """
    import fancyrag.utils.env as env_module

    yield

    importlib.reload(env_module)


def test_ensure_env_loads_dotenv_when_env_missing(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from_dotenv\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FANCYRAG_DOTENV_PATH", str(env_file))

    env_module = importlib.import_module("fancyrag.utils.env")
    importlib.reload(env_module)

    assert env_module.ensure_env("OPENAI_API_KEY") == "from_dotenv"


def test_existing_environment_variables_take_precedence(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from_dotenv\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "from_process_env")
    monkeypatch.setenv("FANCYRAG_DOTENV_PATH", str(env_file))

    env_module = importlib.import_module("fancyrag.utils.env")
    importlib.reload(env_module)

    assert env_module.ensure_env("OPENAI_API_KEY") == "from_process_env"
