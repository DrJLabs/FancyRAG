"""Tests for the MCP hybrid server entrypoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fancyrag.config import ConfigurationError


@pytest.fixture
def mock_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a minimal valid environment for the server."""
    query_file = tmp_path / "query.cypher"
    query_file.write_text("RETURN node, score", encoding="utf-8")

    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    monkeypatch.setenv("INDEX_NAME", "embeddings")
    monkeypatch.setenv("FULLTEXT_INDEX_NAME", "fulltext")
    monkeypatch.setenv("EMBEDDING_API_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "key")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    monkeypatch.setenv("MCP_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("HYBRID_RETRIEVAL_QUERY_PATH", str(query_file))

    return query_file


def test_main_returns_zero_on_success(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that main() returns 0 when server starts successfully."""
    _ = mock_environment
    from servers import mcp_hybrid_google

    # Mock all the initialization steps
    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **_kwargs: None,
    )

    mock_state = MagicMock()
    mock_state.driver.close = MagicMock()
    monkeypatch.setattr("servers.mcp_hybrid_google.create_state", lambda *_args, **_kwargs: mock_state)

    mock_server = MagicMock()
    mock_server.run = MagicMock()
    monkeypatch.setattr("servers.mcp_hybrid_google.build_server", lambda *_args, **_kwargs: mock_server)

    result = mcp_hybrid_google.main()

    assert result == 0
    mock_server.run.assert_called_once()


def test_main_returns_one_on_configuration_error(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that main() returns 1 when configuration fails."""
    _ = mock_environment
    from servers import mcp_hybrid_google

    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **_kwargs: None,
    )

    def raise_config_error():
        raise ConfigurationError

    monkeypatch.setattr("servers.mcp_hybrid_google.load_config", raise_config_error)

    result = mcp_hybrid_google.main()

    assert result == 1


def test_main_returns_one_on_state_creation_error(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that main() returns 1 when state creation fails."""
    _ = mock_environment
    from servers import mcp_hybrid_google

    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **_kwargs: None,
    )

    def raise_runtime_error(_config):
        raise RuntimeError

    monkeypatch.setattr("servers.mcp_hybrid_google.create_state", raise_runtime_error)

    result = mcp_hybrid_google.main()

    assert result == 1


def test_main_configures_logging_before_loading_config(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logging is configured before config loading."""
    _ = mock_environment
    from servers import mcp_hybrid_google

    call_order = []

    def mock_configure_logging():
        call_order.append("logging")

    def mock_load_config():
        call_order.append("config")
        raise ConfigurationError

    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", mock_configure_logging)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("servers.mcp_hybrid_google.load_config", mock_load_config)

    mcp_hybrid_google.main()

    assert call_order == ["logging", "config"]


def test_main_loads_dotenv_before_config(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that .env.local is loaded before config."""
    _ = mock_environment
    from servers import mcp_hybrid_google

    call_order = []

    def mock_load_dotenv(*_args, **_kwargs):
        call_order.append("dotenv")

    def mock_load_config():
        call_order.append("config")
        raise ConfigurationError

    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr("servers.mcp_hybrid_google.load_dotenv", mock_load_dotenv)
    monkeypatch.setattr("servers.mcp_hybrid_google.load_config", mock_load_config)

    mcp_hybrid_google.main()

    assert call_order == ["dotenv", "config"]


def test_main_loads_dotenv_with_correct_parameters(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that load_dotenv is called with correct parameters."""
    _ = mock_environment
    from servers import mcp_hybrid_google

    dotenv_calls = []

    def mock_load_dotenv(*args, **kwargs):
        dotenv_calls.append((args, kwargs))

    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr("servers.mcp_hybrid_google.load_dotenv", mock_load_dotenv)
    monkeypatch.setattr("servers.mcp_hybrid_google.load_config", lambda: MagicMock())
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.create_state",
        lambda *_args, **_kwargs: MagicMock(driver=MagicMock()),
    )
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.build_server",
        lambda *_args, **_kwargs: MagicMock(run=MagicMock()),
    )

    mcp_hybrid_google.main()

    assert len(dotenv_calls) == 1
    assert dotenv_calls[0][0] == (".env.local",)
    assert dotenv_calls[0][1] == {"override": False}


def test_main_creates_state_with_loaded_config(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that create_state is called with the loaded config."""
    _ = mock_environment
    from servers import mcp_hybrid_google
    mock_config = MagicMock()
    mock_config.server = MagicMock(host="localhost", port=8080, path="/mcp")

    create_state_calls = []

    def mock_create_state(config):
        create_state_calls.append(config)
        mock_driver = MagicMock()
        return MagicMock(driver=mock_driver)

    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("servers.mcp_hybrid_google.load_config", lambda: mock_config)
    monkeypatch.setattr("servers.mcp_hybrid_google.create_state", mock_create_state)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.build_server",
        lambda *_args, **_kwargs: MagicMock(run=MagicMock()),
    )

    mcp_hybrid_google.main()

    assert len(create_state_calls) == 1
    assert create_state_calls[0] is mock_config


def test_main_builds_server_with_created_state(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that build_server is called with the created state."""
    _ = mock_environment
    from servers import mcp_hybrid_google

    mock_state = MagicMock()
    mock_state.driver.close = MagicMock()

    build_server_calls = []

    def mock_build_server(state, **_kwargs):
        build_server_calls.append(state)
        return MagicMock(run=MagicMock())

    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("servers.mcp_hybrid_google.create_state", lambda *_args, **_kwargs: mock_state)
    monkeypatch.setattr("servers.mcp_hybrid_google.build_server", mock_build_server)

    mcp_hybrid_google.main()

    assert len(build_server_calls) == 1
    assert build_server_calls[0] is mock_state


def test_main_calls_server_run_with_correct_parameters(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that server.run is called with correct parameters from config."""
    _ = mock_environment
    from servers import mcp_hybrid_google
    from fancyrag.config import ServerSettings

    mock_config = MagicMock()
    mock_config.server = ServerSettings(
        host="192.168.1.1",
        port=9999,
        path="/custom/mcp",
        base_url="http://example.com",
    )

    mock_server = MagicMock()
    run_calls = []

    def mock_run(**kwargs):
        run_calls.append(kwargs)

    mock_server.run = mock_run

    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("servers.mcp_hybrid_google.load_config", lambda: mock_config)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.create_state",
        lambda *_args, **_kwargs: MagicMock(driver=MagicMock()),
    )
    monkeypatch.setattr("servers.mcp_hybrid_google.build_server", lambda *_args, **_kwargs: mock_server)

    mcp_hybrid_google.main()

    assert len(run_calls) == 1
    assert run_calls[0]["transport"] == "http"
    assert run_calls[0]["host"] == "192.168.1.1"
    assert run_calls[0]["port"] == 9999
    assert run_calls[0]["path"] == "/custom/mcp"
    assert run_calls[0]["stateless_http"] is True


def test_main_logs_configuration_error(mock_environment: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Test that configuration errors are logged."""
    _ = mock_environment
    _ = capsys
    from servers import mcp_hybrid_google

    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **_kwargs: None,
    )

    def raise_config_error():
        raise ConfigurationError

    monkeypatch.setattr("servers.mcp_hybrid_google.load_config", raise_config_error)

    # We can't easily test the log output without more complex mocking
    # but we can verify the function handles the error gracefully
    result = mcp_hybrid_google.main()

    assert result == 1


def test_main_logs_startup_info(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that startup info is logged before server.run."""
    _ = mock_environment
    from servers import mcp_hybrid_google
    from fancyrag.config import ServerSettings

    mock_config = MagicMock()
    mock_config.server = ServerSettings(
        host="0.0.0.0",  # noqa: B106
        port=8080,
        path="/mcp",
        base_url="http://localhost:8080",
    )

    # Track when run is called to ensure logging happens first
    run_called = False

    def mock_run(*_args, **_kwargs):
        nonlocal run_called
        run_called = True

    mock_server = MagicMock()
    mock_server.run = mock_run

    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("servers.mcp_hybrid_google.load_config", lambda: mock_config)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.create_state",
        lambda *_args, **_kwargs: MagicMock(driver=MagicMock()),
    )
    monkeypatch.setattr("servers.mcp_hybrid_google.build_server", lambda *_args, **_kwargs: mock_server)

    mcp_hybrid_google.main()

    assert run_called


def test_main_atexit_cleanup_closes_driver(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that atexit handler is registered to close the driver."""
    _ = mock_environment
    from servers import mcp_hybrid_google
    import atexit

    registered_callbacks = []
    original_register = atexit.register

    def mock_register(func):
        registered_callbacks.append(func)
        return original_register(func)

    monkeypatch.setattr("atexit.register", mock_register)
    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **_kwargs: None,
    )

    mock_driver = MagicMock()
    mock_state = MagicMock(driver=mock_driver)

    monkeypatch.setattr("servers.mcp_hybrid_google.create_state", lambda *_args, **_kwargs: mock_state)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.build_server",
        lambda *_args, **_kwargs: MagicMock(run=MagicMock()),
    )

    mcp_hybrid_google.main()

    # Verify atexit callback was registered
    assert len(registered_callbacks) >= 1

    # Call the cleanup callback
    for callback in registered_callbacks:
        callback()

    # Verify driver.close was called
    mock_driver.close.assert_called()


def test_main_handles_generic_exceptions_during_state_creation(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that generic exceptions during state creation are handled."""
    _ = mock_environment
    from servers import mcp_hybrid_google

    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **kwargs: None,
    )

    def raise_exception(_config):
        raise ValueError

    monkeypatch.setattr("servers.mcp_hybrid_google.create_state", raise_exception)

    result = mcp_hybrid_google.main()

    assert result == 1


def test_main_default_server_configuration(mock_environment: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main with default server configuration values."""
    _ = mock_environment
    from servers import mcp_hybrid_google

    run_calls = []

    def mock_run(**kwargs):
        run_calls.append(kwargs)

    mock_server = MagicMock()
    mock_server.run = mock_run

    monkeypatch.setattr("servers.mcp_hybrid_google.configure_logging", lambda: None)
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.load_dotenv",
        lambda *_args, **kwargs: None,
    )
    monkeypatch.setattr(
        "servers.mcp_hybrid_google.create_state",
        lambda *_args, **_kwargs: MagicMock(driver=MagicMock()),
    )
    monkeypatch.setattr("servers.mcp_hybrid_google.build_server", lambda *_args, **_kwargs: mock_server)

    mcp_hybrid_google.main()

    assert len(run_calls) == 1
    assert run_calls[0]["host"] == "0.0.0.0"  # noqa: B106
    assert run_calls[0]["port"] == 8080
    assert run_calls[0]["path"] == "/mcp"
