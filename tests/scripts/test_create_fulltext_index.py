"""Unit tests for the full-text index provisioning script."""

from __future__ import annotations

import logging
import sys
from importlib import util
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "create_fulltext_index.py"
)
PACKAGE_NAME = "scripts"
MODULE_NAME = f"{PACKAGE_NAME}.create_fulltext_index"

package_spec = util.spec_from_file_location(
    PACKAGE_NAME, MODULE_PATH.parent / "__init__.py"
)
module_spec = util.spec_from_file_location(MODULE_NAME, MODULE_PATH)

if (
    package_spec is None
    or package_spec.loader is None
    or module_spec is None
    or module_spec.loader is None
):  # pragma: no cover - defensive
    raise RuntimeError("Unable to load create_fulltext_index module for testing")

package_module = util.module_from_spec(package_spec)
module = util.module_from_spec(module_spec)

assert isinstance(package_module, ModuleType)
assert isinstance(module, ModuleType)

sys.modules[PACKAGE_NAME] = package_module
sys.modules[MODULE_NAME] = module

package_spec.loader.exec_module(package_module)  # type: ignore[arg-type]
module_spec.loader.exec_module(module)  # type: ignore[arg-type]

setattr(package_module, "create_fulltext_index", module)

_module = module

ConfigurationError = getattr(_module, "ConfigurationError")
IndexConfig = getattr(_module, "IndexConfig")
Neo4jReadinessTimeout = getattr(_module, "Neo4jReadinessTimeout")
ensure_fulltext_index = getattr(_module, "ensure_fulltext_index")
load_config = getattr(_module, "load_config")
wait_for_readiness = getattr(_module, "wait_for_readiness")


class FakeRecord:
    """Minimal stand-in for neo4j.Record supporting subscript and get."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def get(self, key: str, default: Any | None = None) -> Any:
        return self._values.get(key, default)

    def keys(self) -> list[str]:
        return list(self._values.keys())

    def items(self) -> list[tuple[str, Any]]:
        return list(self._values.items())


class FakeDriver:
    """Test double that mimics Neo4j driver behaviour for execute_query."""

    def __init__(
        self,
        *,
        index_exists: bool = False,
        ready_after_attempts: int = 1,
        fail_on_ping: bool = False,
    ) -> None:
        self.index_exists = index_exists
        self.calls: list[dict[str, Any]] = []
        self._connect_attempts = 0
        self.ready_after_attempts = ready_after_attempts
        self.fail_on_ping = fail_on_ping

    def __enter__(self) -> "FakeDriver":  # pragma: no cover - simple passthrough
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: Any | None,
    ) -> bool:  # pragma: no cover - simple passthrough
        self.close()
        return False

    def verify_connectivity(self) -> None:
        self._connect_attempts += 1
        if self._connect_attempts < self.ready_after_attempts:
            raise RuntimeError("Neo4j still starting")

    def execute_query(
        self, query: str, parameters: dict[str, Any], database_: str | None = None
    ):
        self.calls.append(
            {
                "query": query,
                "parameters": parameters,
                "database": database_,
            }
        )
        if "RETURN 1" in query and self.fail_on_ping:
            raise RuntimeError("Ping failure")
        if "SHOW INDEXES" in query:
            record = FakeRecord({"count": 1 if self.index_exists else 0})
            return ([record], None, None)
        if "CREATE FULLTEXT INDEX" in query:
            self.index_exists = True
            return ([], None, None)
        return ([], None, None)

    def close(self) -> None:  # pragma: no cover - required by driver protocol
        pass


@pytest.fixture(name="config")
def fixture_config() -> IndexConfig:
    return IndexConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="secret",
        database="neo4j",
        index_name="chunk_text_fulltext",
        label="Chunk",
        property_name="text",
        ready_attempts=3,
        ready_delay_seconds=0.01,
    )


def test_ensure_fulltext_index_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, config: IndexConfig
) -> None:
    events: list[dict[str, Any]] = []

    def fake_emit(level: int, event: str, *, logger, **fields: Any) -> None:  # noqa: ANN001
        fields.update({"level": level, "event": event})
        events.append(fields)

    created: list[str] = []

    def fake_create(
        driver: FakeDriver,
        name: str,
        *,
        label: str,
        node_properties: list[str],
        fail_if_exists: bool = False,
        neo4j_database: str | None = None,
    ) -> None:
        created.append(name)
        driver.index_exists = True

    monkeypatch.setattr("scripts.create_fulltext_index.emit_event", fake_emit)
    monkeypatch.setattr(
        "scripts.create_fulltext_index.graphrag_create_fulltext_index",
        fake_create,
    )

    driver = FakeDriver(index_exists=False)

    ensure_fulltext_index(config, driver=driver, logger=logging.getLogger("test"))
    ensure_fulltext_index(config, driver=driver, logger=logging.getLogger("test"))

    assert [event["event"] for event in events] == [
        "fulltext_index_created",
        "fulltext_index_already_exists",
    ]
    assert events[0]["status"] == "created"
    assert events[1]["status"] == "skipped"
    assert created == ["chunk_text_fulltext"]


def test_load_config_missing_variables_raises_configuration_error() -> None:
    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
    }

    with pytest.raises(ConfigurationError) as error_info:
        load_config(env)

    message = str(error_info.value)
    assert "NEO4J_PASSWORD" in message
    assert "neo4j" not in message  # Ensure credentials are not leaked


def test_load_config_applies_defaults() -> None:
    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
    }

    config = load_config(env)
    assert config.database == "neo4j"
    assert config.index_name == "chunk_text_fulltext"
    assert config.label == "Chunk"
    assert config.property_name == "text"
    assert config.ready_attempts == 10
    assert config.ready_delay_seconds == 3.0


def test_env_example_contains_new_index_variables() -> None:
    with open(".env.example", "r", encoding="utf-8") as handle:
        content = handle.read()

    required_lines = {
        "FULLTEXT_INDEX_NAME=",
        "FULLTEXT_LABEL=",
        "FULLTEXT_PROPERTY=",
        "FULLTEXT_READY_ATTEMPTS=",
        "FULLTEXT_READY_DELAY=",
    }

    for line in required_lines:
        assert line in content, f"Expected {line} in .env.example"


def test_wait_for_readiness_retries_until_ready(
    monkeypatch: pytest.MonkeyPatch, config: IndexConfig
) -> None:
    events: list[dict[str, Any]] = []

    def fake_emit(level: int, event: str, *, logger, **fields: Any) -> None:  # noqa: ANN001
        fields.update({"level": level, "event": event})
        events.append(fields)

    monkeypatch.setattr("scripts.create_fulltext_index.emit_event", fake_emit)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    driver = FakeDriver(index_exists=False, ready_after_attempts=3)

    wait_for_readiness(
        driver,
        config.database,
        attempts=config.ready_attempts,
        delay_seconds=config.ready_delay_seconds,
        logger=logging.getLogger("test"),
    )

    emitted_events = [event["event"] for event in events]
    assert emitted_events.count("fulltext_index_waiting_for_neo4j") == 2
    assert emitted_events[-1] == "fulltext_index_neo4j_ready"


def test_wait_for_readiness_raises_after_timeout(
    monkeypatch: pytest.MonkeyPatch, config: IndexConfig
) -> None:
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    driver = FakeDriver(index_exists=False, ready_after_attempts=5)

    with pytest.raises(Neo4jReadinessTimeout):
        wait_for_readiness(
            driver,
            config.database,
            attempts=2,
            delay_seconds=0,
            logger=logging.getLogger("test"),
        )


# ============================================================================
# Additional comprehensive tests for improved coverage
# ============================================================================


def test_configure_logging_sets_up_handler() -> None:
    """Test that configure_logging sets up a handler with correct settings."""
    from scripts.create_fulltext_index import LOGGER, configure_logging

    # Clear any existing handlers
    LOGGER.handlers.clear()
    LOGGER.propagate = True

    configure_logging()

    assert len(LOGGER.handlers) == 1
    assert isinstance(LOGGER.handlers[0], logging.StreamHandler)
    assert LOGGER.level == logging.INFO
    assert LOGGER.propagate is False


def test_configure_logging_is_idempotent() -> None:
    """Test that calling configure_logging multiple times doesn't add duplicate handlers."""
    from scripts.create_fulltext_index import LOGGER, configure_logging

    LOGGER.handlers.clear()

    configure_logging()
    handler_count = len(LOGGER.handlers)

    configure_logging()
    configure_logging()

    assert len(LOGGER.handlers) == handler_count


def test_emit_event_produces_valid_json() -> None:
    """Test that emit_event produces valid JSON with correct structure."""
    from scripts.create_fulltext_index import emit_event
    import json
    from io import StringIO

    logger = logging.getLogger("test_emit")
    logger.handlers.clear()
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    emit_event(
        logging.INFO,
        "test_event",
        logger=logger,
        field1="value1",
        field2=42,
    )

    output = stream.getvalue().strip()
    parsed = json.loads(output)

    assert parsed["event"] == "test_event"
    assert parsed["level"] == "info"
    assert parsed["field1"] == "value1"
    assert parsed["field2"] == 42
    assert "timestamp" in parsed


def test_emit_event_handles_different_log_levels() -> None:
    """Test that emit_event correctly handles different log levels."""
    from scripts.create_fulltext_index import emit_event
    import json
    from io import StringIO

    logger = logging.getLogger("test_levels")
    logger.handlers.clear()
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    emit_event(logging.WARNING, "warning_event", logger=logger)
    emit_event(logging.ERROR, "error_event", logger=logger)
    emit_event(logging.DEBUG, "debug_event", logger=logger)

    lines = stream.getvalue().strip().split("\n")
    assert len(lines) == 3

    parsed_warning = json.loads(lines[0])
    parsed_error = json.loads(lines[1])
    parsed_debug = json.loads(lines[2])

    assert parsed_warning["level"] == "warning"
    assert parsed_error["level"] == "error"
    assert parsed_debug["level"] == "debug"


def test_index_config_safe_payload_excludes_credentials() -> None:
    """Test that safe_payload does not leak sensitive credentials."""
    config = IndexConfig(
        uri="bolt://localhost:7687",
        username="secret_user",
        password="super_secret_password",
        database="neo4j",
        index_name="test_index",
        label="TestLabel",
        property_name="test_prop",
        ready_attempts=5,
        ready_delay_seconds=1.0,
    )

    payload = config.safe_payload()

    assert "password" not in payload
    assert "username" not in payload
    assert "uri" not in payload
    assert "super_secret_password" not in str(payload)
    assert "secret_user" not in str(payload)
    assert payload["index_name"] == "test_index"
    assert payload["database"] == "neo4j"
    assert payload["label"] == "TestLabel"
    assert payload["property"] == "test_prop"


def test_index_config_is_frozen() -> None:
    """Test that IndexConfig is immutable (frozen dataclass)."""
    config = IndexConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="secret",
        database="neo4j",
        index_name="test_index",
        label="TestLabel",
        property_name="test_prop",
        ready_attempts=5,
        ready_delay_seconds=1.0,
    )

    with pytest.raises(AttributeError):
        config.password = "new_password"  # type: ignore[misc]


def test_load_config_with_all_custom_values() -> None:
    """Test load_config with all optional environment variables set to custom values."""
    env = {
        "NEO4J_URI": "bolt://custom-host:7687",
        "NEO4J_USERNAME": "custom_user",
        "NEO4J_PASSWORD": "custom_pass",
        "NEO4J_DATABASE": "custom_db",
        "FULLTEXT_INDEX_NAME": "custom_index",
        "FULLTEXT_LABEL": "CustomLabel",
        "FULLTEXT_PROPERTY": "custom_prop",
        "FULLTEXT_READY_ATTEMPTS": "20",
        "FULLTEXT_READY_DELAY": "5.5",
    }

    config = load_config(env)

    assert config.uri == "bolt://custom-host:7687"
    assert config.username == "custom_user"
    assert config.password == "custom_pass"
    assert config.database == "custom_db"
    assert config.index_name == "custom_index"
    assert config.label == "CustomLabel"
    assert config.property_name == "custom_prop"
    assert config.ready_attempts == 20
    assert config.ready_delay_seconds == 5.5


def test_load_config_invalid_ready_attempts_not_integer() -> None:
    """Test that load_config raises ConfigurationError for non-integer FULLTEXT_READY_ATTEMPTS."""
    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
        "FULLTEXT_READY_ATTEMPTS": "not_a_number",
    }

    with pytest.raises(ConfigurationError) as error_info:
        load_config(env)

    assert "FULLTEXT_READY_ATTEMPTS must be an integer" in str(error_info.value)


def test_load_config_invalid_ready_attempts_less_than_one() -> None:
    """Test that load_config raises ConfigurationError for FULLTEXT_READY_ATTEMPTS < 1."""
    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
        "FULLTEXT_READY_ATTEMPTS": "0",
    }

    with pytest.raises(ConfigurationError) as error_info:
        load_config(env)

    assert "FULLTEXT_READY_ATTEMPTS must be >= 1" in str(error_info.value)


def test_load_config_invalid_ready_attempts_negative() -> None:
    """Test that load_config raises ConfigurationError for negative FULLTEXT_READY_ATTEMPTS."""
    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
        "FULLTEXT_READY_ATTEMPTS": "-5",
    }

    with pytest.raises(ConfigurationError) as error_info:
        load_config(env)

    assert "FULLTEXT_READY_ATTEMPTS must be >= 1" in str(error_info.value)


def test_load_config_invalid_ready_delay_not_number() -> None:
    """Test that load_config raises ConfigurationError for non-numeric FULLTEXT_READY_DELAY."""
    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
        "FULLTEXT_READY_DELAY": "invalid",
    }

    with pytest.raises(ConfigurationError) as error_info:
        load_config(env)

    assert "FULLTEXT_READY_DELAY must be a number" in str(error_info.value)


def test_load_config_invalid_ready_delay_negative() -> None:
    """Test that load_config raises ConfigurationError for negative FULLTEXT_READY_DELAY."""
    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
        "FULLTEXT_READY_DELAY": "-1.5",
    }

    with pytest.raises(ConfigurationError) as error_info:
        load_config(env)

    assert "FULLTEXT_READY_DELAY must be >= 0" in str(error_info.value)


def test_load_config_ready_delay_zero_is_valid() -> None:
    """Test that load_config accepts zero as valid value for FULLTEXT_READY_DELAY."""
    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
        "FULLTEXT_READY_DELAY": "0",
    }

    config = load_config(env)
    assert config.ready_delay_seconds == 0.0


def test_load_config_missing_multiple_variables() -> None:
    """Test that load_config reports all missing required variables."""
    env = {}

    with pytest.raises(ConfigurationError) as error_info:
        load_config(env)

    message = str(error_info.value)
    assert "NEO4J_URI" in message
    assert "NEO4J_USERNAME" in message
    assert "NEO4J_PASSWORD" in message


def test_load_config_missing_uri_only() -> None:
    """Test that load_config raises ConfigurationError when only NEO4J_URI is missing."""
    env = {
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
    }

    with pytest.raises(ConfigurationError) as error_info:
        load_config(env)

    message = str(error_info.value)
    assert "NEO4J_URI" in message


def test_load_config_missing_username_only() -> None:
    """Test that load_config raises ConfigurationError when only NEO4J_USERNAME is missing."""
    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_PASSWORD": "secret",
    }

    with pytest.raises(ConfigurationError) as error_info:
        load_config(env)

    message = str(error_info.value)
    assert "NEO4J_USERNAME" in message


def test_fulltext_index_exists_returns_true_when_index_exists() -> None:
    """Test that fulltext_index_exists returns True when the index is present."""
    from scripts.create_fulltext_index import fulltext_index_exists

    driver = FakeDriver(index_exists=True)
    result = fulltext_index_exists(driver, "test_index", "neo4j")

    assert result is True
    assert len(driver.calls) == 1
    assert "SHOW INDEXES" in driver.calls[0]["query"]
    assert driver.calls[0]["parameters"]["name"] == "test_index"


def test_fulltext_index_exists_returns_false_when_index_absent() -> None:
    """Test that fulltext_index_exists returns False when the index is not present."""
    from scripts.create_fulltext_index import fulltext_index_exists

    driver = FakeDriver(index_exists=False)
    result = fulltext_index_exists(driver, "test_index", "neo4j")

    assert result is False


def test_wait_for_readiness_succeeds_immediately() -> None:
    """Test that wait_for_readiness succeeds when Neo4j is ready immediately."""
    from scripts.create_fulltext_index import wait_for_readiness

    driver = FakeDriver(ready_after_attempts=1)
    logger = logging.getLogger("test")

    # Should not raise
    wait_for_readiness(driver, "neo4j", attempts=3, delay_seconds=0.01, logger=logger)

    assert driver._connect_attempts == 1


def test_wait_for_readiness_fails_on_execute_query() -> None:
    """Test that wait_for_readiness handles execute_query failures during ping."""
    from scripts.create_fulltext_index import wait_for_readiness

    driver = FakeDriver(ready_after_attempts=1, fail_on_ping=True)
    logger = logging.getLogger("test")

    with pytest.raises(Neo4jReadinessTimeout):
        wait_for_readiness(driver, "neo4j", attempts=2, delay_seconds=0, logger=logger)


def test_ensure_fulltext_index_creates_with_correct_parameters(
    monkeypatch: pytest.MonkeyPatch, config: IndexConfig
) -> None:
    """Test that ensure_fulltext_index passes correct parameters to graphrag function."""
    created_calls: list[dict[str, Any]] = []

    def fake_create(
        driver: FakeDriver,
        name: str,
        *,
        label: str,
        node_properties: list[str],
        fail_if_exists: bool = False,
        neo4j_database: str | None = None,
    ) -> None:
        created_calls.append(
            {
                "driver": driver,
                "name": name,
                "label": label,
                "node_properties": node_properties,
                "fail_if_exists": fail_if_exists,
                "neo4j_database": neo4j_database,
            }
        )
        driver.index_exists = True

    monkeypatch.setattr(
        "scripts.create_fulltext_index.emit_event", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "scripts.create_fulltext_index.graphrag_create_fulltext_index",
        fake_create,
    )

    driver = FakeDriver(index_exists=False)
    ensure_fulltext_index(config, driver=driver, logger=logging.getLogger("test"))

    assert len(created_calls) == 1
    call = created_calls[0]
    assert call["name"] == "chunk_text_fulltext"
    assert call["label"] == "Chunk"
    assert call["node_properties"] == ["text"]
    assert call["fail_if_exists"] is False
    assert call["neo4j_database"] == "neo4j"


def test_ensure_fulltext_index_returns_created_status(
    monkeypatch: pytest.MonkeyPatch, config: IndexConfig
) -> None:
    """Test that ensure_fulltext_index returns 'created' status when creating new index."""
    monkeypatch.setattr(
        "scripts.create_fulltext_index.emit_event", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "scripts.create_fulltext_index.graphrag_create_fulltext_index",
        lambda *args, **kwargs: None,
    )

    driver = FakeDriver(index_exists=False)
    result = ensure_fulltext_index(
        config, driver=driver, logger=logging.getLogger("test")
    )

    assert result == "created"


def test_ensure_fulltext_index_returns_skipped_status(
    monkeypatch: pytest.MonkeyPatch, config: IndexConfig
) -> None:
    """Test that ensure_fulltext_index returns 'skipped' status when index already exists."""
    monkeypatch.setattr(
        "scripts.create_fulltext_index.emit_event", lambda *args, **kwargs: None
    )

    driver = FakeDriver(index_exists=True)
    result = ensure_fulltext_index(
        config, driver=driver, logger=logging.getLogger("test")
    )

    assert result == "skipped"


def test_main_success_with_existing_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main function succeeds when index already exists."""
    from scripts.create_fulltext_index import main

    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
    }

    monkeypatch.setattr("os.environ", env)
    monkeypatch.setattr("scripts.create_fulltext_index.load_dotenv", lambda x: None)
    monkeypatch.setattr(
        "scripts.create_fulltext_index.emit_event", lambda *args, **kwargs: None
    )

    driver_instance = FakeDriver(index_exists=True, ready_after_attempts=1)

    def fake_driver_constructor(uri: str, auth: tuple[str, str]):
        return driver_instance

    monkeypatch.setattr(
        "scripts.create_fulltext_index.GraphDatabase.driver", fake_driver_constructor
    )

    exit_code = main()

    assert exit_code == 0


def test_main_success_with_new_index_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main function succeeds when creating a new index."""
    from scripts.create_fulltext_index import main

    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
    }

    monkeypatch.setattr("os.environ", env)
    monkeypatch.setattr("scripts.create_fulltext_index.load_dotenv", lambda x: None)
    monkeypatch.setattr(
        "scripts.create_fulltext_index.emit_event", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "scripts.create_fulltext_index.graphrag_create_fulltext_index",
        lambda *args, **kwargs: None,
    )

    driver_instance = FakeDriver(index_exists=False, ready_after_attempts=1)

    def fake_driver_constructor(uri: str, auth: tuple[str, str]):
        return driver_instance

    monkeypatch.setattr(
        "scripts.create_fulltext_index.GraphDatabase.driver", fake_driver_constructor
    )

    exit_code = main()

    assert exit_code == 0


def test_main_fails_on_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main function returns 1 on configuration error."""
    from scripts.create_fulltext_index import main

    env = {}  # Missing required variables

    monkeypatch.setattr("os.environ", env)
    monkeypatch.setattr("scripts.create_fulltext_index.load_dotenv", lambda x: None)
    monkeypatch.setattr(
        "scripts.create_fulltext_index.emit_event", lambda *args, **kwargs: None
    )

    exit_code = main()

    assert exit_code == 1


def test_main_fails_on_readiness_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main function returns 1 when Neo4j readiness times out."""
    from scripts.create_fulltext_index import main

    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
        "FULLTEXT_READY_ATTEMPTS": "2",
        "FULLTEXT_READY_DELAY": "0",
    }

    monkeypatch.setattr("os.environ", env)
    monkeypatch.setattr("scripts.create_fulltext_index.load_dotenv", lambda x: None)
    monkeypatch.setattr(
        "scripts.create_fulltext_index.emit_event", lambda *args, **kwargs: None
    )

    driver_instance = FakeDriver(ready_after_attempts=10)  # Will never be ready

    def fake_driver_constructor(uri: str, auth: tuple[str, str]):
        return driver_instance

    monkeypatch.setattr(
        "scripts.create_fulltext_index.GraphDatabase.driver", fake_driver_constructor
    )

    exit_code = main()

    assert exit_code == 1


def test_main_fails_on_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test main function returns 1 when Neo4j ClientError occurs."""
    from scripts.create_fulltext_index import main
    from neo4j.exceptions import ClientError

    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
    }

    monkeypatch.setattr("os.environ", env)
    monkeypatch.setattr("scripts.create_fulltext_index.load_dotenv", lambda x: None)
    monkeypatch.setattr(
        "scripts.create_fulltext_index.emit_event", lambda *args, **kwargs: None
    )

    def fake_create_index(*args, **kwargs):
        raise ClientError("Test error")

    monkeypatch.setattr(
        "scripts.create_fulltext_index.graphrag_create_fulltext_index",
        fake_create_index,
    )

    driver_instance = FakeDriver(index_exists=False, ready_after_attempts=1)

    def fake_driver_constructor(uri: str, auth: tuple[str, str]):
        return driver_instance

    monkeypatch.setattr(
        "scripts.create_fulltext_index.GraphDatabase.driver", fake_driver_constructor
    )

    exit_code = main()

    assert exit_code == 1


def test_main_emits_correct_events_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that main function emits correct structured log events on success."""
    from scripts.create_fulltext_index import main

    env = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "secret",
    }

    events: list[dict[str, Any]] = []

    def capture_emit(level: int, event: str, *, logger, **fields: Any) -> None:  # noqa: ANN001
        fields.update({"level": level, "event": event})
        events.append(fields)

    monkeypatch.setattr("os.environ", env)
    monkeypatch.setattr("scripts.create_fulltext_index.load_dotenv", lambda x: None)
    monkeypatch.setattr("scripts.create_fulltext_index.emit_event", capture_emit)

    driver_instance = FakeDriver(index_exists=True, ready_after_attempts=1)

    def fake_driver_constructor(uri: str, auth: tuple[str, str]):
        return driver_instance

    monkeypatch.setattr(
        "scripts.create_fulltext_index.GraphDatabase.driver", fake_driver_constructor
    )

    exit_code = main()

    assert exit_code == 0
    event_names = [e["event"] for e in events]
    assert "fulltext_index_neo4j_ready" in event_names
    assert "fulltext_index_already_exists" in event_names


def test_main_emits_error_event_on_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that main function emits error event on configuration error."""
    from scripts.create_fulltext_index import main

    env = {}

    events: list[dict[str, Any]] = []

    def capture_emit(level: int, event: str, *, logger, **fields: Any) -> None:  # noqa: ANN001
        fields.update({"level": level, "event": event})
        events.append(fields)

    monkeypatch.setattr("os.environ", env)
    monkeypatch.setattr("scripts.create_fulltext_index.load_dotenv", lambda x: None)
    monkeypatch.setattr("scripts.create_fulltext_index.emit_event", capture_emit)

    exit_code = main()

    assert exit_code == 1
    assert len(events) == 1
    assert events[0]["event"] == "fulltext_index_configuration_error"
    assert events[0]["status"] == "error"
    assert "message" in events[0]


def test_wait_for_readiness_emits_warning_events(
    monkeypatch: pytest.MonkeyPatch, config: IndexConfig
) -> None:
    """Test that wait_for_readiness emits warning events during retries."""
    events: list[dict[str, Any]] = []

    def fake_emit(level: int, event: str, *, logger, **fields: Any) -> None:  # noqa: ANN001
        fields.update({"level": level, "event": event})
        events.append(fields)

    monkeypatch.setattr("scripts.create_fulltext_index.emit_event", fake_emit)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    driver = FakeDriver(ready_after_attempts=3)

    wait_for_readiness(
        driver,
        config.database,
        attempts=5,
        delay_seconds=1.0,
        logger=logging.getLogger("test"),
    )

    warning_events = [
        e for e in events if e["event"] == "fulltext_index_waiting_for_neo4j"
    ]
    assert len(warning_events) == 2
    for event in warning_events:
        assert event["status"] == "waiting"
        assert "attempt" in event
        assert "attempts" in event
        assert "delay_seconds" in event
        assert "message" in event


def test_configuration_error_inheritance() -> None:
    """Test that ConfigurationError inherits from RuntimeError."""
    assert issubclass(ConfigurationError, RuntimeError)


def test_neo4j_readiness_timeout_inheritance() -> None:
    """Test that Neo4jReadinessTimeout inherits from RuntimeError."""
    assert issubclass(Neo4jReadinessTimeout, RuntimeError)


def test_fake_driver_close_method_exists() -> None:
    """Test that FakeDriver implements close method required by driver protocol."""
    driver = FakeDriver()
    # Should not raise
    driver.close()
