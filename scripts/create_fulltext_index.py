"""Create or verify the Neo4j full-text index required for hybrid retrieval."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ClientError
from neo4j_graphrag.indexes import (
    create_fulltext_index as graphrag_create_fulltext_index,
)


LOGGER = logging.getLogger("fancryrag.fulltext_index")


class ConfigurationError(RuntimeError):
    """Raised when required environment configuration is missing."""


class Neo4jReadinessTimeout(RuntimeError):
    """Raised when Neo4j does not become ready within the allotted window."""


@dataclass(frozen=True)
class IndexConfig:
    """Configuration values required to ensure the full-text index exists."""

    uri: str
    username: str
    password: str
    database: str
    index_name: str
    label: str
    property_name: str
    ready_attempts: int
    ready_delay_seconds: float

    def safe_payload(self) -> dict[str, str]:
        """Return fields safe for logging without exposing credentials."""

        return {
            "index_name": self.index_name,
            "database": self.database,
            "label": self.label,
            "property": self.property_name,
        }


def configure_logging() -> None:
    """Attach a JSON-compatible stream handler for CLI usage."""

    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def emit_event(
    level: int, event: str, *, logger: logging.Logger, **fields: Any
) -> None:
    """Emit a structured log event with a standard envelope."""

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": logging.getLevelName(level).lower(),
        "event": event,
    }
    payload.update(fields)
    logger.log(level, json.dumps(payload, ensure_ascii=True))


def load_config(env: Mapping[str, str]) -> IndexConfig:
    """Load required configuration from the provided environment mapping."""

    required_keys = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    missing = [key for key in required_keys if not env.get(key)]
    if missing:
        raise ConfigurationError(
            "Missing required environment variables: " + ", ".join(sorted(missing))
        )

    database = env.get("NEO4J_DATABASE") or "neo4j"
    index_name = env.get("FULLTEXT_INDEX_NAME") or "chunk_text_fulltext"
    label = env.get("FULLTEXT_LABEL") or "Chunk"
    property_name = env.get("FULLTEXT_PROPERTY") or "text"

    try:
        ready_attempts = int(env.get("FULLTEXT_READY_ATTEMPTS", "10"))
    except ValueError as error:
        raise ConfigurationError(
            "FULLTEXT_READY_ATTEMPTS must be an integer"
        ) from error
    if ready_attempts < 1:
        raise ConfigurationError("FULLTEXT_READY_ATTEMPTS must be >= 1")

    try:
        ready_delay_seconds = float(env.get("FULLTEXT_READY_DELAY", "3"))
    except ValueError as error:
        raise ConfigurationError("FULLTEXT_READY_DELAY must be a number") from error
    if ready_delay_seconds < 0:
        raise ConfigurationError("FULLTEXT_READY_DELAY must be >= 0")

    return IndexConfig(
        uri=env["NEO4J_URI"],
        username=env["NEO4J_USERNAME"],
        password=env["NEO4J_PASSWORD"],
        database=database,
        index_name=index_name,
        label=label,
        property_name=property_name,
        ready_attempts=ready_attempts,
        ready_delay_seconds=ready_delay_seconds,
    )


def wait_for_readiness(
    driver: Driver,
    database: str,
    *,
    attempts: int,
    delay_seconds: float,
    logger: logging.Logger,
) -> None:
    """Block until Neo4j responds to simple queries or raise timeout."""

    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            driver.verify_connectivity()
            driver.execute_query("RETURN 1 AS ok", {}, database_=database)
        except Exception as error:
            last_error = error
            emit_event(
                logging.WARNING,
                "fulltext_index_waiting_for_neo4j",
                logger=logger,
                status="waiting",
                attempt=attempt,
                attempts=attempts,
                delay_seconds=delay_seconds,
                message=str(error),
            )
            if attempt < attempts:
                time.sleep(delay_seconds)
            continue

        emit_event(
            logging.INFO,
            "fulltext_index_neo4j_ready",
            logger=logger,
            status="ready",
            attempt=attempt,
            attempts=attempts,
        )
        return

    raise Neo4jReadinessTimeout(
        f"Neo4j not ready after {attempts} attempts with {delay_seconds}s delay"
    ) from last_error


def fulltext_index_exists(driver: Driver, name: str, database: str) -> bool:
    """Return True if the target full-text index already exists."""

    query = (
        "SHOW INDEXES YIELD name, type WHERE name = $name AND type = 'FULLTEXT' "
        "RETURN count(*) AS count"
    )
    records, _, _ = driver.execute_query(query, {"name": name}, database_=database)
    return bool(records[0]["count"])


def ensure_fulltext_index(
    config: IndexConfig, *, driver: Driver, logger: logging.Logger
) -> str:
    """Create the Neo4j full-text index if it is absent."""

    metadata = config.safe_payload()
    if fulltext_index_exists(driver, config.index_name, config.database):
        emit_event(
            logging.INFO,
            "fulltext_index_already_exists",
            logger=logger,
            status="skipped",
            **metadata,
        )
        return "skipped"

    graphrag_create_fulltext_index(
        driver,
        config.index_name,
        label=config.label,
        node_properties=[config.property_name],
        fail_if_exists=False,
        neo4j_database=config.database,
    )
    emit_event(
        logging.INFO,
        "fulltext_index_created",
        logger=logger,
        status="created",
        **metadata,
    )
    return "created"


def main() -> int:
    """CLI entrypoint that orchestrates configuration, creation, and logging."""

    configure_logging()
    load_dotenv(".env.local")

    try:
        config = load_config(os.environ)
    except ConfigurationError as error:
        emit_event(
            logging.ERROR,
            "fulltext_index_configuration_error",
            logger=LOGGER,
            status="error",
            message=str(error),
        )
        return 1

    try:
        driver = GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )
    except Exception as error:  # pragma: no cover - defensive, hard to simulate
        emit_event(
            logging.ERROR,
            "fulltext_index_driver_error",
            logger=LOGGER,
            status="error",
            **config.safe_payload(),
            message=str(error),
        )
        return 1

    try:
        with driver:
            wait_for_readiness(
                driver,
                config.database,
                attempts=config.ready_attempts,
                delay_seconds=config.ready_delay_seconds,
                logger=LOGGER,
            )
            ensure_fulltext_index(config, driver=driver, logger=LOGGER)
    except Neo4jReadinessTimeout as error:
        emit_event(
            logging.ERROR,
            "fulltext_index_readiness_timeout",
            logger=LOGGER,
            status="error",
            **config.safe_payload(),
            attempts=config.ready_attempts,
            delay_seconds=config.ready_delay_seconds,
            message=str(error),
        )
        return 1
    except ClientError as error:
        emit_event(
            logging.ERROR,
            "fulltext_index_execution_error",
            logger=LOGGER,
            status="error",
            **config.safe_payload(),
            message=str(error),
        )
        return 1
    except Exception as error:  # pragma: no cover - unexpected failure path
        emit_event(
            logging.ERROR,
            "fulltext_index_unexpected_error",
            logger=LOGGER,
            status="error",
            **config.safe_payload(),
            message=str(error),
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
