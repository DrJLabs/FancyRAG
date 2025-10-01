from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

COMPOSE_FILE = Path(__file__).resolve().parents[3] / "docker-compose.neo4j-qdrant.yml"


@pytest.fixture(scope="module")
def compose_data() -> dict[str, Any]:
    """Return the parsed docker-compose configuration for assertions."""
    if not COMPOSE_FILE.exists():
        pytest.skip("Compose file not present")
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


def test_compose_defines_expected_images_and_ports(compose_data: dict[str, Any]) -> None:
    services = compose_data["services"]
    neo4j = services["neo4j"]
    qdrant = services["qdrant"]

    assert neo4j["image"] == "neo4j:5.26.12"
    assert qdrant["image"] == "qdrant/qdrant:v1.15.4"
    assert any(port.endswith(":7474") for port in neo4j["ports"])
    assert any(port.endswith(":7687") for port in neo4j["ports"])
    assert any(port.endswith(":6333") for port in qdrant["ports"])


def test_compose_env_variables_present(compose_data: dict[str, Any]) -> None:
    """
    Validate that the docker-compose data declares required environment variables and volume sources for the neo4j and qdrant services.
    
    This test asserts that:
    - The neo4j service environment contains `NEO4J_AUTH` and at least one of `NEO4J_PLUGINS` or `NEO4JLABS_PLUGINS`.
    - The qdrant service environment contains `QDRANT__SERVICE__HTTP_PORT`.
    - The neo4j volumes include the source "./.data/neo4j/data".
    - The qdrant volumes include the source "./.data/qdrant/storage".
    
    Parameters:
        compose_data (dict[str, Any]): Parsed docker-compose YAML as a mapping with a top-level "services" key.
    """
    services = compose_data["services"]
    neo4j = services["neo4j"]
    qdrant = services["qdrant"]

    assert "NEO4J_AUTH" in neo4j["environment"]
    env_keys = set(neo4j["environment"].keys())
    assert "NEO4J_AUTH" in env_keys
    assert {"NEO4J_PLUGINS", "NEO4JLABS_PLUGINS"} & env_keys
    assert "QDRANT__SERVICE__HTTP_PORT" in qdrant["environment"]

    neo4j_sources = {volume["source"] for volume in neo4j["volumes"]}
    qdrant_sources = {volume["source"] for volume in qdrant["volumes"]}
    assert "./.data/neo4j/data" in neo4j_sources
    assert "./.data/qdrant/storage" in qdrant_sources


def test_compose_healthchecks_defined(compose_data: dict[str, Any]) -> None:
    """
    Verify that Neo4j and Qdrant services define expected healthchecks in the compose data.
    
    Checks:
    - Neo4j healthcheck `test` begins with `"CMD-SHELL"` and the command contains `"neo4j-admin"`.
    - Qdrant healthcheck `test` begins with `"CMD-SHELL"` and the command contains `"readyz"`.
    
    Parameters:
        compose_data (dict[str, Any]): Parsed docker-compose YAML as a mapping (the fixture-provided compose file data).
    """
    services = compose_data["services"]
    neo4j = services["neo4j"]
    qdrant = services["qdrant"]

    assert neo4j["healthcheck"]["test"][0] == "CMD-SHELL"
    assert "neo4j-admin" in neo4j["healthcheck"]["test"][1]

    qdrant_health = qdrant["healthcheck"]["test"]
    assert qdrant_health[0] == "CMD-SHELL"
    assert "readyz" in qdrant_health[1]


def test_compose_restart_policy(compose_data: dict[str, Any]) -> None:
    services = compose_data["services"]
    assert services["neo4j"]["restart"] == "unless-stopped"
    assert services["qdrant"]["restart"] == "unless-stopped"
