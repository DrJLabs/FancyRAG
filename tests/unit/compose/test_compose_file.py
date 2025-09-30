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

    assert neo4j["image"] == "neo4j:5.26.0"
    assert qdrant["image"] == "qdrant/qdrant:1.9.4"
    assert "7474:7474" in neo4j["ports"]
    assert "7687:7687" in neo4j["ports"]
    assert "6333:6333" in qdrant["ports"]


def test_compose_env_variables_present(compose_data: dict[str, Any]) -> None:
    services = compose_data["services"]
    neo4j = services["neo4j"]
    qdrant = services["qdrant"]

    assert "NEO4J_AUTH" in neo4j["environment"]
    assert "NEO4JLABS_PLUGINS" in neo4j["environment"]
    assert "QDRANT__SERVICE__HTTP_PORT" in qdrant["environment"]

    neo4j_sources = {volume["source"] for volume in neo4j["volumes"]}
    qdrant_sources = {volume["source"] for volume in qdrant["volumes"]}
    assert "./.data/neo4j/data" in neo4j_sources
    assert "./.data/qdrant/storage" in qdrant_sources


def test_compose_healthchecks_defined(compose_data: dict[str, Any]) -> None:
    services = compose_data["services"]
    neo4j = services["neo4j"]
    qdrant = services["qdrant"]

    assert neo4j["healthcheck"]["test"][0] == "CMD-SHELL"
    assert "cypher-shell" in neo4j["healthcheck"]["test"][1]
    assert qdrant["healthcheck"]["test"][:2] == ["CMD", "curl"]
    assert qdrant["healthcheck"]["test"][3].endswith("/readyz")


def test_compose_restart_policy(compose_data: dict[str, Any]) -> None:
    services = compose_data["services"]
    assert services["neo4j"]["restart"] == "unless-stopped"
    assert services["qdrant"]["restart"] == "unless-stopped"
