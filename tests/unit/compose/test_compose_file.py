from __future__ import annotations

from pathlib import Path

import pytest

COMPOSE_FILE = Path(__file__).resolve().parents[3] / "docker-compose.neo4j-qdrant.yml"


@pytest.fixture(scope="module")
def compose_text() -> str:
    if not COMPOSE_FILE.exists():
        pytest.skip("Compose file not present")
    return COMPOSE_FILE.read_text(encoding="utf-8")


def test_compose_defines_expected_images_and_ports(compose_text: str) -> None:
    assert "image: neo4j:5.26.0" in compose_text
    assert "image: qdrant/qdrant:1.9.4" in compose_text
    assert "7474:7474" in compose_text
    assert "7687:7687" in compose_text
    assert "6333:6333" in compose_text


def test_compose_env_variables_present(compose_text: str) -> None:
    assert "NEO4J_AUTH" in compose_text
    assert "NEO4JLABS_PLUGINS" in compose_text
    assert "QDRANT__SERVICE__HTTP_PORT" in compose_text
    assert "source: ./.data/neo4j/data" in compose_text
    assert "source: ./.data/qdrant/storage" in compose_text


def test_compose_healthchecks_defined(compose_text: str) -> None:
    assert "healthcheck:" in compose_text
    assert "cypher-shell" in compose_text
    assert "readyz" in compose_text


def test_compose_restart_policy(compose_text: str) -> None:
    assert "restart: unless-stopped" in compose_text
