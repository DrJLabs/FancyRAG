from __future__ import annotations

from pathlib import Path

import pytest

COMPOSE_FILE = Path(__file__).resolve().parents[3] / "docker-compose.neo4j-qdrant.yml"


@pytest.fixture(scope="module")
def compose_text() -> str:
    """
    Provide the text contents of the project's docker-compose.neo4j-qdrant.yml used by tests.
    
    If the compose file is not present, the test module is skipped via pytest.skip. 
    
    Returns:
        The compose file contents as a UTF-8 decoded string.
    """
    if not COMPOSE_FILE.exists():
        pytest.skip("Compose file not present")
    return COMPOSE_FILE.read_text(encoding="utf-8")


def test_compose_defines_expected_images_and_ports(compose_text: str) -> None:
    """
    Verifies the compose file declares the expected container images and port mappings.
    
    Checks that the compose text contains the image references `neo4j:5.26.0` and `qdrant/qdrant:1.9.4`, and the port mappings `7474:7474`, `7687:7687`, and `6333:6333`.
    
    Parameters:
        compose_text (str): The full contents of the docker-compose file as a UTF-8 string.
    """
    assert "image: neo4j:5.26.0" in compose_text
    assert "image: qdrant/qdrant:1.9.4" in compose_text
    assert "7474:7474" in compose_text
    assert "7687:7687" in compose_text
    assert "6333:6333" in compose_text


def test_compose_env_variables_present(compose_text: str) -> None:
    """
    Check that the Docker Compose file text contains required environment variables, volume sources, and service port configuration keys.
    
    Parameters:
    	compose_text (str): Full contents of the compose file as a single string; used to assert presence of expected keys such as `NEO4J_AUTH`, `NEO4JLABS_PLUGINS`, `QDRANT__SERVICE__HTTP_PORT`, and the volume sources for Neo4j and Qdrant.
    """
    assert "NEO4J_AUTH" in compose_text
    assert "NEO4JLABS_PLUGINS" in compose_text
    assert "QDRANT__SERVICE__HTTP_PORT" in compose_text
    assert "source: ./.data/neo4j/data" in compose_text
    assert "source: ./.data/qdrant/storage" in compose_text


def test_compose_healthchecks_defined(compose_text: str) -> None:
    """
    Verify the Docker Compose content defines healthchecks and includes the expected health probe commands.
    
    Parameters:
    	compose_text (str): The full contents of the compose file to inspect; the test asserts it contains "healthcheck:", "cypher-shell", and "readyz".
    """
    assert "healthcheck:" in compose_text
    assert "cypher-shell" in compose_text
    assert "readyz" in compose_text


def test_compose_restart_policy(compose_text: str) -> None:
    assert "restart: unless-stopped" in compose_text
