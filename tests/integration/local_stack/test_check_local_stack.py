from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECK_SCRIPT = PROJECT_ROOT / "scripts" / "check_local_stack.sh"
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.neo4j-qdrant.yml"


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker command not available")
def test_check_local_stack_config_renders_successfully() -> None:
    """
    Verify that the local stack check script renders the Docker Compose configuration including Neo4j and Qdrant.
    
    Runs the check script with the test compose file and asserts the process exits with code 0 and that the rendered output contains the strings "neo4j:" and "qdrant:".
    """
    env = os.environ.copy()
    env["COMPOSE_FILE"] = str(COMPOSE_FILE)

    result = subprocess.run(
        [str(CHECK_SCRIPT), "--config"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "neo4j:" in result.stdout
    assert "qdrant:" in result.stdout
