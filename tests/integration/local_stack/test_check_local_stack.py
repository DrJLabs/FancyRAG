from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECK_SCRIPT = PROJECT_ROOT / "scripts" / "check_local_stack.sh"
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker command not available")
def test_check_local_stack_config_renders_successfully() -> None:
    """
    Verify that the local stack check script renders the Docker Compose configuration.

    Runs the check script with the base compose file and asserts the process exits with code 0 and that the rendered output contains the strings "neo4j:" and "mcp:".
    """
    env = os.environ.copy()
    env["COMPOSE_FILE"] = str(COMPOSE_FILE)
    env.pop("MCP_ENV_FILE", None)

    env_path = PROJECT_ROOT / ".env"
    env_local_path = PROJECT_ROOT / ".env.local"
    original_env = env_path.read_text(encoding="utf-8") if env_path.exists() else None
    original_env_local = (
        env_local_path.read_text(encoding="utf-8") if env_local_path.exists() else None
    )

    try:
        if env_local_path.exists():
            env_local_path.unlink()
        env_path.write_text("NEO4J_PASSWORD=local-neo4j\n", encoding="utf-8")

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
        assert "mcp:" in result.stdout
    finally:
        if original_env is None:
            env_path.unlink(missing_ok=True)
        else:
            env_path.write_text(original_env, encoding="utf-8")

        if original_env_local is None:
            env_local_path.unlink(missing_ok=True)
        else:
            env_local_path.write_text(original_env_local, encoding="utf-8")
