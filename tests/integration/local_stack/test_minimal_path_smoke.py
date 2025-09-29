from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.neo4j-qdrant.yml"
CHECK_SCRIPT = PROJECT_ROOT / "scripts" / "check_local_stack.sh"

REQUIRED_SCRIPTS = [
    PROJECT_ROOT / "scripts" / name
    for name in (
        "create_vector_index.py",
        "kg_build.py",
        "export_to_qdrant.py",
        "ask_qdrant.py",
    )
]


def run_command(*args: str, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"Command {' '.join(args)} failed:\n{result.stdout}")
    return result


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("docker") is None, reason="docker command not available")
def test_minimal_path_smoke(tmp_path: Path) -> None:
    if not COMPOSE_FILE.exists():
        pytest.skip("Compose file missing; ensure Story 2.4 assets are generated.")
    if not CHECK_SCRIPT.exists():
        pytest.skip("check_local_stack.sh missing; ensure Story 2.4 assets are generated.")

    missing_scripts = [path.name for path in REQUIRED_SCRIPTS if not path.exists()]
    if missing_scripts:
        pytest.skip(
            "Minimal-path scripts not available (Story 2.5): " + ", ".join(sorted(missing_scripts))
        )

    env = os.environ.copy()
    env.setdefault("COMPOSE_FILE", str(COMPOSE_FILE))
    env.setdefault("PYTHONPATH", "src")
    env.setdefault("NEO4J_USERNAME", "neo4j")
    env.setdefault("NEO4J_PASSWORD", "neo4j")
    env.setdefault("NEO4J_URI", "bolt://localhost:7687")
    env.setdefault("NEO4J_BOLT_ADVERTISED_ADDRESS", "localhost:7687")
    env.setdefault("NEO4J_HTTP_ADVERTISED_ADDRESS", "localhost:7474")
    env.setdefault("QDRANT_URL", "http://localhost:6333")
    env.setdefault("QDRANT_API_KEY", "")

    # Ensure bind-mount directories exist before starting the stack.
    for relative in (".data/neo4j/data", ".data/neo4j/logs", ".data/neo4j/import", ".data/qdrant/storage"):
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)

    run_command(str(CHECK_SCRIPT), "--config", env=env)

    stack_started = False
    try:
        up_result = run_command(str(CHECK_SCRIPT), "--up", env=env, check=False)
        if up_result.returncode != 0:
            pytest.skip(f"docker compose up failed: {up_result.stdout.strip()}")
        stack_started = True
        run_command(str(CHECK_SCRIPT), "--status", "--wait", env=env)

        # Execute minimal path scripts sequentially.
        python = sys.executable
        run_command(python, "scripts/create_vector_index.py", "--dimensions", "1536", "--name", "chunks_vec", env=env)
        run_command(python, "scripts/kg_build.py", "--source", "docs/samples/pilot.txt", env=env)
        run_command(python, "scripts/export_to_qdrant.py", "--collection", "chunks_main", env=env)
        run_command(
            python,
            "scripts/ask_qdrant.py",
            "--question",
            "What did Acme launch?",
            "--top-k",
            "3",
            env=env,
        )

    finally:
        if stack_started:
            run_command(str(CHECK_SCRIPT), "--down", "--destroy-volumes", env=env, check=False)
