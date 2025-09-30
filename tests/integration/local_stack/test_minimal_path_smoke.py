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
    """
    Run a subprocess command with stdout/stderr captured and optionally assert success.
    
    Parameters:
        *args (str): Command and its arguments as separate strings (executable followed by args).
        env (dict[str, str]): Environment mapping to use for the subprocess.
        check (bool): If true, raise AssertionError when the command exits with a non-zero status.
    
    Returns:
        subprocess.CompletedProcess[str]: The completed process object; `stdout` contains combined stdout and stderr.
    
    Raises:
        AssertionError: If `check` is true and the subprocess exit code is non-zero. 
    """
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
def test_minimal_path_smoke() -> None:
    """
    Integration test that boots a local Docker stack, runs a minimal end-to-end workflow, and tears the stack down.
    
    Skips if required Docker compose file, helper script, or minimal-path scripts are missing. When executed, the test brings up the local stack, waits for readiness, runs the minimal sequence of scripts to create a vector index, build the knowledge graph, export data to Qdrant, and query Qdrant, then tears the stack down and destroys volumes.
    """
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
    env["COMPOSE_FILE"] = str(COMPOSE_FILE)
    env["PYTHONPATH"] = "stubs:src"
    env["NEO4J_USERNAME"] = os.environ.get("NEO4J_USERNAME", "neo4j")
    env["NEO4J_PASSWORD"] = os.environ.get("NEO4J_PASSWORD", "neo4j")
    env["NEO4J_AUTH"] = f"{env['NEO4J_USERNAME']}/{env['NEO4J_PASSWORD']}"
    env["NEO4J_URI"] = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    env["NEO4J_BOLT_ADVERTISED_ADDRESS"] = os.environ.get("NEO4J_BOLT_ADVERTISED_ADDRESS", "localhost:7687")
    env["NEO4J_HTTP_ADVERTISED_ADDRESS"] = os.environ.get("NEO4J_HTTP_ADVERTISED_ADDRESS", "localhost:7474")
    env["QDRANT_URL"] = os.environ.get("QDRANT_URL", "http://localhost:6333")
    env["QDRANT_API_KEY"] = os.environ.get("QDRANT_API_KEY", "")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable is required for minimal path smoke test")
    env["OPENAI_API_KEY"] = api_key

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
        status_result = run_command(str(CHECK_SCRIPT), "--status", "--wait", env=env, check=False)
        if status_result.returncode != 0:
            logs = run_command(
                "docker",
                "compose",
                "-f",
                env["COMPOSE_FILE"],
                "logs",
                "neo4j",
                env=env,
                check=False,
            )
            pytest.fail(
                "Stack status check failed:\n"
                f"{status_result.stdout}\n"
                "--- neo4j logs ---\n"
                f"{logs.stdout}"
            )

        # Execute minimal path scripts sequentially.
        python = sys.executable
        run_command(
            python,
            "scripts/create_vector_index.py",
            "--index-name",
            "chunks_vec",
            "--label",
            "Chunk",
            "--embedding-property",
            "embedding",
            "--dimensions",
            "1536",
            "--similarity",
            "cosine",
            env=env,
        )
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
