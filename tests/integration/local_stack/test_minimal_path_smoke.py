from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fancyrag.utils.env import ensure_env, load_project_dotenv

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


TRUTHY_VALUES = {"1", "true", "yes", "on"}
STACK_SERVICE_NAMES = ("neo4j-graphrag", "qdrant-graphrag")


def _is_truthy(value: str | None) -> bool:
    """Return True when the provided string represents an affirmative value."""

    return str(value or "").strip().lower() in TRUTHY_VALUES


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


def _docker_containers_running(names: tuple[str, ...]) -> bool:
    """Return True when all named containers appear in `docker ps`."""

    docker_path = shutil.which("docker")
    if docker_path is None:
        return False
    result = subprocess.run(
        [docker_path, "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    running = set(line.strip() for line in result.stdout.splitlines() if line.strip())
    return all(name in running for name in names)


STACK_ALREADY_RUNNING = _docker_containers_running(STACK_SERVICE_NAMES)

SKIP_FOR_DOCKER = (
    shutil.which("docker") is None
    and not _is_truthy(os.environ.get("LOCAL_STACK_SKIP_DOCKER_CHECK"))
    and not STACK_ALREADY_RUNNING
)


@pytest.mark.integration
@pytest.mark.skipif(SKIP_FOR_DOCKER, reason="docker command not available")
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

    # Load .env values before copying `os.environ` so stack credentials are available.
    load_project_dotenv()

    env = os.environ.copy()
    env["COMPOSE_FILE"] = str(COMPOSE_FILE)
    env["PYTHONPATH"] = "stubs:src"

    neo4j_host = os.environ.get("NEO4J_HOST", "localhost")
    neo4j_http_host = os.environ.get("NEO4J_HTTP_HOST", neo4j_host)
    neo4j_bolt_port = os.environ.get("NEO4J_BOLT_PORT", "7687")
    neo4j_http_port = os.environ.get("NEO4J_HTTP_PORT", "7474")

    env["NEO4J_HOST"] = neo4j_host
    env["NEO4J_HTTP_HOST"] = neo4j_http_host
    env["NEO4J_BOLT_PORT"] = neo4j_bolt_port
    env["NEO4J_HTTP_PORT"] = neo4j_http_port

    env["NEO4J_USERNAME"] = os.environ.get("NEO4J_USERNAME", "neo4j")
    env["NEO4J_PASSWORD"] = os.environ.get("NEO4J_PASSWORD", "local-neo4j")
    env["NEO4J_AUTH"] = f"{env['NEO4J_USERNAME']}/{env['NEO4J_PASSWORD']}"
    env["NEO4J_URI"] = os.environ.get("NEO4J_URI", f"bolt://{neo4j_host}:{neo4j_bolt_port}")
    env["NEO4J_BOLT_ADVERTISED_ADDRESS"] = os.environ.get(
        "NEO4J_BOLT_ADVERTISED_ADDRESS", f"{neo4j_host}:{neo4j_bolt_port}"
    )
    env["NEO4J_HTTP_ADVERTISED_ADDRESS"] = os.environ.get(
        "NEO4J_HTTP_ADVERTISED_ADDRESS", f"{neo4j_http_host}:{neo4j_http_port}"
    )

    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_http_port = os.environ.get("QDRANT_HTTP_PORT", "6333")
    qdrant_grpc_port = os.environ.get("QDRANT_GRPC_PORT", "6334")

    env["QDRANT_HOST"] = qdrant_host
    env["QDRANT_HTTP_PORT"] = qdrant_http_port
    env["QDRANT_GRPC_PORT"] = qdrant_grpc_port
    env["QDRANT_URL"] = os.environ.get("QDRANT_URL", f"http://{qdrant_host}:{qdrant_http_port}")
    env["QDRANT_API_KEY"] = os.environ.get("QDRANT_API_KEY", "")

    try:
        api_key = ensure_env("OPENAI_API_KEY")
    except SystemExit:
        pytest.skip("OPENAI_API_KEY environment variable is required for minimal path smoke test")
    env["OPENAI_API_KEY"] = api_key

    # Ensure bind-mount directories exist before starting the stack.
    for relative in (".data/neo4j/data", ".data/neo4j/logs", ".data/neo4j/import", ".data/qdrant/storage"):
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)

    skip_docker_ops = _is_truthy(os.environ.get("LOCAL_STACK_SKIP_DOCKER_CHECK"))
    if not skip_docker_ops and _docker_containers_running(STACK_SERVICE_NAMES):
        skip_docker_ops = True

    if not skip_docker_ops:
        run_command(str(CHECK_SCRIPT), "--config", env=env)

    stack_started = False
    try:
        if not skip_docker_ops:
            up_result = run_command(str(CHECK_SCRIPT), "--up", env=env, check=False)
            if up_result.returncode != 0:
                pytest.skip(f"docker compose up failed: {up_result.stdout.strip()}")
            stack_started = True
            run_command(str(CHECK_SCRIPT), "--status", "--wait", env=env)
        else:
            # Assume external orchestrator started the stack when docker CLI is unavailable.
            stack_started = True

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
        run_command(
            python,
            "scripts/kg_build.py",
            "--source",
            "docs/samples/pilot.txt",
            "--reset-database",
            env=env,
        )
        run_command(
            python,
            "scripts/export_to_qdrant.py",
            "--collection",
            "chunks_main",
            "--recreate-collection",
            env=env,
        )
        run_command(
            python,
            "scripts/ask_qdrant.py",
            "--question",
            "What did Acme launch?",
            "--top-k",
            "3",
            env=env,
        )

        run_command(
            python,
            "-m",
            "scripts.check_docs",
            "--json-output",
            str(PROJECT_ROOT / "artifacts" / "docs" / "check_docs_smoke.json"),
            env=env,
        )

    finally:
        if stack_started and not skip_docker_ops:
            run_command(
                str(CHECK_SCRIPT), "--down", "--destroy-volumes", env=env, check=False
            )
