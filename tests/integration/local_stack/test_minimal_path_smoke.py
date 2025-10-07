from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
import json

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

    try:
        run_command("make", "service-run", env=env)

        service_root = PROJECT_ROOT / "artifacts" / "local_stack" / "service"
        summaries = sorted(service_root.glob("*/service_run.json"))
        assert summaries, "service_run summary missing"
        latest_summary = summaries[-1]
        summary_data = json.loads(latest_summary.read_text(encoding="utf-8"))
        assert summary_data["status"] == "success"
        assert summary_data["stages"], "expected recorded stages"

        artifacts = summary_data.get("artifacts", {})
        kg_log_rel = artifacts.get("kg_log")
        assert kg_log_rel, "kg_log artifact missing"
        kg_log_path = (PROJECT_ROOT / kg_log_rel).resolve()
        assert kg_log_path.exists(), f"kg_build log missing at {kg_log_path}"

        log_data = json.loads(kg_log_path.read_text(encoding="utf-8"))
        assert log_data["status"] == "success"
        assert log_data.get("run_ids"), "expected at least one run id"
        assert log_data.get("files"), "log should include ingested files"

        qa_section = log_data.get("qa")
        assert qa_section and qa_section["status"] == "pass"
        for key in ("report_json", "report_markdown"):
            report_path = Path(qa_section[key])
            if not report_path.is_absolute():
                candidate = PROJECT_ROOT / report_path
                if not candidate.exists():
                    candidate = kg_log_path.parent / report_path
                report_path = candidate
            assert report_path.exists(), f"{key} report missing at {report_path}"

        vector_log_rel = artifacts.get("vector_index_log")
        assert vector_log_rel, "vector index log missing"
        assert (PROJECT_ROOT / vector_log_rel).exists()

        export_log_rel = artifacts.get("export_log")
        assert export_log_rel, "export log missing"
        assert (PROJECT_ROOT / export_log_rel).exists()

        docs_check_rel = artifacts.get("docs_check")
        assert docs_check_rel, "documentation check log missing"
        assert (PROJECT_ROOT / docs_check_rel).exists()

        qa_dir_rel = artifacts.get("qa_dir")
        assert qa_dir_rel, "qa_dir artifact missing"
        qa_dir_path = (PROJECT_ROOT / qa_dir_rel).resolve()
        assert qa_dir_path.exists() and qa_dir_path.is_dir()

    finally:
        run_command("make", "service-reset", env=env, check=False)
