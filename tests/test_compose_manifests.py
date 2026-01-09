"""Static validations for Compose manifests and Docker ignore rules."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"
SMOKE_COMPOSE_PATH = PROJECT_ROOT / "docker-compose.smoke.yml"
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_mcp_service_configuration() -> None:
    compose = _load_yaml(COMPOSE_PATH)

    services = compose.get("services", {})
    assert "mcp" in services, "docker-compose.yml must define an mcp service"

    service = services["mcp"]

    build = service.get("build", {})
    assert build.get("context") == ".", "mcp build context should be repository root"

    depends_on = service.get("depends_on", {})
    assert "neo4j" in depends_on, "mcp must depend on neo4j"
    neo4j_dependency = depends_on["neo4j"]
    assert (
        isinstance(neo4j_dependency, dict)
        and neo4j_dependency.get("condition") == "service_healthy"
    ), "mcp must wait for neo4j health"

    networks = service.get("networks", [])
    assert "rag-net" in networks, "mcp must join rag-net network"

    ports = service.get("ports", [])
    assert any("8080" in str(port) for port in ports), "mcp must expose port 8080"

    healthcheck = service.get("healthcheck", {})
    test_cmd = healthcheck.get("test", [])
    assert any("/mcp/health" in str(token) for token in test_cmd), (
        "mcp healthcheck should probe /mcp/health"
    )


def test_smoke_manifest_stays_in_sync() -> None:
    compose_service = _load_yaml(COMPOSE_PATH)["services"]["mcp"]
    smoke_service = _load_yaml(SMOKE_COMPOSE_PATH)["services"]["mcp"]

    assert (
        smoke_service.get("build", {}).get("context")
        == compose_service.get("build", {}).get("context")
    ), "Smoke manifest must reuse the same build context as production compose"

    assert smoke_service.get("image") == compose_service.get("image"), (
        "Smoke manifest must reuse the same image tag"
    )

    assert smoke_service.get("env_file") == compose_service.get("env_file"), (
        "Smoke manifest must load identical env_file configuration"
    )


def test_dockerignore_excludes_env_files() -> None:
    ignore_entries = {
        line.strip()
        for line in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert ".env" in ignore_entries, ".env must be excluded from Docker build context"
    assert ".env.*" in ignore_entries, ".env.* must be excluded from Docker build context"
    assert "!.env.example" in ignore_entries, (
        "Template environment file should remain accessible for documentation"
    )
