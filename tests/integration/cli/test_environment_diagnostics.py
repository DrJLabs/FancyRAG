import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"


def _create_stub_modules(dest: Path) -> None:
    (dest / "neo4j_graphrag.py").write_text("__version__ = '0.9.0'\n", encoding="utf-8")
    (dest / "qdrant_client.py").write_text("__version__ = '1.10.4'\n", encoding="utf-8")
    (dest / "neo4j").mkdir()
    (dest / "neo4j" / "__init__.py").write_text("__version__ = '5.23.0'\n", encoding="utf-8")
    (dest / "openai").mkdir()
    (dest / "openai" / "__init__.py").write_text("__version__ = '1.40.0'\n", encoding="utf-8")
    (dest / "structlog").mkdir()
    (dest / "structlog" / "__init__.py").write_text("__version__ = '24.1.0'\n", encoding="utf-8")


def _initialise_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "src", repo / "src")
    (repo / "requirements.lock").write_text("neo4j-graphrag==0.9.0\n", encoding="utf-8")
    (repo / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    stubs = repo / "stubs"
    stubs.mkdir(parents=True)
    _create_stub_modules(stubs)

    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=repo, check=True)
    subprocess.run(["git", "add", "src", "requirements.lock", "stubs"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-f", ".gitignore"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return repo


@pytest.fixture()
def repo(tmp_path):
    return _initialise_repo(tmp_path)


def _run_cli(repo: Path, extra_env: dict[str, str] | None = None, arguments: list[str] | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "src"), str(repo / "stubs"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if extra_env:
        env.update(extra_env)
    args = [sys.executable, "-m", "cli.diagnostics", "workspace", "--root", str(repo)]
    if arguments:
        args.extend(arguments)
    return subprocess.run(args, cwd=repo, capture_output=True, text=True, env=env)


def test_workspace_diagnostics_writes_report(repo):
    result = _run_cli(repo, arguments=["--output", "artifacts/environment/versions.json"])
    assert result.returncode == 0, result.stderr
    report_path = repo / "artifacts/environment/versions.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["lockfile"]["exists"] is True
    assert len(data["lockfile"]["sha256"]) == 64
    assert any(pkg["module"] == "neo4j_graphrag" for pkg in data["packages"])
    assert data["git"]["sha"]


def test_workspace_diagnostics_redacts_secret(repo):
    result = _run_cli(
        repo,
        extra_env={"OPENAI_API_KEY": "sk-test-999"},
        arguments=["--output", "artifacts/environment/versions.json"],
    )
    assert result.returncode == 0, result.stderr
    assert "sk-test-999" not in result.stdout
    assert "sk-test-999" not in result.stderr


def test_workspace_diagnostics_fails_on_missing_dependency(repo):
    missing_repo = repo / "stubs" / "neo4j_graphrag.py"
    missing_repo.unlink()
    result = _run_cli(repo, arguments=["--no-report"])
    assert result.returncode == 1
    assert "neo4j_graphrag" in result.stderr


def test_report_artifacts_remain_untracked(repo):
    result = _run_cli(repo, arguments=["--output", "artifacts/environment/versions.json"])
    assert result.returncode == 0, result.stderr
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    lines = [line.strip() for line in status.splitlines() if line.strip()]
    assert all("artifacts/" not in line for line in lines)


def test_overview_documents_diagnostics_reference():
    overview = (REPO_ROOT / "docs" / "architecture" / "overview.md").read_text(encoding="utf-8")
    assert "python -m cli.diagnostics" in overview
    assert "artifacts/environment/versions.json" in overview
