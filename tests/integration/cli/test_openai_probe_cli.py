import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"


def _create_stub_openai(dest: Path) -> None:
    package = dest / "openai"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """
class APIError(Exception):
    pass


class APIConnectionError(APIError):
    pass


class APIStatusError(APIError):
    def __init__(self, message: str, *, status_code: int | None = None, response=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response
        self.body = body


class RateLimitError(APIStatusError):
    def __init__(self, message: str, *, status_code: int = 429, response=None, body=None):
        super().__init__(message, status_code=status_code, response=response, body=body)


class _Usage:
    def __init__(self, **values):
        self.__dict__.update(values)


class _ChatCompletions:
    def create(self, **_):
        return type(
            "ChatResponse",
            (),
            {
                "usage": _Usage(prompt_tokens=9, completion_tokens=3),
                "choices": [type("Choice", (), {"finish_reason": "stop"})()],
            },
        )()


class _Embeddings:
    def create(self, **_):
        return type(
            "EmbedResponse",
            (),
            {
                "usage": _Usage(total_tokens=7),
                "data": [type("Embed", (), {"embedding": [0.0] * 1536})()],
            },
        )()


class OpenAI:
    def __init__(self, *_, **__):
        self.chat = type("Chat", (), {"completions": _ChatCompletions()})()
        self.embeddings = _Embeddings()
        self.responses = []

""",
        encoding="utf-8",
    )


def _initialise_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(SRC_PATH, repo / "src")
    (repo / "requirements.lock").write_text("neo4j-graphrag==0.9.0\n", encoding="utf-8")
    (repo / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    stubs = repo / "stubs"
    stubs.mkdir(parents=True)
    _create_stub_openai(stubs)

    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=repo, check=True)
    subprocess.run(["git", "add", "src", "requirements.lock", "stubs"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-f", ".gitignore"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return repo


def _run_probe(repo: Path, extra_env: dict[str, str] | None = None, arguments: list[str] | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "src"), str(repo / "stubs"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env.setdefault("GRAPH_RAG_ACTOR", "integration-test")
    if extra_env:
        env.update(extra_env)
    args = [
        sys.executable,
        "-m",
        "cli.diagnostics",
        "openai-probe",
        "--root",
        str(repo),
        "--artifacts-dir",
        "artifacts/openai",
    ]
    if arguments:
        args.extend(arguments)
    return subprocess.run(args, cwd=repo, capture_output=True, text=True, env=env)


def test_openai_probe_cli_generates_artifacts(tmp_path):
    repo = _initialise_repo(tmp_path)
    extra_env = {"OPENAI_MODEL": "gpt-4o-mini"}
    result = _run_probe(repo, extra_env=extra_env)

    assert result.returncode == 0, result.stderr
    report_path = repo / "artifacts" / "openai" / "probe.json"
    metrics_path = repo / "artifacts" / "openai" / "metrics.prom"

    assert report_path.exists()
    assert metrics_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["settings"]["chat_override"] is True
    assert report["chat"]["fallback_used"] is True
    assert report["embedding"]["vector_length"] == 1536
    assert "sk-" not in report_path.read_text(encoding="utf-8")

    metrics = metrics_path.read_text(encoding="utf-8")
    assert "graphrag_openai_chat_latency_ms_bucket" in metrics
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    assert all("artifacts/" not in line for line in git_status.splitlines())
