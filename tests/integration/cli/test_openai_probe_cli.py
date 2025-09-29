import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "openai_probe"


def _create_stub_openai(dest: Path) -> None:
    """
    Create an on-disk stub of the `openai` package at the given destination.
    
    The stub provides minimal classes and objects that mimic the public surface used by the code under test: exception types (APIError, APIConnectionError, APIStatusError, RateLimitError), lightweight usage/result containers, a chat completions helper that returns a single "stop" choice and usage counts, an embeddings helper that returns a 1536-length embedding vector, and an OpenAI class that exposes `chat.completions`, `embeddings`, and a `responses` list.
    
    Parameters:
        dest (Path): Directory in which an `openai` package directory (with `__init__.py`) will be created.
    """
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
    """
    Create a temporary Git repository populated with the project source, stub OpenAI package, and basic project files.
    
    Parameters:
        tmp_path (Path): Base temporary directory where the repository directory named "repo" will be created.
    
    Returns:
        repo (Path): Path to the created repository directory containing "src", "stubs", "requirements.lock", and ".gitignore".
    """
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
    """
    Run the CLI openai-probe against a repository scaffold and return the completed process result.
    
    Parameters:
        repo (Path): Path to the repository to run the probe in; its "src" and "stubs" directories are added to PYTHONPATH.
        extra_env (dict[str, str] | None): Additional environment variables to set or override for the probe process.
        arguments (list[str] | None): Extra command-line arguments to append to the probe invocation.
    
    Returns:
        subprocess.CompletedProcess: The completed process result for the probe run, containing `returncode`, `stdout`, and `stderr`.
    """
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
    _assert_matches_fixture(report_path, metrics_path)
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    assert all("artifacts/" not in line for line in git_status.splitlines())


def _assert_matches_fixture(report_path: Path, metrics_path: Path) -> None:
    expected_report = json.loads((FIXTURE_ROOT / "probe.json").read_text(encoding="utf-8"))
    normalised_report = _normalise_report(report_path)
    assert normalised_report == expected_report

    expected_metrics = (FIXTURE_ROOT / "metrics.prom").read_text(encoding="utf-8").strip()
    normalised_metrics = _normalise_metrics(metrics_path.read_text(encoding="utf-8")).strip()
    assert normalised_metrics == expected_metrics


def _normalise_report(report_path: Path) -> Dict[str, Any]:
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    def _latency(value: float) -> float:
        if value is None:
            return 0.0
        return 0.0 if value < 0.05 else round(value, 2)

    return {
        "actor": raw.get("actor"),
        "artifacts": {
            "metrics": "artifacts/openai/metrics.prom",
            "report": "artifacts/openai/probe.json",
        },
        "chat": {
            "completion_tokens": raw["chat"].get("completion_tokens", 0),
            "fallback_used": raw["chat"].get("fallback_used", False),
            "finish_reason": raw["chat"].get("finish_reason"),
            "latency_ms": _latency(raw["chat"].get("latency_ms", 0.0)),
            "model": raw["chat"].get("model"),
            "prompt_tokens": raw["chat"].get("prompt_tokens", 0),
            "status": raw["chat"].get("status"),
        },
        "embedding": {
            "expected_dimensions": raw["embedding"].get("expected_dimensions", 0),
            "latency_ms": _latency(raw["embedding"].get("latency_ms", 0.0)),
            "model": raw["embedding"].get("model"),
            "status": raw["embedding"].get("status"),
            "tokens_consumed": raw["embedding"].get("tokens_consumed", 0),
            "vector_length": raw["embedding"].get("vector_length", 0),
        },
        "generated_at": "<timestamp>",
        "settings": {
            "backoff_seconds": raw["settings"].get("backoff_seconds"),
            "chat_model": raw["settings"].get("chat_model"),
            "chat_override": raw["settings"].get("chat_override"),
            "embedding_dimensions": raw["settings"].get("embedding_dimensions"),
            "embedding_model": raw["settings"].get("embedding_model"),
            "fallback_enabled": raw["settings"].get("fallback_enabled"),
            "max_attempts": raw["settings"].get("max_attempts"),
        },
        "status": raw.get("status"),
    }


def _normalise_metrics(metrics_text: str) -> str:
    normalised_lines: list[str] = []
    for line in metrics_text.strip().splitlines():
        if line.startswith("#"):
            normalised_lines.append(line)
            continue
        metric, value = line.split(" ", 1)
        metric_name, brace, labels = metric.partition("{")
        suffix = f"{brace}{labels}" if brace else ""
        try:
            numeric = float(value)
        except ValueError:
            normalised_lines.append(line)
            continue
        if metric_name.endswith("_created"):
            normalised_lines.append(f"{metric_name}{suffix} <created>")
            continue
        if abs(numeric - round(numeric)) < 1e-9:
            normalised_lines.append(f"{metric_name}{suffix} {int(round(numeric))}")
            continue
        adjusted = 0.0 if abs(numeric) < 0.05 else round(numeric, 2)
        normalised_lines.append(f"{metric_name}{suffix} {adjusted:.2f}")
    return "\n".join(normalised_lines) + "\n"
