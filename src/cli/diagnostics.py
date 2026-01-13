"""Workspace diagnostics utilities for GraphRAG CLI environments."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from cli.openai_client import (
    ChatResult,
    EmbeddingResult,
    OpenAIClientError,
    SharedOpenAIClient,
)
from cli.sanitizer import mask_base_url, sanitize_text, scrub_object
from cli.telemetry import create_metrics
from config.settings import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_RETRY_ATTEMPTS,
    OpenAISettings,
)

MODULES: List[tuple[str, str]] = [
    ("neo4j_graphrag", "neo4j-graphrag"),
    ("neo4j", "neo4j"),
    ("qdrant_client", "qdrant-client"),
    ("openai", "openai"),
    ("structlog", "structlog"),
    ("pytest", "pytest"),
]

DEFAULT_REPORT_PATH = Path("artifacts") / "environment" / "versions.json"
DEFAULT_PROBE_REPORT_PATH = Path("artifacts") / "openai" / "probe.json"
DEFAULT_PROBE_METRICS_PATH = Path("artifacts") / "openai" / "metrics.prom"


class DependencyError(RuntimeError):
    """Raised when a required module cannot be imported."""


class ProbeFailure(RuntimeError):
    """Operational failure of the OpenAI readiness probe with remediation guidance."""

    def __init__(self, message: str, *, remediation: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a ProbeFailure with a user-facing message, remediation guidance, and optional details.
        
        Parameters:
            message (str): Short description of the failure.
            remediation (str): Actionable guidance to remediate the failure.
            details (Optional[Dict[str, Any]]): Additional structured information about the failure; defaults to an empty dict.
        """
        super().__init__(message)
        self.remediation = remediation
        self.details = details or {}


@dataclass
class PackageInfo:
    module: str
    distribution: str
    version: str
    version_source: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "module": self.module,
            "distribution": self.distribution,
            "version": self.version,
            "version_source": self.version_source,
        }


def _compute_repo_root(explicit: Optional[Path] = None) -> Path:
    """
    Determine the repository root directory, using an explicit path or by locating repository markers near this file.
    
    Parameters:
        explicit (Optional[Path]): If provided, this path is returned after resolving to an absolute path. If omitted, the function searches upward from this file for a directory containing a `.git` folder or a `requirements.lock` file and returns the first match; if none is found, the current working directory is returned.
    
    Returns:
        Path: Resolved absolute path of the repository root directory.
    """
    if explicit is not None:
        return explicit.resolve()

    # Attempt to locate the repository root by walking up from this file
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / ".git").exists() or (parent / "requirements.lock").exists():
            return parent
    return Path.cwd().resolve()


def _print(level: str, message: str, *, file = sys.stdout) -> None:
    """
    Prints a sanitized, level-prefixed log line to the given file-like object.
    
    Parameters:
    	level (str): A short level label (e.g., "INFO", "ERROR") to prefix the message.
    	message (str): The text message to print; it will be sanitized before output.
    	file: A file-like object with a write() method to receive the output (defaults to sys.stdout).
    """
    text = sanitize_text(message)
    print(f"[{level}] {text}", file=file)


def _load_requirements_versions(lock_path: Path) -> Dict[str, str]:
    if not lock_path.exists():
        return {}
    versions: Dict[str, str] = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        versions[name.lower()] = version
    return versions


def _distribution_name(_module: str, declared: str) -> str:
    # Some distributions use underscores internally; normalise to canonical form
    return declared


def _version_from_metadata(
    distribution: str, fallback_module: Any
) -> tuple[Optional[str], str]:
    try:
        return metadata.version(distribution), "metadata"
    except metadata.PackageNotFoundError:
        module_version = getattr(fallback_module, "__version__", None)
        if module_version:
            return str(module_version), "module"
        return None, "unknown"


def _collect_packages(requirements: Dict[str, str]) -> List[PackageInfo]:
    packages: List[PackageInfo] = []
    for module_name, distribution_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            raise DependencyError(
                f"Missing or broken dependency: unable to import '{module_name}'. {exc}"
            ) from exc

        distribution = _distribution_name(module_name, distribution_name)
        version, source = _version_from_metadata(distribution, module)
        if version is None:
            req_version = requirements.get(distribution.lower()) or requirements.get(
                distribution.replace("-", "_").lower()
            )
            if req_version:
                version = req_version
                source = "requirements.lock"
            else:
                version = "unknown"
        packages.append(PackageInfo(module=module_name, distribution=distribution, version=version, version_source=source))
    return packages


def _hash_lockfile(lock_path: Path) -> Dict[str, Optional[str]]:
    if not lock_path.exists():
        return {"path": str(lock_path), "exists": False, "sha256": None}
    data = lock_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    return {"path": str(lock_path), "exists": True, "sha256": digest}


def _git_metadata(root: Path) -> Dict[str, Optional[str]]:
    git_dir = root / ".git"
    if not git_dir.exists():
        return {"sha": None}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        return {"sha": sha}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None}


def build_report(root: Path) -> Dict[str, object]:
    lock_path = root / "requirements.lock"
    requirements = _load_requirements_versions(lock_path)
    packages = _collect_packages(requirements)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(root),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": [pkg.to_dict() for pkg in packages],
        "lockfile": _hash_lockfile(lock_path),
        "git": _git_metadata(root),
    }


def write_report(report: Dict[str, object], path: Path) -> None:
    """
    Write a sanitized diagnostic report to disk as pretty-printed JSON.
    
    Ensures the parent directory of `path` exists, sanitizes the provided `report` to remove or redact sensitive data, and writes the result as UTF-8 encoded JSON with 2-space indentation, sorted object keys, and a trailing newline.
    
    Parameters:
        report (Dict[str, object]): The diagnostic report mapping to serialize.
        path (Path): Destination filesystem path where the JSON report will be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = scrub_object(report)
    path.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def run_workspace(root: Path, *, write: bool, output: Optional[Path]) -> int:
    """
    Run workspace dependency diagnostics, print findings, and optionally write a JSON report.
    
    Prints detected package versions, lockfile checksum, and git commit information to stdout/stderr.
    If a required package is missing, prints an error with remediation guidance and returns exit code 1.
    When `write` is True, writes the generated report JSON to `output` if provided or to the default report path under `root`.
    
    Parameters:
        root (Path): Path to the workspace root used to build the diagnostics report.
        write (bool): Whether to persist the diagnostics report to disk.
        output (Optional[Path]): Optional destination path for the report; if None and `write` is True,
            the default report path beneath `root` is used.
    
    Returns:
        int: Process exit code — `0` on success, `1` if a dependency import failed.
    """
    try:
        report = build_report(root)
    except DependencyError as exc:
        _print("ERROR", str(exc), file=sys.stderr)
        _print(
            "ERROR",
            "Rerun 'scripts/bootstrap.sh --verify' or manually install the missing package.",
            file=sys.stderr,
        )
        return 1

    _print("INFO", "Dependency imports succeeded:")
    for pkg in report["packages"]:
        _print(
            "INFO",
            f"  - {pkg['module']} ({pkg['distribution']}) version {pkg['version']} [{pkg['version_source']}].",
        )

    lock = report["lockfile"]
    if lock["exists"]:
        _print("INFO", f"requirements.lock sha256={lock['sha256']}")
    else:
        _print("WARN", "requirements.lock not found; diagnostics recorded null checksum.")

    git_sha = report["git"].get("sha")
    if git_sha:
        _print("INFO", f"Git commit: {git_sha}")
    else:
        _print("WARN", "Git commit information unavailable (no repository detected).")

    if write:
        destination = output or (root / DEFAULT_REPORT_PATH)
        write_report(report, destination)
        _print("INFO", f"Report written to {destination}")
    else:
        _print("INFO", "Report writing skipped per --no-report flag")

    return 0




def run_openai_probe(
    root: Path,
    *,
    artifacts_dir: Optional[Path],
    skip_live: bool,
    max_attempts: int,
    base_delay: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    client_factory: Optional[Callable[[], Any]] = None,
) -> int:
    """
    Run an OpenAI readiness probe and write probe artifacts to the workspace.
    
    Performs chat and embedding probes (unless skip_live is True), records probe metrics, writes a sanitized JSON report and a metrics file to the artifacts directory, and prints a short status line.
    
    Parameters:
        root (Path): Workspace root used to resolve relative artifact paths.
        artifacts_dir (Optional[Path]): Directory to write probe artifacts. If omitted, a default artifacts subpath under the workspace root is used.
        skip_live (bool): If True, skip making live OpenAI calls and write a placeholder report and metrics.
        max_attempts (int): Maximum number of retry attempts for probe calls subject to backoff and rate-limit handling.
        base_delay (float): Base delay in seconds for exponential backoff between retry attempts.
        sleep_fn (Callable[[float], None]): Function used to sleep between backoff attempts; defaults to time.sleep (primarily for testing).
        client_factory (Callable[[], Union[SharedOpenAIClient, OpenAI]]): Factory that returns a preconfigured
            :class:`SharedOpenAIClient` or raw ``OpenAI`` SDK client; primarily used for dependency injection in
            tests. When omitted, a new :class:`SharedOpenAIClient` is created from environment settings.
    
    Returns:
        int: Exit code: `0` on success or when probe was skipped; `1` if the probe failed and a failure report was written.
    """
    metrics = create_metrics()

    settings: Optional[OpenAISettings] = None

    def _settings_payload() -> Dict[str, Any]:
        if settings is None:
            return {
                "chat_model": None,
                "embedding_model": None,
                "embedding_dimensions": None,
                "chat_override": None,
                "max_attempts": None,
                "backoff_seconds": None,
                "fallback_enabled": None,
                "base_url_override": None,
                "base_url": None,
                "base_url_masked": None,
                "allow_insecure_base_url": None,
            }
        return {
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.expected_embedding_dimensions(),
            "chat_override": settings.is_chat_override,
            "max_attempts": settings.max_attempts,
            "backoff_seconds": settings.backoff_seconds,
            "fallback_enabled": settings.enable_fallback,
            "base_url_override": settings.api_base_url is not None,
            "base_url": settings.api_base_url,
            "base_url_masked": mask_base_url(settings.api_base_url),
            "allow_insecure_base_url": settings.allow_insecure_base_url,
        }

    artifacts_root = artifacts_dir or (root / DEFAULT_PROBE_REPORT_PATH.parent)
    if not artifacts_root.is_absolute():
        artifacts_root = (root / artifacts_root).resolve()

    report_path = artifacts_root / DEFAULT_PROBE_REPORT_PATH.name
    metrics_path = artifacts_root / DEFAULT_PROBE_METRICS_PATH.name
    artifacts_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    def _record_failure(exc: ProbeFailure) -> int:
        actor_name = settings.actor if settings is not None else "openai-probe"
        failure_report = {
            "status": "failed",
            "generated_at": timestamp,
            "actor": actor_name,
            "settings": _settings_payload(),
            "error": scrub_object(
                {
                    "message": str(exc),
                    "remediation": exc.remediation,
                    "details": exc.details,
                }
            ),
        }
        write_report(failure_report, report_path)
        metrics_path.write_text(metrics.export(), encoding="utf-8")
        _print("ERROR", str(exc), file=sys.stderr)
        _print("ERROR", exc.remediation, file=sys.stderr)
        return 1

    try:
        settings = OpenAISettings.load(actor="openai-probe")
    except ValueError as exc:
        failure = ProbeFailure(
            "OpenAI configuration is invalid.",
            remediation="Verify OPENAI_MODEL and OPENAI_EMBEDDING_* settings are set to supported values.",
            details={"reason": "settings_load", "message": str(exc)},
        )
        return _record_failure(failure)

    settings = settings.model_copy(
        update={"max_attempts": max_attempts, "backoff_seconds": base_delay}
    )

    if skip_live:
        report = {
            "status": "skipped",
            "generated_at": timestamp,
            "actor": settings.actor,
            "settings": _settings_payload(),
            "notes": "Live OpenAI calls skipped by operator request.",
        }
        write_report(report, report_path)
        metrics_path.write_text(metrics.export(), encoding="utf-8")
        _print("INFO", f"Probe skipped; artifacts written to {report_path} and {metrics_path}")
        return 0

    try:
        shared_client: SharedOpenAIClient
        try:
            candidate = client_factory() if client_factory else None
            if isinstance(candidate, SharedOpenAIClient):
                shared_client = candidate
            else:
                shared_client = SharedOpenAIClient(
                    settings,
                    client=candidate,
                    embedding_client=candidate,
                    metrics=metrics,
                    sleep_fn=sleep_fn,
                )
        except OpenAIClientError as exc:
            raise ProbeFailure(
                str(exc),
                remediation=exc.remediation,
                details=exc.details,
            ) from exc
        except Exception as exc:  # pragma: no cover - unexpected client init failure
            raise ProbeFailure(
                "Unable to create OpenAI client.",
                remediation="Confirm the openai package is installed and API key configured.",
                details={"reason": "client_init", "message": str(exc)},
            ) from exc

        try:
            chat_result = shared_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an OpenAI readiness probe verifying model guardrails.",
                    },
                    {
                        "role": "user",
                        "content": "Return a brief acknowledgement that chat completions are reachable.",
                    },
                ],
                temperature=0,
                max_tokens=32,
            )
            embedding_result = shared_client.embedding(
                input_text="GraphRAG readiness probe vector check.",
            )
        except OpenAIClientError as exc:
            raise ProbeFailure(
                str(exc),
                remediation=exc.remediation,
                details=exc.details,
            ) from exc

        chat_summary = _chat_summary_from_result(chat_result)
        embedding_summary = _embedding_summary_from_result(
            embedding_result,
            expected_dimensions=settings.expected_embedding_dimensions(),
        )
    except ProbeFailure as exc:
        return _record_failure(exc)

    report = {
        "status": "success",
        "generated_at": timestamp,
        "actor": settings.actor,
        "settings": _settings_payload(),
        "chat": chat_summary,
        "embedding": embedding_summary,
        "artifacts": {
            "report": str(report_path),
            "metrics": str(metrics_path),
        },
    }

    write_report(report, report_path)
    metrics_path.write_text(metrics.export(), encoding="utf-8")

    _print(
        "INFO",
        "OpenAI readiness probe completed successfully. Report stored at "
        f"{report_path} and metrics at {metrics_path}.",
    )
    return 0


def _chat_summary_from_result(result: ChatResult) -> Dict[str, Any]:
    return {
        "status": "success",
        "model": result.model,
        "fallback_used": result.fallback_used,
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason,
    }


def _embedding_summary_from_result(
    result: EmbeddingResult,
    *,
    expected_dimensions: int,
) -> Dict[str, Any]:
    return {
        "status": "success",
        "model": result.model,
        "expected_dimensions": expected_dimensions,
        "vector_length": len(result.vector),
        "latency_ms": result.latency_ms,
        "tokens_consumed": result.tokens_consumed,
    }


def _build_parser() -> argparse.ArgumentParser:
    """
    Builds an ArgumentParser configured with the CLI's "workspace" and "openai-probe" subcommands.
    
    The "workspace" subcommand validates workspace dependencies and supports:
      --root, --no-report, --output
    
    The "openai-probe" subcommand runs the OpenAI readiness probe and supports:
      --root, --artifacts-dir, --skip-live, --max-attempts, --backoff-seconds
    
    Returns:
        argparse.ArgumentParser: A parser with the configured subcommands and arguments.
    """
    parser = argparse.ArgumentParser(description="Workspace diagnostics utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    workspace = subparsers.add_parser("workspace", help="Validate workspace dependencies")
    workspace.add_argument(
        "--root",
        type=Path,
        help="Path to repository root (defaults to auto-detection).",
    )
    workspace.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the JSON report to disk.",
    )
    workspace.add_argument(
        "--output",
        type=Path,
        help="Custom output path for the JSON report (defaults to artifacts/environment/versions.json)",
    )

    probe = subparsers.add_parser("openai-probe", help="Run OpenAI readiness probe")
    probe.add_argument(
        "--root",
        type=Path,
        help="Path to repository root (defaults to auto-detection).",
    )
    probe.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Directory for probe artifacts (defaults to artifacts/openai).",
    )
    probe.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip live OpenAI calls and emit placeholder artifacts only.",
    )
    probe.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_RETRY_ATTEMPTS,
        help=f"Maximum retry attempts for rate-limited calls (default: {DEFAULT_MAX_RETRY_ATTEMPTS}).",
    )
    probe.add_argument(
        "--backoff-seconds",
        type=float,
        default=DEFAULT_BACKOFF_SECONDS,
        help=f"Initial exponential backoff (seconds) for rate limiting (default: {DEFAULT_BACKOFF_SECONDS}).",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """
    Parse command-line arguments and execute the selected diagnostics subcommand.
    
    Supports the "workspace" and "openai-probe" subcommands and validates their options before dispatching.
    
    Parameters:
    	argv (Optional[Iterable[str]]): If provided, an iterable of argument strings to parse (typically sys.argv[1:]); if None, the system argv is used.
    
    Returns:
    	exit_code (int): Process exit code where 0 indicates success and a non-zero value indicates failure.
    """
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = _compute_repo_root(args.root)

    if args.command == "workspace":
        write = not args.no_report
        output = args.output
        if output is not None and not output.is_absolute():
            output = root / output
        if output is not None and not write:
            _print("WARN", "--output specified but --no-report provided; report will not be written.")

        return run_workspace(root, write=write, output=output)

    if args.command == "openai-probe":
        artifacts_dir = args.artifacts_dir
        if artifacts_dir is not None and not artifacts_dir.is_absolute():
            artifacts_dir = root / artifacts_dir
        if args.max_attempts <= 0:
            parser.error("--max-attempts must be a positive integer")
        if args.backoff_seconds <= 0:
            parser.error("--backoff-seconds must be positive")

        return run_openai_probe(
            root,
            artifacts_dir=artifacts_dir,
            skip_live=args.skip_live,
            max_attempts=args.max_attempts,
            base_delay=args.backoff_seconds,
        )

    parser.error("Unsupported command")


if __name__ == "__main__":
    sys.exit(main())
