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
from typing import Any, Callable, Dict, Iterable, List, Optional

from cli.sanitizer import sanitize_text, scrub_object
from cli.telemetry import create_metrics
from cli.utils import ensure_embedding_dimensions
from config.settings import OpenAISettings

try:  # pragma: no cover - import guarded for environments without OpenAI
    from openai import APIConnectionError, APIError, APIStatusError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - fallback types for static analysis/tests
    class APIError(Exception):
        pass

    class APIConnectionError(APIError):
        pass

    class APIStatusError(APIError):
        def __init__(self, message: str, *, status_code: int | None = None) -> None:
            """
            Initialize the exception with an error message and an optional numeric status code.
            
            Parameters:
                message (str): Human-readable error message.
                status_code (int | None): Optional numeric status code providing additional context (for example, an HTTP status); stored on the instance as `status_code`.
            """
            super().__init__(message)
            self.status_code = status_code

    class RateLimitError(APIError):
        pass

    class OpenAI:  # type: ignore[no-redef]
        def __init__(self) -> None:  # pragma: no cover - fallback stub
            """
            Initialize the fallback OpenAI client constructor that always fails.
            
            Raises:
                RuntimeError: Indicates the required 'openai' package is not installed.
            """
            raise RuntimeError("openai package is not installed")

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
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 0.5


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


def _create_openai_client() -> OpenAI:
    """
    Create an OpenAI client using the default configuration.
    
    Returns:
        client (OpenAI): An OpenAI client instance configured with default settings.
    """
    return OpenAI()


def _is_rate_limit_error(error: Exception) -> bool:
    """
    Detects whether an exception represents an OpenAI rate-limit condition.
    
    Parameters:
        error (Exception): The exception to inspect.
    
    Returns:
        True if the error is a rate-limiting error (e.g., a `RateLimitError` or an `APIStatusError` with HTTP status 429), False otherwise.
    """
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, APIStatusError) and getattr(error, "status_code", None) == 429:
        return True
    return False


def _usage_value(usage: Any, attr: str) -> int:
    """
    Extracts an integer usage metric named by `attr` from a usage object.
    
    Parameters:
    	usage (Any): Usage data which may be None, an object with attributes, or a dict.
    	attr (str): Name of the attribute or key to extract from `usage`.
    
    Returns:
    	int: The integer value of the requested metric, or 0 if the metric is missing or falsy.
    """
    if usage is None:
        return 0

    if isinstance(usage, dict):
        value = usage.get(attr)
    else:
        value = getattr(usage, attr, None)

    return int(value or 0)


def _execute_with_backoff(
    operation: Callable[[], Dict[str, Any]],
    *,
    description: str,
    max_attempts: int,
    base_delay: float,
    sleep_fn: Callable[[float], None],
) -> Dict[str, Any]:
    """
    Execute `operation` and retry on OpenAI rate-limit errors using exponential backoff.
    
    Parameters:
        operation (Callable[[], Dict[str, Any]]): Callable that performs the probe step and returns a result mapping.
        description (str): Short text describing the operation for error messages.
        max_attempts (int): Maximum number of attempts before giving up.
        base_delay (float): Initial backoff delay in seconds; doubled after each retry.
        sleep_fn (Callable[[float], None]): Function used to pause between retries (e.g., time.sleep).
    
    Returns:
        result (Dict[str, Any]): The mapping returned by a successful `operation` call.
    
    Raises:
        ProbeFailure: If repeated rate-limit errors occur for `max_attempts` attempts, with remediation guidance and aggregated error details.
        Exception: Any non-rate-limit exception raised by `operation` is propagated unchanged.
    """
    attempt = 0
    delay = base_delay
    errors: List[str] = []
    while attempt < max_attempts:
        attempt += 1
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - controlled handling
            if _is_rate_limit_error(exc):
                errors.append(str(exc))
                if attempt >= max_attempts:
                    raise ProbeFailure(
                        f"Rate limit exceeded for {description} after {max_attempts} attempts.",
                        remediation=(
                            "Reduce concurrent OpenAI usage, review token budgets, "
                            "and retry later or run with --skip-live while investigating."
                        ),
                        details={"errors": errors, "reason": "rate_limit"},
                    ) from exc
                sleep_fn(delay)
                delay *= 2
                continue
            raise


def _chat_probe(
    client: OpenAI,
    *,
    settings: OpenAISettings,
    max_attempts: int,
    base_delay: float,
    sleep_fn: Callable[[float], None],
    metrics,
) -> Dict[str, Any]:
    """
    Perform a chat completion probe against the OpenAI chat API, record probe metrics, and return a concise result summary.
    
    Parameters:
        settings (OpenAISettings): Probe configuration including `chat_model`, `actor`, and `is_chat_override`.
        max_attempts (int): Maximum number of attempts for the probe when retrying on rate limits.
        base_delay (float): Base backoff delay in seconds used for exponential backoff between attempts.
        sleep_fn (Callable[[float], None]): Function used to sleep between retries (e.g., time.sleep).
        metrics: Metrics collector with an `observe_chat` method used to record latency and token usage.
    
    Returns:
        Dict[str, Any]: A summary dictionary containing:
            - "status": Probe status, e.g., "success".
            - "model": The chat model name used.
            - "fallback_used": Whether a chat override was applied.
            - "latency_ms": Observed latency in milliseconds (rounded to two decimals).
            - "prompt_tokens": Prompt token count (int).
            - "completion_tokens": Completion token count (int).
            - "finish_reason": Finish reason string from the model, or `None` if unavailable.
    """
    def _call() -> Dict[str, Any]:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=settings.chat_model,
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
        latency_ms = (time.perf_counter() - start) * 1000.0
        usage = getattr(response, "usage", None)
        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")

        metrics.observe_chat(
            model=settings.chat_model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            actor=settings.actor,
        )

        finish_reason = None
        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            finish_reason = getattr(first, "finish_reason", None) or (
                first.get("finish_reason") if isinstance(first, dict) else None
            )

        return {
            "status": "success",
            "model": settings.chat_model,
            "fallback_used": settings.is_chat_override,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
        }

    try:
        return _execute_with_backoff(
            _call,
            description="chat completion",
            max_attempts=max_attempts,
            base_delay=base_delay,
            sleep_fn=sleep_fn,
        )
    except APIConnectionError as exc:  # pragma: no cover - network failures
        raise ProbeFailure(
            "Failed to reach OpenAI for chat completion.",
            remediation="Check network connectivity and OpenAI status page before retrying.",
            details={"reason": "connection_error"},
        ) from exc
    except APIError as exc:
        raise ProbeFailure(
            "OpenAI chat completion returned an error.",
            remediation="Verify model configuration and ensure the API key has access to the selected model.",
            details={"reason": "api_error", "message": str(exc)},
        ) from exc


def _embedding_probe(
    client: OpenAI,
    *,
    settings: OpenAISettings,
    max_attempts: int,
    base_delay: float,
    sleep_fn: Callable[[float], None],
    metrics,
) -> Dict[str, Any]:
    """
    Perform an embedding probe against the OpenAI embedding endpoint and report metrics.
    
    Creates an embedding for a fixed probe input, validates the presence and dimensionality of the returned vector, records latency and token usage via the provided metrics collector, and returns a summary of the probe result.
    
    Parameters:
        settings (OpenAISettings): Probe configuration (embedding model, expected dimensions, actor).
        max_attempts (int): Maximum retry attempts for transient failures.
        base_delay (float): Base backoff delay in seconds.
        sleep_fn (Callable[[float], None]): Function used to sleep between backoff attempts.
        
    Returns:
        dict: Summary containing keys:
            - "status": "success" on success.
            - "model": embedding model used.
            - "expected_dimensions": expected embedding length from settings.
            - "vector_length": length of the returned embedding vector.
            - "latency_ms": observed request latency in milliseconds (rounded to 2 decimals).
            - "tokens_consumed": total tokens reported by the provider (0 if unavailable).
    
    Raises:
        ProbeFailure: If the response is missing embedding data, the embedding dimensions do not match expectations, network connectivity to OpenAI fails, or the OpenAI API returns an error. Details and remediation guidance are provided on the exception.
    """
    expected_length = settings.expected_embedding_dimensions()

    def _call() -> Dict[str, Any]:
        """
        Validate an OpenAI embedding response, record embedding metrics, and return a summary of the probe result.
        
        Raises:
            ProbeFailure: If the response contains no embedding data or the embedding vector is missing.
        
        Returns:
            dict: Summary of the embedding probe containing:
                - "status": Probe outcome, e.g., "success".
                - "model": The embedding model used.
                - "expected_dimensions": The expected embedding length.
                - "vector_length": Length of the returned embedding vector.
                - "latency_ms": Round-trip latency in milliseconds (rounded to two decimals).
                - "tokens_consumed": Number of tokens reported as consumed.
        """
        start = time.perf_counter()
        response = client.embeddings.create(
            model=settings.embedding_model,
            input="GraphRAG readiness probe vector check.",
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        data = getattr(response, "data", None) or []
        if not data:
            raise ProbeFailure(
                "Embedding response did not include any vector data.",
                remediation="Inspect OpenAI embedding response structure; retry after confirming service health.",
                details={"reason": "no_embedding"},
            )
        embedding = getattr(data[0], "embedding", None)
        if embedding is None and isinstance(data[0], dict):
            embedding = data[0].get("embedding")
        if embedding is None:
            raise ProbeFailure(
                "Embedding vector missing from OpenAI response.",
                remediation="Upgrade openai SDK or retry once service resumes returning vectors.",
                details={"reason": "missing_embedding"},
            )

        ensure_embedding_dimensions(embedding, settings=settings)

        usage = getattr(response, "usage", None)
        tokens_consumed = _usage_value(usage, "total_tokens")
        metrics.observe_embedding(
            model=settings.embedding_model,
            latency_ms=latency_ms,
            vector_length=len(embedding),
            tokens_consumed=tokens_consumed,
            actor=settings.actor,
        )
        return {
            "status": "success",
            "model": settings.embedding_model,
            "expected_dimensions": expected_length,
            "vector_length": len(embedding),
            "latency_ms": round(latency_ms, 2),
            "tokens_consumed": tokens_consumed,
        }

    try:
        return _execute_with_backoff(
            _call,
            description="embedding request",
            max_attempts=max_attempts,
            base_delay=base_delay,
            sleep_fn=sleep_fn,
        )
    except ValueError as exc:
        raise ProbeFailure(
            str(exc),
            remediation="Ensure OPENAI_EMBEDDING_DIMENSIONS matches the provider response or adjust overrides.",
            details={"reason": "dimension_mismatch"},
        ) from exc
    except APIConnectionError as exc:  # pragma: no cover - network failures
        raise ProbeFailure(
            "Failed to reach OpenAI for embeddings.",
            remediation="Check network connectivity and OpenAI status before retrying.",
            details={"reason": "connection_error"},
        ) from exc
    except APIError as exc:
        raise ProbeFailure(
            "OpenAI embeddings API returned an error.",
            remediation="Verify embedding model availability and review OpenAI account usage limits.",
            details={"reason": "api_error", "message": str(exc)},
        ) from exc


def run_openai_probe(
    root: Path,
    *,
    artifacts_dir: Optional[Path],
    skip_live: bool,
    max_attempts: int,
    base_delay: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    client_factory: Callable[[], OpenAI] = _create_openai_client,
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
        client_factory (Callable[[], OpenAI]): Factory that returns an OpenAI client instance; used to construct the client and injectable for tests.
    
    Returns:
        int: Exit code: `0` on success or when probe was skipped; `1` if the probe failed and a failure report was written.
    """
    settings = OpenAISettings.load(actor="openai-probe")
    metrics = create_metrics()

    artifacts_root = artifacts_dir or (root / DEFAULT_PROBE_REPORT_PATH.parent)
    if not artifacts_root.is_absolute():
        artifacts_root = (root / artifacts_root).resolve()

    report_path = artifacts_root / DEFAULT_PROBE_REPORT_PATH.name
    metrics_path = artifacts_root / DEFAULT_PROBE_METRICS_PATH.name
    artifacts_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    if skip_live:
        report = {
            "status": "skipped",
            "generated_at": timestamp,
            "actor": settings.actor,
            "settings": {
                "chat_model": settings.chat_model,
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.expected_embedding_dimensions(),
            },
            "notes": "Live OpenAI calls skipped by operator request.",
        }
        write_report(report, report_path)
        metrics_path.write_text(metrics.export(), encoding="utf-8")
        _print("INFO", f"Probe skipped; artifacts written to {report_path} and {metrics_path}")
        return 0

    def _record_failure(exc: ProbeFailure) -> int:
        failure_report = {
            "status": "failed",
            "generated_at": timestamp,
            "actor": settings.actor,
            "settings": {
                "chat_model": settings.chat_model,
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.expected_embedding_dimensions(),
            },
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
        try:
            client = client_factory()
        except ProbeFailure:
            raise
        except Exception as exc:  # pragma: no cover - misconfiguration
            raise ProbeFailure(
                "Unable to create OpenAI client.",
                remediation="Confirm the openai package is installed and API key configured.",
                details={"reason": "client_init", "message": str(exc)},
            ) from exc

        chat_summary = _chat_probe(
            client,
            settings=settings,
            max_attempts=max_attempts,
            base_delay=base_delay,
            sleep_fn=sleep_fn,
            metrics=metrics,
        )
        embedding_summary = _embedding_probe(
            client,
            settings=settings,
            max_attempts=max_attempts,
            base_delay=base_delay,
            sleep_fn=sleep_fn,
            metrics=metrics,
        )
    except ProbeFailure as exc:
        return _record_failure(exc)

    report = {
        "status": "success",
        "generated_at": timestamp,
        "actor": settings.actor,
        "settings": {
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.expected_embedding_dimensions(),
            "chat_override": settings.is_chat_override,
        },
        "chat": scrub_object(chat_summary),
        "embedding": scrub_object(embedding_summary),
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
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Maximum retry attempts for rate-limited calls (default: {DEFAULT_MAX_ATTEMPTS}).",
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

    root: Optional[Path] = None
    if args.command in {"workspace", "openai-probe"}:
        root = _compute_repo_root(args.root)

    if args.command == "workspace":
        assert root is not None

        write = not args.no_report
        output = args.output
        if output is not None and not output.is_absolute():
            output = root / output
        if output is not None and not write:
            _print("WARN", "--output specified but --no-report provided; report will not be written.")

        return run_workspace(root, write=write, output=output)

    if args.command == "openai-probe":
        assert root is not None
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
