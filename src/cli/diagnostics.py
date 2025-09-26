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

from cli.sanitizer import sanitize_mapping, scrub_object
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
            super().__init__(message)
            self.status_code = status_code

    class RateLimitError(APIError):
        pass

    class OpenAI:  # type: ignore[no-redef]
        def __init__(self) -> None:  # pragma: no cover - fallback stub
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
    if explicit is not None:
        return explicit.resolve()

    # Attempt to locate the repository root by walking up from this file
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / ".git").exists() or (parent / "requirements.lock").exists():
            return parent
    return Path.cwd().resolve()


def _print(level: str, message: str, *, file = sys.stdout) -> None:
    from cli.sanitizer import sanitize_text

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
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = scrub_object(report)
    path.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_workspace(root: Path, *, write: bool, output: Optional[Path]) -> int:
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
    return OpenAI()


def _is_rate_limit_error(error: Exception) -> bool:
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, APIStatusError) and getattr(error, "status_code", None) == 429:
        return True
    return False


def _usage_value(usage: Any, attr: str) -> int:
    if usage is None:
        return 0
    value = getattr(usage, attr, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(attr)
    return int(value or 0)


def _execute_with_backoff(
    operation: Callable[[], Dict[str, Any]],
    *,
    description: str,
    max_attempts: int,
    base_delay: float,
    sleep_fn: Callable[[float], None],
) -> Dict[str, Any]:
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
    expected_length = settings.expected_embedding_dimensions()

    def _call() -> Dict[str, Any]:
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

    try:
        client = client_factory()
    except Exception as exc:  # pragma: no cover - misconfiguration
        raise ProbeFailure(
            "Unable to create OpenAI client.",
            remediation="Confirm the openai package is installed and API key configured.",
            details={"reason": "client_init", "message": str(exc)},
        ) from exc

    try:
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
        failure_report = {
            "status": "failed",
            "generated_at": timestamp,
            "actor": settings.actor,
            "settings": {
                "chat_model": settings.chat_model,
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.expected_embedding_dimensions(),
            },
            "error": sanitize_mapping(
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
        "chat": sanitize_mapping(chat_summary),
        "embedding": sanitize_mapping(embedding_summary),
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
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "workspace":
        root = _compute_repo_root(args.root)

        write = not args.no_report
        output = args.output
        if output is not None and not output.is_absolute():
            output = root / output
        if output is not None and not write:
            _print("WARN", "--output specified but --no-report provided; report will not be written.")

        return run_workspace(root, write=write, output=output)

    if args.command == "openai-probe":
        root = _compute_repo_root(args.root)
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
