"""Workspace diagnostics utilities for GraphRAG CLI environments."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MODULES: List[tuple[str, str]] = [
    ("neo4j_graphrag", "neo4j-graphrag"),
    ("neo4j", "neo4j"),
    ("qdrant_client", "qdrant-client"),
    ("openai", "openai"),
    ("structlog", "structlog"),
    ("pytest", "pytest"),
]

SECRET_ENV_KEYS = {
    "OPENAI_API_KEY",
    "QDRANT_API_KEY",
    "NEO4J_PASSWORD",
    "NEO4J_BOLT_PASSWORD",
}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{4,}"),
    re.compile(r"(?i)(api[_-]?key)\b[^=]*=\s*([A-Za-z0-9-]{6,})"),
]

DEFAULT_REPORT_PATH = Path("artifacts") / "environment" / "versions.json"


class DependencyError(RuntimeError):
    """Raised when a required module cannot be imported."""


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


def _sanitize_text(message: str) -> str:
    sanitized = message
    for key in SECRET_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            sanitized = sanitized.replace(value, "***")
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("***", sanitized)
    return sanitized


def _print(level: str, message: str, *, file = sys.stdout) -> None:
    text = _sanitize_text(message)
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
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command != "workspace":  # pragma: no cover - defensive
        parser.error("Unsupported command")

    root = _compute_repo_root(args.root)

    write = not args.no_report
    output = args.output
    if output is not None and not output.is_absolute():
        output = root / output
    if output is not None and not write:
        _print("WARN", "--output specified but --no-report provided; report will not be written.")

    return run_workspace(root, write=write, output=output)


if __name__ == "__main__":
    sys.exit(main())
