"""Filesystem and repository path utilities for FancyRAG."""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path


@functools.lru_cache(maxsize=1)
def resolve_repo_root() -> Path | None:
    """Return the repository root directory if git metadata is available."""

    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        result = subprocess.run(
            [git_executable, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None
    root = result.stdout.strip()
    return Path(root) if root else None


def ensure_directory(path: Path) -> None:
    """Ensure that the parent directory for `path` exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


def relative_to_repo(path: Path, *, base: Path | None = None) -> str:
    """Resolve `path` relative to the repository root or fallback directories."""

    candidate_roots: list[Path] = []
    if base is not None:
        candidate_roots.append(base.resolve())
    repo_root = resolve_repo_root()
    if repo_root is not None:
        candidate_roots.append(repo_root)
    candidate_roots.append(Path.cwd())

    resolved = path.resolve()
    for root in candidate_roots:
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            continue
    return str(resolved)
