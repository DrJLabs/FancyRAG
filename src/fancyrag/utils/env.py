"""Environment utility helpers for FancyRAG tooling."""

from __future__ import annotations

import importlib.util
import os
from itertools import chain
from pathlib import Path
from typing import Optional

_DOTENV_AVAILABLE = importlib.util.find_spec("dotenv") is not None

if _DOTENV_AVAILABLE:
    from dotenv import load_dotenv
else:
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:  # type: ignore[no-redef]
        """Fallback that indicates no `.env` file was loaded when python-dotenv is absent."""

        return False


_ENV_LOADED = False
_DOTENV_PATH: Path | None = None


def _discover_dotenv_path() -> Optional[Path]:
    """
    Locate the most relevant `.env` file for the current environment.
    
    Search behavior:
    - If the environment variable `FANCYRAG_DOTENV_PATH` is set and points to an existing path (after expanding `~`), that path is returned.
    - Otherwise, search for a `.env` file in this process's current working directory, then each of its parent directories, and finally the module root (three levels above this file); the first existing `.env` file found is returned.
    
    Returns:
        Path: The filesystem path to the discovered `.env` file, or `None` if no `.env` file was found.
    """

    override_path = os.getenv("FANCYRAG_DOTENV_PATH")
    if override_path:
        candidate = Path(override_path).expanduser()
        return candidate if candidate.exists() else None

    cwd = Path.cwd().resolve()
    module_root = Path(__file__).resolve().parents[3]

    for base in chain([cwd], cwd.parents, [module_root]):
        candidate = base / ".env"
        if candidate.exists():
            return candidate

    return None


def _load_dotenv_once() -> None:
    """Load environment variables from a `.env` file one time."""

    global _ENV_LOADED
    global _DOTENV_PATH
    if _ENV_LOADED:
        return

    env_path = _discover_dotenv_path()
    if env_path is not None:
        if load_dotenv(env_path, override=False):
            _DOTENV_PATH = env_path

    _ENV_LOADED = True


def load_project_dotenv() -> Path | None:
    """Ensure the project's `.env` file (if any) is loaded and return its path."""

    _load_dotenv_once()
    return _DOTENV_PATH


def get_settings(*, refresh: bool = False, require: set[str] | None = None):
    """Return the cached FancyRAGSettings aggregate, loading it on demand.

    Args:
        refresh: When True, bypass the cache and reload settings from the environment.
        require: Optional set of component names (e.g., {"qdrant"}) that must be
            configured; a ``ValueError`` is raised if any are missing.
    """

    from config.settings import FancyRAGSettings

    return FancyRAGSettings.load(refresh=refresh, require=require)


def ensure_env(var: str) -> str:
    """Legacy helper that ensures a required environment variable is present."""

    try:
        settings = get_settings(refresh=True)
    except ValueError:
        settings = None

    if settings is not None:
        resolved = settings.export_environment()
        if value := resolved.get(var):
            return value

    # Ensure .env variables are available even when typed settings failed.
    load_project_dotenv()
    if value := os.getenv(var):
        return value

    raise SystemExit(f"Missing required environment variable: {var}")


__all__ = [
    "ensure_env",
    "get_settings",
    "load_project_dotenv",
]
