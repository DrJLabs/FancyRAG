"""Environment utility helpers for FancyRAG tooling."""

from __future__ import annotations

import os
from itertools import chain
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


_ENV_LOADED = False


def _discover_dotenv_path() -> Optional[Path]:
    """Return the most relevant `.env` path, if any."""

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
    if _ENV_LOADED:
        return

    env_path = _discover_dotenv_path()
    if env_path is not None:
        load_dotenv(env_path, override=False)

    _ENV_LOADED = True


def ensure_env(var: str) -> str:
    """Return the value of ``var`` or exit the process with an error message."""

    _load_dotenv_once()

    value = os.getenv(var)
    if value is not None:
        return value
    raise SystemExit(f"Missing required environment variable: {var}")
