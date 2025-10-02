"""Environment utility helpers for FancyRAG tooling."""

from __future__ import annotations

import os
from itertools import chain
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


_ENV_LOADED = False


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
    if _ENV_LOADED:
        return

    env_path = _discover_dotenv_path()
    if env_path is not None:
        load_dotenv(env_path, override=False)

    _ENV_LOADED = True


def ensure_env(var: str) -> str:
    """
    Ensure an environment variable is present and return its value.
    
    This function triggers a one-time load of a .env file (if found) before reading the environment.
    
    Parameters:
        var (str): Name of the environment variable to read.
    
    Returns:
        str: The value of the requested environment variable.
    
    Raises:
        SystemExit: If the environment variable is not set; exits with the message
        "Missing required environment variable: {var}".
    """

    _load_dotenv_once()

    value = os.getenv(var)
    if value is not None:
        return value
    raise SystemExit(f"Missing required environment variable: {var}")
