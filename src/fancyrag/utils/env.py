"""Environment utility helpers for FancyRAG tooling."""

from __future__ import annotations

import os


def ensure_env(var: str) -> str:
    """Return the value of ``var`` or exit the process with an error message."""
    value = os.getenv(var)
    if value is not None:
        return value
    raise SystemExit(f"Missing required environment variable: {var}")
