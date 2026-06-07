"""Compatibility shim for the legacy ``fancryrag`` namespace."""

from __future__ import annotations

import importlib
import sys

_canonical = importlib.import_module("fancyrag")

sys.modules[__name__] = _canonical

_aliases = (
    "config",
    "embeddings",
    "logging_setup",
    "mcp",
    "mcp.runtime",
)
for name in _aliases:
    try:
        sys.modules[f"{__name__}.{name}"] = importlib.import_module(f"fancyrag.{name}")
    except ModuleNotFoundError:
        continue

__all__ = getattr(_canonical, "__all__", [])
