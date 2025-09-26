"""Helpers to access structlog or a local shim when the dependency is missing."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Callable

from . import structlog_shim

__all__ = ["capture_logs", "ensure_structlog", "structlog"]


def ensure_structlog() -> ModuleType:
    """Return the real structlog module or install the shim on demand."""

    try:
        return import_module("structlog")
    except ModuleNotFoundError:
        return structlog_shim.ensure()


structlog = ensure_structlog()

_testing = getattr(structlog, "testing", None)
if _testing is not None and hasattr(_testing, "capture_logs"):
    capture_logs: Callable[..., Any] = getattr(_testing, "capture_logs")
else:  # pragma: no cover - only triggered when the API surface changes upstream
    def capture_logs(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("structlog.testing.capture_logs is unavailable")

