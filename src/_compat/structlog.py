"""Helpers to access structlog or a local shim when the dependency is missing."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Optional

from . import structlog_shim

__all__ = ["capture_logs", "ensure_structlog", "get_logger", "structlog"]


def ensure_structlog() -> ModuleType:
    """Return the real structlog module or install the shim on demand."""

    try:
        module = import_module("structlog")
    except ModuleNotFoundError:
        return structlog_shim.ensure()

    # structlog 25.x removed the top-level ``get_logger`` helper in favour of the
    # ``structlog.stdlib`` namespace. The rest of the codebase still imports the
    # helper from the compatibility layer, so provide a shim that delegates to the
    # new API when necessary.
    return module


structlog = ensure_structlog()


def get_logger(name: Optional[str] = None, *args: Any, **kwargs: Any) -> Any:
    """Return a structlog-compatible logger using the active backend."""

    module = ensure_structlog()

    getter = getattr(module, "get_logger", None)
    if callable(getter):
        return getter(name, *args, **kwargs)

    stdlib: Optional[ModuleType] = getattr(module, "stdlib", None)
    stdlib_getter = getattr(stdlib, "get_logger", None) if stdlib is not None else None
    if callable(stdlib_getter):
        return stdlib_getter(name, *args, **kwargs)

    # Fall back to the shim's lightweight logger implementation when neither API is available.
    return structlog_shim.get_logger(name)

_testing = getattr(structlog, "testing", None)
if _testing is not None and hasattr(_testing, "capture_logs"):
    capture_logs: Callable[..., Any] = getattr(_testing, "capture_logs")
else:  # pragma: no cover - only triggered when the API surface changes upstream
    def capture_logs(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        raise RuntimeError("structlog.testing.capture_logs is unavailable")
