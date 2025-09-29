"""Lightweight structlog-compatible shim used when the dependency is absent."""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, List, Optional

_LogSink = Callable[..., None]

__all__ = [
    "BoundLogger",
    "capture_logs",
    "ensure",
    "get_log_sink",
    "get_logger",
    "set_log_sink",
]

_thread_local = threading.local()
_ensure_lock = threading.Lock()


def _default_sink(*, level: str, event: str, logger: str, **kwargs: Any) -> None:
    """Default sink used when no custom sink is registered."""


def _get_current_sink() -> _LogSink:
    sink = getattr(_thread_local, "sink", None)
    if sink is None:
        return _default_sink
    return sink


@dataclass
class BoundLogger:
    """Minimal structlog-compatible logger used by the CLI and tests."""

    name: str
    context: Dict[str, Any] = field(default_factory=dict)

    def bind(self, **new_values: Any) -> "BoundLogger":
        merged = {**self.context, **new_values}
        return BoundLogger(name=self.name, context=merged)

    def _log(self, level: str, event: str, **kwargs: Any) -> None:
        data = {**self.context, **kwargs}
        sink = _get_current_sink()
        sink(level=level, event=event, logger=self.name, **data)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log("info", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log("error", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log("warning", event, **kwargs)


def set_log_sink(sink: Optional[_LogSink]) -> None:
    """Install a log sink callable for the active thread."""

    if sink is None:
        if hasattr(_thread_local, "sink"):
            delattr(_thread_local, "sink")
        return
    _thread_local.sink = sink


def get_log_sink() -> _LogSink:
    """Return the sink currently associated with the active thread."""

    return _get_current_sink()


def get_logger(name: Optional[str] = None) -> BoundLogger:
    """Return a :class:`BoundLogger` with the provided ``name``."""

    return BoundLogger(name=name or "structlog")


@contextmanager
def capture_logs() -> Iterator[List[Dict[str, Any]]]:
    """Collect log events emitted while the context is active."""

    events: List[Dict[str, Any]] = []

    def _capture_sink(*, level: str, event: str, logger: str, **kwargs: Any) -> None:
        entry = {"event": event, "level": level, "logger": logger, **kwargs}
        events.append(entry)

    previous_sink = get_log_sink()
    set_log_sink(_capture_sink)
    try:
        yield events
    finally:
        set_log_sink(previous_sink)


def _create_module() -> ModuleType:
    module = ModuleType("structlog")
    module.__doc__ = (
        "Lightweight structlog-compatible interface used by FancyRAG when the "
        "real dependency is unavailable. Only the small subset exercised in the "
        "codebase is implemented."
    )
    module.BoundLogger = BoundLogger
    module.get_logger = get_logger
    module.set_log_sink = set_log_sink
    module.get_log_sink = get_log_sink
    module.__all__ = ["BoundLogger", "get_logger", "set_log_sink", "get_log_sink"]

    testing_module = ModuleType("structlog.testing")
    testing_module.capture_logs = capture_logs
    testing_module.__all__ = ["capture_logs"]
    module.testing = testing_module

    sys.modules.setdefault("structlog.testing", testing_module)
    sys.modules.setdefault("structlog", module)
    return module


def ensure() -> ModuleType:
    """Return a structlog-like module, installing the shim when required."""
    existing = sys.modules.get("structlog")
    if existing is not None:
        return existing

    with _ensure_lock:
        # Re-check inside the lock to handle the race condition.
        existing = sys.modules.get("structlog")
        if existing is not None:
            return existing
        return _create_module()
