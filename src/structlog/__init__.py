"""Lightweight structlog-compatible interface used for testing.

This module provides a tiny subset of the real ``structlog`` package so the
project can run in constrained environments where the dependency is not
available.  Only the APIs exercised in our codebase are implemented:

* :func:`get_logger`
* :meth:`BoundLogger.bind`
* :meth:`BoundLogger.info`
* :meth:`BoundLogger.error`

The implementation keeps logging semantics intentionally simple – log entries
are forwarded to a pluggable sink callable.  Tests install an in-memory sink
via :func:`structlog.testing.capture_logs` to assert on emitted events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

_LogSink = Callable[..., None]

__all__ = ["get_logger", "BoundLogger", "set_log_sink"]


def _default_sink(*, level: str, event: str, logger: str, **kwargs: Any) -> None:
    """Fallback sink that simply ignores log entries.

    The goal of this shim is determinism in tests rather than rich logging.
    Production environments should install the real ``structlog`` package,
    which provides far more sophisticated configuration options.
    """


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
        _log_sink(level=level, event=event, logger=self.name, **data)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log("info", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log("error", event, **kwargs)


_log_sink: _LogSink = _default_sink


def set_log_sink(sink: Optional[_LogSink]) -> None:
    """Install a new log sink callable.

    Passing ``None`` restores the default sink.
    """

    global _log_sink
    _log_sink = sink or _default_sink


def get_logger(name: Optional[str] = None) -> BoundLogger:
    """Return a :class:`BoundLogger` instance.

    The real ``structlog`` allows much richer configuration; this shim keeps the
    API surface minimal by returning a logger bound to the supplied name (or a
    generic placeholder when ``None`` is provided).
    """

    return BoundLogger(name=name or "structlog")
