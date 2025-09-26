"""Lightweight structlog-compatible interface used for testing.

This module provides a tiny subset of the real ``structlog`` package so the
project can run in constrained environments where the dependency is not
available. Only the APIs exercised in our codebase are implemented:

* :func:`get_logger`
* :meth:`BoundLogger.bind`
* :meth:`BoundLogger.info`
* :meth:`BoundLogger.error`

The implementation keeps logging semantics intentionally simple - log entries
are forwarded to a pluggable sink callable. Tests install an in-memory sink via
:func:`structlog.testing.capture_logs` to assert on emitted events.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import sys


def _try_import_real_structlog() -> bool:
    """Replace this shim with the real ``structlog`` module when available."""

    try:
        distribution = importlib.metadata.distribution("structlog")
    except importlib.metadata.PackageNotFoundError:
        return False

    package_init = distribution.locate_file("structlog/__init__.py")
    spec = importlib.util.spec_from_file_location(__name__, str(package_init))
    if spec is None or spec.loader is None:
        return False

    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    loader.exec_module(module)  # type: ignore[call-arg]

    current_dict = sys.modules[__name__].__dict__
    current_dict.clear()
    current_dict.update(module.__dict__)
    return True


if not _try_import_real_structlog():
    import threading
    from dataclasses import dataclass, field
    from typing import Any, Callable, Dict, Optional

    _LogSink = Callable[..., None]

    __all__ = ["BoundLogger", "get_log_sink", "get_logger", "set_log_sink"]

    _thread_local = threading.local()

    def _default_sink(*, level: str, event: str, logger: str, **kwargs: Any) -> None:
        """Fallback sink that simply ignores log entries.

        The goal of this shim is determinism in tests rather than rich logging.
        Production environments should install the real ``structlog`` package,
        which provides far more sophisticated configuration options.
        """

    def _get_current_sink() -> _LogSink:
        return getattr(_thread_local, "sink", _default_sink)


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


    def set_log_sink(sink: Optional[_LogSink]) -> None:
        """Install a new log sink callable for the current thread.

        Passing ``None`` restores the default sink.
        """

        if sink is None:
            if hasattr(_thread_local, "sink"):
                delattr(_thread_local, "sink")
            return

        _thread_local.sink = sink


    def get_log_sink() -> _LogSink:
        """Return the currently installed log sink for the active thread."""

        return _get_current_sink()


    def get_logger(name: Optional[str] = None) -> BoundLogger:
        """Return a :class:`BoundLogger` instance.

        The real ``structlog`` allows much richer configuration; this shim keeps
        the API surface minimal by returning a logger bound to the supplied name
        (or a generic placeholder when ``None`` is provided).
        """

        return BoundLogger(name=name or "structlog")


if "_try_import_real_structlog" in globals():
    del _try_import_real_structlog
