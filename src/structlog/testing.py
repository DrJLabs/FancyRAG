"""Testing helpers for the lightweight ``structlog`` shim.

Only the :func:`capture_logs` context manager is implemented because the test
suite relies on it to assert emitted events.  The helper temporarily installs a
sink that records log entries and restores the previous sink afterwards.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Dict, List

from . import get_log_sink, set_log_sink

__all__ = ["capture_logs"]


@contextmanager
def capture_logs() -> Iterator[List[Dict[str, Any]]]:
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
