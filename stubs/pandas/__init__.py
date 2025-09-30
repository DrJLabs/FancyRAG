"""Lightweight pandas stub used for local development where full pandas is unavailable."""

from __future__ import annotations

import datetime as _dt
import types

NA = object()
NaT = object()

class Series:  # pragma: no cover - simple stub
    """Placeholder for pandas.Series"""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        self._data = args
        self._kwargs = kwargs


class DataFrame:  # pragma: no cover - simple stub
    """Placeholder for pandas.DataFrame"""

    def __init__(self, *args, **kwargs) -> None:
        self._data = args
        self._kwargs = kwargs


class Categorical:  # pragma: no cover - simple stub
    """Placeholder for pandas.Categorical"""

    def __init__(self, *args, **kwargs) -> None:
        self._data = args
        self._kwargs = kwargs


class ExtensionArray:  # pragma: no cover - simple stub
    """Placeholder for pandas.core.arrays.ExtensionArray"""

    def __init__(self, *args, **kwargs) -> None:
        self._data = args
        self._kwargs = kwargs


core = types.SimpleNamespace(arrays=types.SimpleNamespace(ExtensionArray=ExtensionArray))


class Timestamp(_dt.datetime):  # pragma: no cover - simple stub
    pass


class Timedelta(_dt.timedelta):  # pragma: no cover - simple stub
    pass

__all__ = [
    "Series",
    "DataFrame",
    "Categorical",
    "ExtensionArray",
    "Timestamp",
    "Timedelta",
    "NA",
    "NaT",
    "core",
]
