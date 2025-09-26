"""Reusable sanitization helpers for CLI diagnostics artifacts and logs."""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Mapping

__all__ = [
    "SECRET_ENV_KEYS",
    "SECRET_PATTERNS",
    "SENSITIVE_KEY_NAMES",
    "sanitize_text",
    "scrub_object",
]


SECRET_ENV_KEYS: frozenset[str] = frozenset(
    {
        "OPENAI_API_KEY",
        "QDRANT_API_KEY",
        "NEO4J_PASSWORD",
        "NEO4J_BOLT_PASSWORD",
        "NEO4J_AUTH",
    }
)


SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{4,}"),
    re.compile(r"(?i)(api[_-]?key)\b[^=]*=\s*([A-Za-z0-9-]{6,})"),
    re.compile(r"(?i)(bearer)\s+[A-Za-z0-9._-]{10,}"),
)


SENSITIVE_KEY_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "authorization",
        "bearer",
        "token",
        "secret",
        "password",
        "access_token",
        "refresh_token",
    }
)


def _redacted_value(value: str) -> str:
    """
    Return a redacted replacement for a non-empty string.
    
    Parameters:
        value (str): The string to redact; if empty, it is returned unchanged.
    
    Returns:
        str: `"***"` if `value` is non-empty, otherwise the original (empty) string.
    """
    return "***" if value else value


def sanitize_text(text: str, *, extra_patterns: Iterable[re.Pattern[str]] | None = None) -> str:
    """
    Redact known secret values and pattern matches from a text string.
    
    Replaces occurrences of environment values listed in SECRET_ENV_KEYS, matches of SECRET_PATTERNS, and matches of any provided extra_patterns with "***".
    
    Parameters:
        text (str): Input text to sanitize.
        extra_patterns (Iterable[re.Pattern[str]] | None): Optional additional compiled regular expressions whose matches will be replaced with "***".
    
    Returns:
        str: A copy of `text` with secrets and pattern matches replaced by "***".
    """

    sanitized = text
    for key in SECRET_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            sanitized = sanitized.replace(value, "***")
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("***", sanitized)
    if extra_patterns:
        for pattern in extra_patterns:
            sanitized = pattern.sub("***", sanitized)
    return sanitized


def scrub_object(obj: Any) -> Any:
    """
    Deeply sanitize arbitrary objects for safe JSON/file serialization.

    Strings are sanitized with :func:`sanitize_text`. Mapping values are recursively
    sanitized with sensitive keys masked. Lists and tuples are traversed element by
    element, ensuring nested structures cannot leak secrets. Unsupported types are
    returned unchanged.
    """

    if isinstance(obj, str):
        return sanitize_text(obj)
    if isinstance(obj, Mapping):
        return {
            key: "***" if key.lower() in SENSITIVE_KEY_NAMES else scrub_object(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [scrub_object(item) for item in obj]
    if isinstance(obj, tuple):
        if len(obj) == 2 and isinstance(obj[0], str):
            key = scrub_object(obj[0])
            value = "***" if obj[0].lower() in SENSITIVE_KEY_NAMES else scrub_object(obj[1])
            return (key, value)
        return tuple(scrub_object(item) for item in obj)
    return obj
