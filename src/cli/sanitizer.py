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


SENSITIVE_KEY_PATTERN = r"(?:api[_-]?key|authorization|bearer|token|secret|password|access[_-]?token|refresh[_-]?token)"
SENSITIVE_NAME_TOKENS: tuple[str, ...] = ("token", "secret", "password")
SENSITIVE_KEY_SUFFIX_PATTERN = re.compile(r"(?i)(?:^|[_-])key(?:$|[_-])")


SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{4,}"),
    re.compile(
        rf"(?i)"  # case insensitive
        rf"\"?{SENSITIVE_KEY_PATTERN}\"?"  # optional quotes around key name
        rf"\s*[:=]\s*"  # key/value separator
        rf"['\"]?(?:Bearer\s+)?[A-Za-z0-9._-]{{4,}}['\"]?"  # sanitized value with optional quotes/Bearer prefix
    ),
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
        "x-openai-client",
        "openai-organization",
    }
)

def _is_sensitive_name(name: str) -> bool:
    """Return True if a mapping or environment key name should be considered sensitive."""

    lower = name.lower()
    if lower in SENSITIVE_KEY_NAMES:
        return True
    if SENSITIVE_KEY_SUFFIX_PATTERN.search(name):
        return True
    for token in SENSITIVE_NAME_TOKENS:
        if re.search(rf"(?i)(?:^|[_-]){token}(?:$|[_-])", name):
            return True
    return False


def sanitize_text(text: Any, *, extra_patterns: Iterable[re.Pattern[str]] | None = None) -> Any:
    """
    Redact secret values and regex pattern matches from a text string by replacing them with "***".

    Parameters:
        text (Any): Input text to sanitize; non-string values are returned unchanged.
        extra_patterns (Iterable[re.Pattern[str]] | None): Optional compiled regular expressions whose matches will also be redacted.

    Returns:
        Any: A sanitized copy of `text` when it is a string; otherwise the original object is returned unchanged.
    """
    if not isinstance(text, str):
        return text

    sanitized = text
    values_to_mask: list[str] = []

    for key in SECRET_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            values_to_mask.append(value)

    for key, value in os.environ.items():
        if not value:
            continue
        if _is_sensitive_name(key):
            values_to_mask.append(value)

    for value in sorted(values_to_mask, key=len, reverse=True):
        sanitized = sanitized.replace(value, "***")
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("***", sanitized)
    if extra_patterns:
        for pattern in extra_patterns:
            sanitized = pattern.sub("***", sanitized)
    return sanitized


def scrub_object(obj: Any, *, visited: set[int] | None = None) -> Any:
    """Deeply sanitize arbitrary objects for safe JSON/file serialization."""

    if isinstance(obj, str):
        return sanitize_text(obj)
    if visited is None:
        visited = set()

    if isinstance(obj, Mapping):
        obj_id = id(obj)
        if obj_id in visited:
            return "<circular>"
        visited.add(obj_id)
        try:
            result: dict[Any, Any] = {}
            for key, value in obj.items():
                if isinstance(key, str) and _is_sensitive_name(key):
                    result[key] = "***" if value is not None else value
                else:
                    result[key] = scrub_object(value, visited=visited)
            return result
        finally:
            visited.remove(obj_id)
    elif isinstance(obj, list):
        obj_id = id(obj)
        if obj_id in visited:
            return "<circular>"
        visited.add(obj_id)
        try:
            return [scrub_object(item, visited=visited) for item in obj]
        finally:
            visited.remove(obj_id)
    elif isinstance(obj, tuple):
        obj_id = id(obj)
        if obj_id in visited:
            return "<circular>"
        visited.add(obj_id)
        try:
            return tuple(scrub_object(item, visited=visited) for item in obj)
        finally:
            visited.remove(obj_id)
    elif isinstance(obj, set):
        obj_id = id(obj)
        if obj_id in visited:
            return "<circular>"
        visited.add(obj_id)
        try:
            return {scrub_object(item, visited=visited) for item in obj}
        finally:
            visited.remove(obj_id)
    return obj
