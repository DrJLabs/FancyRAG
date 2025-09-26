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
    "sanitize_mapping",
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


def sanitize_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """
    Create a shallow sanitized copy of a mapping with sensitive values redacted.
    
    For any key whose lowercased name is in SENSITIVE_KEY_NAMES, the corresponding value is replaced with "***".
    All other values are sanitized recursively (strings, mappings, lists, tuples are processed as needed).
    The original mapping is not modified; keys and their casing are preserved in the returned dict.
    
    Parameters:
        data (Mapping[str, Any]): Input mapping to sanitize.
    
    Returns:
        dict[str, Any]: A new dictionary with sensitive values redacted and other values sanitized.
    """

    result: dict[str, Any] = {}
    for key, value in data.items():
        lower_key = key.lower()
        if lower_key in SENSITIVE_KEY_NAMES:
            result[key] = "***"
            continue

        result[key] = _sanitize_value(value)
    return result


def _sanitize_value(value: Any) -> Any:
    """
    Recursively sanitize a value by redacting secrets in strings and nested structures.
    
    Parameters:
        value: The input to sanitize. Strings are processed with sanitize_text; Mapping objects are processed with sanitize_mapping; lists and tuples are traversed and each element is sanitized recursively.
    
    Returns:
        The sanitized value with secrets redacted. The return type matches the input structure (str, Mapping-derived dict, list, tuple, or the original object for unsupported types).
    """
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        return sanitize_mapping(value)
    if isinstance(value, list):
        return [
            sanitize_mapping(item)
            if isinstance(item, Mapping)
            else sanitize_text(item)
            if isinstance(item, str)
            else _sanitize_value(item)
            if isinstance(item, list)
            else item
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item) for item in value)
    return value


def scrub_object(obj: Any) -> Any:
    """
    Deeply sanitize an object for safe JSON/file serialization.
    
    Strings have secret-like substrings redacted. Mapping values, list items, and tuple elements are recursively sanitized; other values are returned unchanged.
    
    Returns:
        The input object with sensitive string content redacted and nested structures recursively sanitized.
    """

    if isinstance(obj, str):
        return sanitize_text(obj)
    if isinstance(obj, Mapping):
        sanitized = sanitize_mapping(obj)
        return {key: scrub_object(value) for key, value in sanitized.items()}
    if isinstance(obj, list):
        return [scrub_object(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(scrub_object(item) for item in obj)
    return obj
