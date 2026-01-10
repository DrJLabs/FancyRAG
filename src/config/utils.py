"""Shared configuration parsing helpers."""

from __future__ import annotations

TRUE_BOOL_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_BOOL_VALUES = frozenset({"0", "false", "no", "off"})


def allowed_boolean_values() -> frozenset[str]:
    return TRUE_BOOL_VALUES | FALSE_BOOL_VALUES


def format_allowed_boolean_values() -> str:
    return ", ".join(sorted(allowed_boolean_values()))


def parse_boolean(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in TRUE_BOOL_VALUES:
        return True
    if normalized in FALSE_BOOL_VALUES:
        return False
    raise ValueError(
        f"Invalid boolean value '{raw_value}'. Use one of: {format_allowed_boolean_values()}."
    )
