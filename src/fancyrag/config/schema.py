"""Schema loading and validation helpers for FancyRAG."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from fancyrag.utils.paths import resolve_repo_root

__all__ = [
    "DEFAULT_SCHEMA",
    "DEFAULT_SCHEMA_FILENAME",
    "DEFAULT_SCHEMA_PATH",
    "GraphSchema",
    "load_default_schema",
    "load_schema",
    "resolve_schema_path",
]

DEFAULT_SCHEMA_FILENAME = "kg_schema.json"
_PROJECT_ROOT = resolve_repo_root() or Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_PATH = _PROJECT_ROOT / "scripts" / "config" / DEFAULT_SCHEMA_FILENAME


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


if _module_available("neo4j_graphrag.experimental.components.schema"):
    from neo4j_graphrag.experimental.components.schema import GraphSchema
else:
    class GraphSchema:  # type: ignore[too-few-public-methods]
        """Placeholder schema returned when neo4j_graphrag is unavailable."""

        @classmethod
        def model_validate(cls, *_args: Any, **_kwargs: Any) -> "GraphSchema":
            return cls()


def resolve_schema_path(path: str | Path | None = None) -> Path:
    """Return the filesystem path for the FancyRAG knowledge-graph schema."""

    if path is None:
        return DEFAULT_SCHEMA_PATH

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    repo_candidate = (_PROJECT_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return repo_candidate


def load_schema(path: str | Path) -> GraphSchema:
    """Load and validate a graph schema from the supplied path."""

    candidate = resolve_schema_path(path)
    descriptor = "default schema" if candidate == DEFAULT_SCHEMA_PATH else "schema"
    try:
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{descriptor} file not found: {candidate}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid {descriptor} JSON: {candidate}") from exc
    return GraphSchema.model_validate(raw)


def load_default_schema() -> GraphSchema:
    """Return the validated FancyRAG default graph schema."""

    return load_schema(DEFAULT_SCHEMA_PATH)


DEFAULT_SCHEMA = load_default_schema()
