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
# When the repository root cannot be resolved (e.g., editable installs), fall back to
# walking up three directories from this file. Update this assumption if the module moves.
_PROJECT_ROOT = resolve_repo_root() or Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_PATH = _PROJECT_ROOT / "scripts" / "config" / DEFAULT_SCHEMA_FILENAME


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


try:  # pragma: no cover - exercised indirectly via tests
    if not _module_available("neo4j_graphrag.experimental.components.schema"):
        raise ImportError
    from neo4j_graphrag.experimental.components.schema import GraphSchema as _RealGraphSchema  # type: ignore

    try:
        _RealGraphSchema.model_validate({})
    except TypeError as exc:  # pydantic validator incompatible in this environment
        raise ImportError from exc
    except Exception:
        # Validation failures are expected with dummy input; ignore.
        pass

    GraphSchema = _RealGraphSchema
except Exception:  # pragma: no cover - fallback path validated in unit tests
    class GraphSchema:  # type: ignore[too-few-public-methods]
        """Placeholder schema returned when neo4j_graphrag is unavailable."""

        @classmethod
        def model_validate(cls, *_args: Any, **_kwargs: Any) -> "GraphSchema":
            return cls()


def resolve_schema_path(path: str | Path | None = None) -> Path:
    """Return the filesystem path for the FancyRAG knowledge-graph schema.

    Raises:
        FileNotFoundError: If the schema path cannot be resolved.
    """

    if path is None:
        if not DEFAULT_SCHEMA_PATH.exists():
            raise FileNotFoundError(f"default schema file does not exist: {DEFAULT_SCHEMA_PATH}")
        return DEFAULT_SCHEMA_PATH

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"absolute schema path does not exist: {candidate}")
        return candidate

    repo_candidate = (_PROJECT_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # Fall back to the repository candidate even when the file is absent so
    # callers can surface deterministic locations or create the file later.
    return repo_candidate


def load_schema(path: str | Path | None) -> GraphSchema:
    """Load and validate a graph schema from the supplied path."""

    try:
        candidate = resolve_schema_path(path)
    except FileNotFoundError:
        requested = DEFAULT_SCHEMA_PATH if path is None else Path(path).expanduser()
        descriptor = "default schema" if path is None else "schema"
        raise RuntimeError(f"{descriptor} file not found: {requested}") from None

    descriptor = "default schema" if candidate == DEFAULT_SCHEMA_PATH else "schema"
    try:
        raw_text = candidate.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise RuntimeError(f"{descriptor} file not found: {candidate}") from None

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid {descriptor} JSON: {candidate}") from exc
    return GraphSchema.model_validate(raw)


def load_default_schema() -> GraphSchema:
    """Return the validated FancyRAG default graph schema."""

    return load_schema(None)


# Load the project's default schema at import time so callers can re-export a
# validated GraphSchema instance without repeating disk I/O. Existing tests and
# modules expect this constant to be available via `fancyrag.config`.
DEFAULT_SCHEMA = load_default_schema()
