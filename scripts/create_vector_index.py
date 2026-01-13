#!/usr/bin/env python
"""Create or validate the Neo4j vector index used by the minimal path workflow.

Replaces the Story 2.4 stub with production-ready behaviour that connects to
Neo4j using shared environment settings, idempotently provisions the configured
vector index, and emits structured JSON logs compatible with local smoke
automation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, Neo4jError
try:  # pragma: no branch - optional dependency guard
    from neo4j_graphrag.indexes import (
        Neo4jIndexError,
        create_vector_index,
        retrieve_vector_index_info,
    )
except Exception as exc:  # pragma: no cover - dependency missing in minimal environments
    Neo4jIndexError = RuntimeError  # type: ignore[assignment]
    _GRAPHRAG_IMPORT_ERROR = exc

    def _require_graphrag(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "neo4j_graphrag is required for vector index operations; install the neo4j-graphrag extra"
        ) from _GRAPHRAG_IMPORT_ERROR

    def create_vector_index(*args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return _require_graphrag(*args, **kwargs)

    def retrieve_vector_index_info(*args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return _require_graphrag(*args, **kwargs)

from _compat.structlog import get_logger
from cli.sanitizer import scrub_object
from config.settings import DEFAULT_EMBEDDING_DIMENSIONS
from fancyrag.utils import get_settings


logger = get_logger(__name__)

DEFAULT_INDEX_NAME = "chunks_vec"
DEFAULT_LABEL = "Chunk"
DEFAULT_PROPERTY = "embedding"
DEFAULT_SIMILARITY = "cosine"
DEFAULT_LOG_PATH = Path("artifacts/local_stack/create_vector_index.json")
INITIAL_BACKOFF_SECONDS = 0.25
MAX_CREATE_RETRIES = 3


class VectorIndexMismatchError(RuntimeError):
    """Raised when an existing index does not match the requested configuration."""


@dataclass(frozen=True)
class VectorIndexConfig:
    """Normalised configuration for a Neo4j vector index."""

    name: str
    label: str
    embedding_property: str
    dimensions: int
    similarity: str

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "VectorIndexConfig":
        """
        Constructs a VectorIndexConfig from a Neo4j SHOW INDEXES-style record.
        
        Parameters:
            record (Mapping[str, Any]): A mapping produced by `SHOW INDEXES` (or similar)
                which may contain the keys `name`, `labelsOrTypes`, `properties`, and
                `options` (including `indexConfig`).
        
        Returns:
            VectorIndexConfig: Configuration populated from the record. Behavior:
            - `name` is taken from `record["name"]` (converted to string).
            - `label` and `embedding_property` use the first element of
              `labelsOrTypes` and `properties` respectively, or an empty string if absent.
            - `dimensions` is parsed from `indexConfig["vector.dimensions"]` as an int;
              if parsing fails or the value is missing, defaults to DEFAULT_EMBEDDING_DIMENSIONS.
            - `similarity` is taken from `indexConfig["vector.similarity_function"]`
              and normalized to a lowercase string (empty string if missing).
        """

        labels = list(record.get("labelsOrTypes") or [])
        properties = list(record.get("properties") or [])
        options = record.get("options") or {}
        config = options.get("indexConfig") or {}

        dimensions_raw = config.get("vector.dimensions")
        similarity_raw = config.get("vector.similarity_function")

        try:
            dimensions = int(dimensions_raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            dimensions = DEFAULT_EMBEDDING_DIMENSIONS

        return cls(
            name=str(record.get("name")),
            label=str(labels[0]) if labels else "",
            embedding_property=str(properties[0]) if properties else "",
            dimensions=dimensions,
            similarity=str(similarity_raw or "").lower(),
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for creating or validating the Neo4j vector index.
    
    Parameters:
        argv (Sequence[str] | None): Optional list of argument strings to parse; when None the function reads from the process arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - index_name: Vector index name.
            - label: Target node label for the index.
            - embedding_property: Node property holding the embedding vectors.
            - dimensions: Expected embedding dimensionality (int).
            - similarity: Similarity function to configure (`"cosine"` or `"euclidean"`).
            - database: Optional Neo4j database name or None to use the server default.
            - log_path: Filesystem path for the structured JSON log.
    """
    parser = argparse.ArgumentParser(description="Create or validate the Neo4j vector index used by the minimal path workflow.")
    parser.add_argument(
        "--index-name",
        default=os.environ.get("INDEX_NAME")
        or os.environ.get("NEO4J_VECTOR_INDEX", DEFAULT_INDEX_NAME),
        help="Vector index name (default: %(default)s)",
    )
    parser.add_argument(
        "--label",
        default=os.environ.get("NEO4J_CHUNK_LABEL", DEFAULT_LABEL),
        help="Node label the index targets (default: %(default)s)",
    )
    parser.add_argument(
        "--embedding-property",
        default=os.environ.get("NEO4J_CHUNK_EMBEDDING_PROPERTY", DEFAULT_PROPERTY),
        help="Embedding property to index (default: %(default)s)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSIONS,
        help="Expected embedding dimensions (default: %(default)s)",
    )
    parser.add_argument(
        "--similarity",
        choices=("cosine", "euclidean"),
        default=DEFAULT_SIMILARITY,
        help="Similarity function to configure for the vector index (default: %(default)s)",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("NEO4J_DATABASE"),
        help="Optional Neo4j database name (defaults to server default)",
    )
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
        help="Location for the structured JSON log (default: %(default)s)",
    )
    return parser.parse_args(argv)


def _validate_dimensions(value: int) -> int:
    """
    Validate that the embedding dimensionality is greater than zero.
    
    Parameters:
        value (int): The number of embedding dimensions to validate.
    
    Returns:
        int: The same `value` if it is greater than zero.
    
    Raises:
        ValueError: If `value` is less than or equal to zero.
    """
    if value <= 0:
        raise ValueError("dimensions must be a positive integer")
    return value


def _normalise_similarity(value: str) -> str:
    """
    Normalize a similarity identifier for index configuration.
    
    Parameters:
        value (str): Similarity name (e.g., "cosine", "euclidean") possibly with surrounding whitespace or mixed case.
    
    Returns:
        str: The input trimmed of surrounding whitespace and converted to lowercase.
    """
    return value.strip().lower()


def _ensure_directory(path: Path) -> None:
    """
    Ensure the parent directory of the given path exists, creating any missing parent directories.
    
    If the parent directory already exists this function does nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _fetch_existing_config(driver, *, index_name: str, label: str, embedding_property: str, database: str | None) -> VectorIndexConfig | None:
    """
    Retrieve the existing vector index configuration for the given index, label, and embedding property from Neo4j.
    
    Returns:
        VectorIndexConfig | None: A VectorIndexConfig built from the index record if found, `None` if no matching index exists.
    """
    record = retrieve_vector_index_info(
        driver,
        index_name=index_name,
        label_or_type=label,
        embedding_property=embedding_property,
        neo4j_database=database,
    )
    if record is None:
        return None
    return VectorIndexConfig.from_record(record.data() if hasattr(record, "data") else record)


def _compare_configs(requested: VectorIndexConfig, existing: VectorIndexConfig) -> None:
    """
    Verify that an existing VectorIndexConfig matches the requested VectorIndexConfig and raise if any field differs.
    
    Compares the `label`, `embedding_property`, `dimensions`, and `similarity` fields of `existing` against `requested`. If any values differ, raises VectorIndexMismatchError with a message that lists each differing field showing the current and expected values.
    
    Parameters:
        requested (VectorIndexConfig): Desired index configuration.
        existing (VectorIndexConfig): Currently observed index configuration.
    
    Raises:
        VectorIndexMismatchError: If one or more fields differ between `existing` and `requested`. The exception message includes details for each mismatched field.
    """
    mismatches: dict[str, tuple[Any, Any]] = {}
    if existing.label != requested.label:
        mismatches["label"] = (existing.label, requested.label)
    if existing.embedding_property != requested.embedding_property:
        mismatches["embedding_property"] = (existing.embedding_property, requested.embedding_property)
    if existing.dimensions != requested.dimensions:
        mismatches["dimensions"] = (existing.dimensions, requested.dimensions)
    if existing.similarity != requested.similarity:
        mismatches["similarity"] = (existing.similarity, requested.similarity)
    if mismatches:
        details = ", ".join(
            f"{key} existing={current!r} expected={desired!r}"
            for key, (current, desired) in mismatches.items()
        )
        raise VectorIndexMismatchError(
            f"Existing vector index configuration does not match requested settings: {details}",
        )


def _create_index(driver, *, cfg: VectorIndexConfig, database: str | None) -> None:
    """
    Create the vector index in the specified Neo4j database using the provided configuration.
    
    Parameters:
        cfg (VectorIndexConfig): Normalized index configuration (name, label, embedding_property, dimensions, similarity).
        database (str | None): Target Neo4j database name, or `None` to use the driver's default.
    """
    create_vector_index(
        driver,
        cfg.name,
        label=cfg.label,
        embedding_property=cfg.embedding_property,
        dimensions=cfg.dimensions,
        similarity_fn=cfg.similarity,
        neo4j_database=database,
    )


def _build_log(*, status: str, cfg: VectorIndexConfig, existing: VectorIndexConfig | None, duration_ms: int, database: str | None) -> dict[str, Any]:
    """
    Builds a structured log dictionary describing the vector index operation.
    
    Parameters:
        status (str): Outcome of the operation (e.g., "created", "exists", "mismatch", "runtime_error").
        cfg (VectorIndexConfig): Requested/target index configuration included under the `index` key.
        existing (VectorIndexConfig | None): Existing index configuration to include under the `existing` key when present.
        duration_ms (int): Operation duration in milliseconds.
        database (str | None): Neo4j database name associated with the operation.
    
    Returns:
        dict[str, Any]: A log dictionary containing:
            - timestamp: UTC ISO timestamp of the log record
            - operation: always "create_vector_index"
            - status: provided status
            - duration_ms: provided duration in milliseconds
            - database: provided database name or None
            - index: snapshot of the requested index configuration
            - existing: snapshot of the existing index configuration if present
    """
    log: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "create_vector_index",
        "status": status,
        "duration_ms": duration_ms,
        "database": database,
        "index": {
            "name": cfg.name,
            "label": cfg.label,
            "embedding_property": cfg.embedding_property,
            "dimensions": cfg.dimensions,
            "similarity": cfg.similarity,
        },
    }
    if existing is not None:
        log["existing"] = {
            "name": existing.name,
            "label": existing.label,
            "embedding_property": existing.embedding_property,
            "dimensions": existing.dimensions,
            "similarity": existing.similarity,
        }
    return log


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Ensure a Neo4j vector index exists (create it if missing) and produce a structured operation log.
    
    Parameters:
        argv (Sequence[str] | None): Optional command-line arguments to override defaults. If None, environment variables and default CLI settings are used.
    
    Returns:
        dict[str, Any]: A structured log dictionary describing the operation, including keys such as
            - timestamp: ISO 8601 UTC time of completion
            - operation: the performed operation (e.g., "create_vector_index")
            - status: "created" or "exists"
            - duration_ms: elapsed time in milliseconds
            - database: target Neo4j database name (or None)
            - index: details of the requested index (name, label, embedding_property, dimensions, similarity)
            - existing: snapshot of the existing index configuration when present
    
    Raises:
        VectorIndexMismatchError: If an existing index is found but its configuration does not match the requested configuration.
        RuntimeError: For Neo4j-related errors or if index creation fails after retries.
    """
    args = _parse_args(argv)
    args.dimensions = _validate_dimensions(args.dimensions)
    args.similarity = _normalise_similarity(args.similarity)

    neo4j_settings = get_settings(require={"neo4j"}).neo4j
    uri = neo4j_settings.uri
    auth = neo4j_settings.auth()

    cfg = VectorIndexConfig(
        name=args.index_name,
        label=args.label,
        embedding_property=args.embedding_property,
        dimensions=args.dimensions,
        similarity=args.similarity,
    )

    start = time.perf_counter()
    status = "created"
    existing_cfg: VectorIndexConfig | None = None

    try:
        with GraphDatabase.driver(uri, auth=auth) as driver:
            existing_cfg = _fetch_existing_config(
                driver,
                index_name=cfg.name,
                label=cfg.label,
                embedding_property=cfg.embedding_property,
                database=args.database,
            )

            if existing_cfg is not None:
                _compare_configs(cfg, existing_cfg)
                status = "exists"
            else:
                backoff = INITIAL_BACKOFF_SECONDS
                for attempt in range(MAX_CREATE_RETRIES):
                    try:
                        _create_index(driver, cfg=cfg, database=args.database)
                        break
                    except (Neo4jIndexError, Neo4jError, ClientError) as exc:
                        if attempt == MAX_CREATE_RETRIES - 1:
                            raise RuntimeError(f"Neo4j error: {exc}") from exc
                        time.sleep(backoff)
                        backoff *= 2
                existing_cfg = _fetch_existing_config(
                    driver,
                    index_name=cfg.name,
                    label=cfg.label,
                    embedding_property=cfg.embedding_property,
                    database=args.database,
                )
    except VectorIndexMismatchError:
        raise
    except (Neo4jIndexError, Neo4jError, ClientError) as exc:
        raise RuntimeError(f"Neo4j error: {exc}") from exc

    duration_ms = int((time.perf_counter() - start) * 1000)
    log = _build_log(status=status, cfg=cfg, existing=existing_cfg, duration_ms=duration_ms, database=args.database)

    output_path = Path(args.log_path)
    _ensure_directory(output_path)
    sanitized = scrub_object(log)
    output_path.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    print(json.dumps(sanitized))
    logger.info("vector_index.completed", **sanitized)
    return log


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the create_vector_index workflow and return an appropriate process exit code.
    
    Parameters:
    	argv (Sequence[str] | None): Command-line arguments to pass to the workflow; pass None to use defaults.
    
    Returns:
    	exit_code (int): 0 on success, 1 if an error occurred.
    """
    try:
        run(argv)
        return 0
    except VectorIndexMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        logger.error("vector_index.mismatch", error=str(exc))
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        logger.error("vector_index.runtime_error", error=str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - safety net
        print(f"error: {exc}", file=sys.stderr)
        logger.exception("vector_index.failed", error=str(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
