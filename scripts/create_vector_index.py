#!/usr/bin/env python
"""Create or validate the Neo4j vector index used by the minimal path workflow.

Replaces the Story 2.4 stub with production-ready behaviour that connects to
Neo4j using shared environment settings, idempotently provisions the
``chunks_vec`` index (by default), and emits structured JSON logs compatible
with local smoke automation.
"""

from __future__ import annotations

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
from neo4j_graphrag.indexes import (
    Neo4jIndexError,
    create_vector_index,
    retrieve_vector_index_info,
)

from _compat.structlog import get_logger
from cli.sanitizer import scrub_object
from config.settings import DEFAULT_EMBEDDING_DIMENSIONS
from fancyrag.utils import ensure_env


logger = get_logger(__name__)

DEFAULT_INDEX_NAME = "chunks_vec"
DEFAULT_LABEL = "Chunk"
DEFAULT_PROPERTY = "embedding"
DEFAULT_SIMILARITY = "cosine"
DEFAULT_LOG_PATH = Path("artifacts/local_stack/create_vector_index.json")


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
        """Create a configuration snapshot from SHOW INDEXES output."""

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
    parser = argparse.ArgumentParser(description="Create or validate the Neo4j vector index used by the minimal path workflow.")
    parser.add_argument(
        "--index-name",
        default=os.environ.get("NEO4J_VECTOR_INDEX", DEFAULT_INDEX_NAME),
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
    if value <= 0:
        raise ValueError("dimensions must be a positive integer")
    return value


def _normalise_similarity(value: str) -> str:
    return value.strip().lower()


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _fetch_existing_config(driver, *, index_name: str, label: str, embedding_property: str, database: str | None) -> VectorIndexConfig | None:
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
    args = _parse_args(argv)
    args.dimensions = _validate_dimensions(args.dimensions)
    args.similarity = _normalise_similarity(args.similarity)

    ensure_env("NEO4J_URI")
    ensure_env("NEO4J_USERNAME")
    ensure_env("NEO4J_PASSWORD")

    uri = os.environ["NEO4J_URI"]
    auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])

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
                backoff = 0.25
                for attempt in range(3):
                    try:
                        _create_index(driver, cfg=cfg, database=args.database)
                        break
                    except (Neo4jIndexError, Neo4jError, ClientError) as exc:
                        if attempt == 2:
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
