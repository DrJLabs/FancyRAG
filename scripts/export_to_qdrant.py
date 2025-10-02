#!/usr/bin/env python
"""Export chunk embeddings from Neo4j into a Qdrant collection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from qdrant_client import QdrantClient
import qdrant_client.models as qmodels

try:  # pragma: no cover - optional dependency surface
    from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
except Exception:  # pragma: no cover - fallback when exceptions module unavailable
    class ApiException(Exception):
        """Fallback API exception when qdrant_client is not installed."""

    class ResponseHandlingException(Exception):
        """Fallback response exception when qdrant_client is not installed."""

from cli.sanitizer import scrub_object
from fancyrag.utils import ensure_env


def _fetch_chunks(driver, *, database: str | None) -> list[dict[str, Any]]:
    """
    Retrieve chunk records from Neo4j including vector payload and metadata.

    Parameters:
        database (str | None): Optional Neo4j database name to execute the query against; if None the driver's default database is used.

    Returns:
        list[dict[str, Any]]: A list of records where each dictionary contains the keys
        'chunk_id', 'chunk_index', 'text', 'embedding', 'source_path', 'relative_path',
        'git_commit', and 'checksum'.
    """

    records, _, _ = driver.execute_query(
        """
        MATCH (chunk:Chunk)
        WHERE chunk.embedding IS NOT NULL AND size(chunk.embedding) > 0
        RETURN coalesce(chunk.chunk_id, chunk.uid) AS chunk_id,
               chunk.uid AS chunk_uid,
               chunk.index AS chunk_index,
               chunk.text AS text,
               chunk.embedding AS embedding,
               chunk.source_path AS source_path,
               chunk.relative_path AS relative_path,
               chunk.git_commit AS git_commit,
               chunk.checksum AS checksum
        ORDER BY chunk.index ASC
        """,
        database_=database,
    )
    return [dict(record) for record in records]


def _batched(iterable: Iterable[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    """
    Yield successive batches of items from `iterable` as lists of up to `size` elements.
    
    Parameters:
        iterable (Iterable[dict[str, Any]]): Source sequence of chunk dictionaries.
        size (int): Maximum number of items per yielded batch; must be >= 1.
    
    Returns:
        Iterable[list[dict[str, Any]]]: Consecutive lists containing up to `size` items from `iterable`. The final batch may contain fewer than `size` items.
    """
    batch: list[dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _coerce_point_id(value: Any, fallback: int) -> int | str:
    """
    Normalize a chunk identifier into an integer or string suitable for use as a Qdrant point id.
    
    Parameters:
        value (Any): The original chunk identifier; may be None, int, str, or any other type.
        fallback (int): Value to use when `value` is None.
    
    Returns:
        int | str: `fallback` if `value` is None; the original int if `value` is an int; if `value` is a string, the trimmed string converted to an int when it contains only digits, otherwise the trimmed string; for other types, the string representation of `value`.
    """

    if value is None:
        return fallback
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return str(value)


def _resolve_remote_vector_size(info: qmodels.CollectionInfo | None) -> int | None:
    """Extract the configured vector size from an existing collection, if available."""

    if not info:
        return None
    config = getattr(info, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    if hasattr(vectors, "size"):
        return getattr(vectors, "size")
    if isinstance(vectors, dict):
        for value in vectors.values():
            if hasattr(value, "size"):
                return getattr(value, "size")
    return None


def main() -> None:
    """
    Orchestrate export of chunk embeddings from Neo4j into a Qdrant collection.
    
    Connects to Neo4j using environment credentials, fetches chunk records (including text and embedding),
    ensures the target Qdrant collection exists (optionally recreating it when --recreate-collection is passed)
    and upserts points in batches. Writes a sanitized JSON log artifact to artifacts/local_stack/export_to_qdrant.json
    and prints the same sanitized log to stdout.
    
    Environment requirements:
    - QDRANT_URL, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD must be set. Optional: QDRANT_API_KEY, NEO4J_DATABASE.
    
    Raises:
        RuntimeError: If chunk embeddings are missing or empty.
        SystemExit: Exits with status 1 when the export operation fails.
        Exceptions propagated from Neo4j/Qdrant clients or filesystem operations may also be raised.
    """

    parser = argparse.ArgumentParser(description="Export embeddings to Qdrant")
    parser.add_argument("--collection", default="chunks_main")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate the target Qdrant collection before export (destructive).",
    )
    args = parser.parse_args()

    ensure_env("QDRANT_URL")
    ensure_env("NEO4J_URI")
    ensure_env("NEO4J_USERNAME")
    ensure_env("NEO4J_PASSWORD")

    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    neo4j_uri = os.environ["NEO4J_URI"]
    neo4j_auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    neo4j_database = os.environ.get("NEO4J_DATABASE")

    start = time.perf_counter()
    status = "success"
    message = ""
    exported = 0

    try:
        with GraphDatabase.driver(neo4j_uri, auth=neo4j_auth) as driver:
            chunks = _fetch_chunks(driver, database=neo4j_database)
            if not chunks:
                status = "skipped"
                message = "No chunk nodes available to export"
            else:
                embedding = chunks[0].get("embedding")
                if embedding is None:
                    raise RuntimeError("Chunk embeddings missing or empty")
                if not isinstance(embedding, (list, tuple)):
                    raise RuntimeError("Chunk embeddings must be provided as a sequence")
                if not embedding:
                    raise RuntimeError("Chunk embeddings missing or empty")
                dimensions = len(embedding)

                client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                vector_params = qmodels.VectorParams(
                    size=dimensions,
                    distance=qmodels.Distance.COSINE,
                )

                collection_exists = client.collection_exists(args.collection)
                if args.recreate_collection:
                    if collection_exists:
                        client.delete_collection(args.collection)
                    client.create_collection(
                        collection_name=args.collection,
                        vectors_config=vector_params,
                    )
                else:
                    if not collection_exists:
                        client.create_collection(
                            collection_name=args.collection,
                            vectors_config=vector_params,
                        )
                    else:
                        info = client.get_collection(args.collection)
                        remote_size = _resolve_remote_vector_size(info)
                        if remote_size is not None and remote_size != dimensions:
                            raise RuntimeError(
                                "Existing collection '"
                                f"{args.collection}' uses vector dimension {remote_size}; expected {dimensions}. "
                                "Rerun with --recreate-collection to rebuild the collection."
                            )

                for batch in _batched(chunks, max(1, args.batch_size)):
                    payloads = []
                    ids = []
                    vectors = []
                    for idx, record in enumerate(batch):
                        fallback_id = exported + idx + 1
                        chunk_id = _coerce_point_id(record.get("chunk_id"), fallback=fallback_id)
                        embedding = record.get("embedding")
                        if embedding is None:
                            raise RuntimeError(f"Chunk {chunk_id} is missing its embedding")
                        if not isinstance(embedding, (list, tuple)):
                            raise RuntimeError(f"Chunk {chunk_id} embedding must be a sequence")
                        if len(embedding) != dimensions:
                            raise RuntimeError(
                                f"Chunk {chunk_id} embedding length {len(embedding)} does not match expected dimensions {dimensions}"
                            )
                        ids.append(chunk_id)
                        vectors.append(list(embedding))
                        payloads.append(
                            {
                                "chunk_id": chunk_id,
                                "chunk_uid": record.get("chunk_uid"),
                                "chunk_index": record.get("chunk_index"),
                                "source_path": record.get("source_path"),
                                "relative_path": record.get("relative_path"),
                                "git_commit": record.get("git_commit"),
                                "checksum": record.get("checksum"),
                                "text": record.get("text"),
                            }
                        )
                    client.upsert(
                        collection_name=args.collection,
                        points=qmodels.Batch(ids=ids, vectors=vectors, payloads=payloads),
                    )
                    exported += len(batch)

    except Neo4jError as exc:  # pragma: no cover - defensive guard
        status = "error"
        message = str(exc)
        print(f"error: {exc}", file=sys.stderr)
    except (ApiException, ResponseHandlingException) as exc:  # pragma: no cover - defensive guard
        status = "error"
        message = str(exc)
        print(f"error: {exc}", file=sys.stderr)
    except (RuntimeError, ValueError, TypeError) as exc:  # pragma: no cover - defensive guard
        status = "error"
        message = str(exc)
        print(f"error: {exc}", file=sys.stderr)
    else:
        if not message:
            message = f"Exported {exported} chunks"

    duration_ms = int((time.perf_counter() - start) * 1000)
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "export_to_qdrant",
        "collection": args.collection,
        "status": status,
        "message": message,
        "count": exported,
        "duration_ms": duration_ms,
    }

    artifacts_dir = Path("artifacts/local_stack")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sanitized = scrub_object(log)
    (artifacts_dir / "export_to_qdrant.json").write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    print(json.dumps(sanitized))

    if status == "error":
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
