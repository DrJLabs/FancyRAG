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
from qdrant_client import models as qmodels

from cli.sanitizer import scrub_object
from fancyrag.utils import ensure_env


def _fetch_chunks(driver, *, database: str | None) -> list[dict[str, Any]]:
    """
    Retrieve chunk records from Neo4j including chunk_id, chunk_index, text, embedding, and source_path.
    
    Parameters:
        database (str | None): Optional Neo4j database name to execute the query against; if None the driver's default database is used.
    
    Returns:
        list[dict[str, Any]]: A list of records where each dictionary contains the keys
        'chunk_id', 'chunk_index', 'text', 'embedding', and 'source_path'.
    """

    records, _, _ = driver.execute_query(
        """
        MATCH (chunk:Chunk)
        RETURN chunk.chunk_id AS chunk_id,
               chunk.index AS chunk_index,
               chunk.text AS text,
               chunk.embedding AS embedding,
               chunk.source_path AS source_path
        ORDER BY chunk_index ASC
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
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return str(value)


def main() -> None:
    """
    Orchestrate export of chunk embeddings from Neo4j into a Qdrant collection.
    
    Connects to Neo4j using environment credentials, fetches chunk records (including text and embedding),
    creates or replaces the target Qdrant collection configured for the embedding dimensionality,
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
                if client.collection_exists(args.collection):
                    client.delete_collection(args.collection)
                client.create_collection(
                    collection_name=args.collection,
                    vectors_config=qmodels.VectorParams(
                        size=dimensions,
                        distance=qmodels.Distance.COSINE,
                    ),
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
                                "chunk_index": record.get("chunk_index"),
                                "source_path": record.get("source_path"),
                                "text": record.get("text"),
                            }
                        )
                    client.upsert(
                        collection_name=args.collection,
                        points=qmodels.Batch(ids=ids, vectors=vectors, payloads=payloads),
                    )
                    exported += len(batch)

    except (Neo4jError, Exception) as exc:  # pragma: no cover - defensive guard
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
