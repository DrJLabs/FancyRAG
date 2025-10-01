#!/usr/bin/env python
"""Query Qdrant for the best-matching chunks and surface associated document context."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from qdrant_client import QdrantClient

from cli.openai_client import OpenAIClientError, SharedOpenAIClient
from cli.sanitizer import scrub_object
from config.settings import OpenAISettings
from fancyrag.utils import ensure_env


def _load_settings() -> OpenAISettings:
    """
    Load OpenAI settings configured for the "ask_qdrant" actor.
    
    Returns:
        OpenAISettings: An OpenAISettings instance with the actor set to "ask_qdrant".
    """
    return OpenAISettings.load(actor="ask_qdrant")


def _query_qdrant(
    client: QdrantClient,
    *,
    collection: str,
    vector: list[float],
    limit: int,
) -> list[Any]:
    """
    Query a Qdrant collection for the nearest points to a provided embedding vector.
    
    Parameters:
        client (QdrantClient): Qdrant client used to perform the query.
        collection (str): Name of the Qdrant collection to search.
        vector (list[float]): Embedding vector used as the query.
        limit (int): Maximum number of matching points to return.
    
    Returns:
        list[Any]: A list of matching point objects (including payload) from Qdrant.
    """
    try:
        response = client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return list(response.points)
    except AttributeError:
        return client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )


def _fetch_chunk_context(driver, *, chunk_id: str, database: str | None) -> dict[str, Any]:
    """
    Retrieve stored text and document metadata for a chunk identified by `chunk_id` from Neo4j.
    
    Parameters:
    	chunk_id (str): The chunk identifier to look up.
    	database (str | None): Optional Neo4j database name to execute the query against.
    
    Returns:
    	context (dict[str, Any]): A mapping with at least `chunk_id`. When a matching chunk is found, includes:
    		- `chunk_id` (str)
    		- `text` (str): chunk text content
    		- `source_path` (str): path or source of the chunk
    		- `document_name` (str, optional): name of the parent document when available
    		- `document_source_path` (str, optional): source path of the parent document when available
    	If no matching node exists, returns `{"chunk_id": chunk_id}`.
    """
    records, _, _ = driver.execute_query(
        """
        MATCH (chunk:Chunk {chunk_id: $chunk_id})
        OPTIONAL MATCH (doc:Document)-[:HAS_CHUNK]->(chunk)
        RETURN chunk.chunk_id AS chunk_id,
               chunk.text AS text,
               chunk.source_path AS source_path,
               doc.name AS document_name,
               doc.source_path AS document_source_path
        LIMIT 1
        """,
        {"chunk_id": chunk_id},
        database_=database,
    )
    return dict(records[0]) if records else {"chunk_id": chunk_id}


def main() -> None:
    """
    Run a CLI that queries Qdrant for top-k chunk matches and enriches each match with Neo4j document context.
    
    Accepts command-line flags (--question, --top-k, --collection), reads required environment variables for OpenAI, Qdrant, and Neo4j, obtains an embedding for the provided question, queries Qdrant for nearest chunks, fetches associated chunk/document context from Neo4j, and produces a sanitized JSON result. The function writes the result to artifacts/local_stack/ask_qdrant.json, prints the sanitized JSON to stdout, and raises SystemExit(1) if an error occurs.
    """
    parser = argparse.ArgumentParser(description="Query Qdrant for chunk matches")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", default="chunks_main")
    args = parser.parse_args()

    ensure_env("OPENAI_API_KEY")
    ensure_env("QDRANT_URL")
    ensure_env("NEO4J_URI")
    ensure_env("NEO4J_USERNAME")
    ensure_env("NEO4J_PASSWORD")

    settings = _load_settings()
    client = SharedOpenAIClient(settings)

    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    neo4j_uri = os.environ["NEO4J_URI"]
    neo4j_auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    neo4j_database = os.environ.get("NEO4J_DATABASE")

    start = time.perf_counter()
    status = "success"
    message = ""
    matches: list[dict[str, Any]] = []

    try:
        embedding_result = client.embedding(input_text=args.question)
        query_vector = embedding_result.vector

        points = _query_qdrant(
            qdrant_client,
            collection=args.collection,
            vector=query_vector,
            limit=max(1, args.top_k),
        )
        if not points:
            status = "skipped"
            message = "Qdrant returned no matches"
        else:
            with GraphDatabase.driver(neo4j_uri, auth=neo4j_auth) as driver:
                for point in points:
                    payload = point.payload or {}
                    chunk_id = payload.get("chunk_id") or str(point.id)
                    context = _fetch_chunk_context(driver, chunk_id=chunk_id, database=neo4j_database)
                    context.update(
                        {
                            "score": point.score,
                            "chunk_id": chunk_id,
                        }
                    )
                    matches.append(context)
    except OpenAIClientError as exc:
        status = "error"
        message = getattr(exc, "remediation", None) or str(exc)
        print(f"error: {exc}", file=sys.stderr)
    except (Neo4jError, Exception) as exc:  # pragma: no cover - defensive guard
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        status = "error"
        message = str(exc)
        print(f"error: {exc}", file=sys.stderr)

    duration_ms = int((time.perf_counter() - start) * 1000)
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "ask_qdrant",
        "question": args.question,
        "top_k": args.top_k,
        "status": status,
        "message": message or f"Retrieved {len(matches)} matches",
        "matches": matches,
        "duration_ms": duration_ms,
        "collection": args.collection,
    }

    artifacts_dir = Path("artifacts/local_stack")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sanitized = scrub_object(log)
    (artifacts_dir / "ask_qdrant.json").write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
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
