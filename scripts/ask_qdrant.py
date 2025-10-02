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

try:  # pragma: no cover - optional dependency surface
    from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
except Exception:  # pragma: no cover - fallback when qdrant_client is unavailable
    class ApiException(Exception):
        """Fallback API exception when qdrant_client is not installed."""

    class ResponseHandlingException(Exception):
        """Fallback response exception when qdrant_client is not installed."""

from cli.openai_client import OpenAIClientError, SharedOpenAIClient
from cli.sanitizer import scrub_object
from config.settings import OpenAISettings
from fancyrag.utils import ensure_env
from neo4j_graphrag.exceptions import (
    RetrieverInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever


_RETRIEVAL_QUERY = (
    "WITH node, score "
    "OPTIONAL MATCH (doc:Document)-[:HAS_CHUNK]->(node) "
    "RETURN node.chunk_id AS chunk_id, "
    "node.text AS text, "
    "node.source_path AS source_path, "
    "node.relative_path AS relative_path, "
    "node.git_commit AS git_commit, "
    "node.checksum AS checksum, "
    "node.chunk_index AS chunk_index, "
    "doc.name AS document_name, "
    "doc.source_path AS document_source_path, "
    "score"
)


def _load_settings() -> OpenAISettings:
    """
    Load OpenAI settings configured for the "ask_qdrant" actor.
    
    Returns:
        OpenAISettings: An OpenAISettings instance with the actor set to "ask_qdrant".
    """
    return OpenAISettings.load(actor="ask_qdrant")


def _record_to_match(record: Any) -> dict[str, Any]:
    """
    Normalize a retriever record into the CLI match dictionary.
    
    Converts a retriever record (an object with a data() method, a dict, or another mapping) into a plain dict suitable for output. If the record contains a `chunk_id` or `id`, ensures the resulting payload has a `chunk_id` value coerced to a string.
    
    Returns:
        dict[str, Any]: Normalized match payload with `chunk_id` as a string when present.
    """

    data_getter = getattr(record, "data", None)
    if callable(data_getter):
        payload = data_getter()
    elif isinstance(record, dict):
        payload = dict(record)
    else:  # pragma: no cover - defensive guard when record implements mapping protocol
        payload = dict(record)

    if "chunk_id" in payload and payload["chunk_id"] is not None:
        chunk_identifier = payload["chunk_id"]
    else:
        chunk_identifier = payload.get("id")

    if chunk_identifier is not None:
        payload["chunk_id"] = str(chunk_identifier)

    return payload


def main() -> None:
    """
    Run the CLI to embed a question, retrieve top-k chunk matches from Qdrant, enrich each match with Neo4j document context, and produce a sanitized JSON artifact.
    
    Reads CLI flags (--question, --top-k, --collection) and requires the environment variables OPENAI_API_KEY, QDRANT_URL, NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD (optionally NEO4J_DATABASE). Generates an embedding for the provided question, uses a Neo4j-backed retriever to obtain nearest chunks and associated document fields, normalizes match scores when possible, writes the sanitized result to artifacts/local_stack/ask_qdrant.json, and prints the sanitized JSON to stdout.
    
    Raises:
        SystemExit: Exits with status code 1 when an error occurs during embedding, retrieval, or enrichment.
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
    records: list[Any] = []

    try:
        embedding_result = client.embedding(input_text=args.question)
        query_vector = embedding_result.vector

        with GraphDatabase.driver(neo4j_uri, auth=neo4j_auth) as driver:
            retriever = QdrantNeo4jRetriever(
                driver=driver,
                client=qdrant_client,
                collection_name=args.collection,
                id_property_neo4j=os.environ.get("QDRANT_NEO4J_ID_PROPERTY_NEO4J", "chunk_id"),
                id_property_external=os.environ.get("QDRANT_NEO4J_ID_PROPERTY_EXTERNAL", "chunk_id"),
                neo4j_database=neo4j_database,
                retrieval_query=_RETRIEVAL_QUERY,
            )

            raw_result = retriever.get_search_results(
                query_vector=query_vector,
                top_k=max(1, args.top_k),
            )
            records = getattr(raw_result, "records", [])

        if not records:
            status = "skipped"
            message = "Qdrant returned no matches"
        else:
            for record in records:
                match = _record_to_match(record)
                if "score" in match and match["score"] is not None:
                    try:
                        match["score"] = float(match["score"])
                    except (TypeError, ValueError):  # pragma: no cover - defensive guard
                        pass
                matches.append(match)
    except OpenAIClientError as exc:
        status = "error"
        message = getattr(exc, "remediation", None) or str(exc)
        print(f"error: {exc}", file=sys.stderr)
    except (RetrieverInitializationError, SearchValidationError) as exc:
        status = "error"
        message = str(exc)
        print(f"error: {exc}", file=sys.stderr)
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
