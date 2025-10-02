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


SEMANTIC_SOURCE = "kg_build.semantic_enrichment.v1"


_RETRIEVAL_QUERY = (
    "WITH node, score "
    "OPTIONAL MATCH (doc:Document)-[:HAS_CHUNK]->(node) "
    "RETURN node.chunk_id AS chunk_id, "
    "coalesce(node.chunk_uid, node.uid) AS chunk_uid, "
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

    if "chunk_uid" in payload and payload["chunk_uid"] is not None:
        payload["chunk_uid"] = str(payload["chunk_uid"])

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
    parser.add_argument(
        "--include-semantic",
        action="store_true",
        help="Include semantic enrichment nodes and relationships in the output.",
    )
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

    semantic_context: dict[str, dict[str, Any]] = {}
    semantic_chunk_uids: dict[str, None] = {}

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
                chunk_uid = match.get("chunk_uid")
                if isinstance(chunk_uid, str) and chunk_uid:
                    semantic_chunk_uids.setdefault(chunk_uid, None)
                matches.append(match)
            if args.include_semantic and semantic_chunk_uids:
                with GraphDatabase.driver(neo4j_uri, auth=neo4j_auth) as driver:
                    semantic_context = _fetch_semantic_context(
                        driver,
                        database=neo4j_database,
                        chunk_uids=list(semantic_chunk_uids.keys()),
                    )
                for match in matches:
                    chunk_uid = match.get("chunk_uid")
                    context = semantic_context.get(chunk_uid) if isinstance(chunk_uid, str) else None
                    if context:
                        match["semantic"] = context
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
    except Exception as exc:  # pragma: no cover - final safety net for unexpected errors
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


def _fetch_semantic_context(
    driver,
    *,
    database: str | None,
    chunk_uids: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch semantic enrichment nodes and relationships for the specified chunk IDs."""

    if not chunk_uids:
        return {}

    query = """
    MATCH (entity:__Entity__)
    WHERE entity.semantic_source = $source AND entity.chunk_uid IN $chunk_uids
    OPTIONAL MATCH (entity)-[rel {semantic_source: $source}]-(target:__Entity__)
    WITH entity.chunk_uid AS chunk_uid,
         collect(DISTINCT {
             id: coalesce(entity.id, elementId(entity)),
             element_id: elementId(entity),
             labels: labels(entity),
             properties: entity
         }) AS entity_nodes,
         collect(DISTINCT CASE
             WHEN target IS NULL THEN NULL
             ELSE {
                 id: coalesce(target.id, elementId(target)),
                 element_id: elementId(target),
                 labels: labels(target),
                 properties: target
             }
         END) AS related_nodes,
         collect(DISTINCT CASE
             WHEN rel IS NULL THEN NULL
             ELSE {
                 type: type(rel),
                 start: elementId(startNode(rel)),
                 end: elementId(endNode(rel)),
                 properties: rel
             }
         END) AS relationship_entries
    RETURN chunk_uid,
           entity_nodes,
           [node IN related_nodes WHERE node IS NOT NULL] AS related_nodes,
           [rel IN relationship_entries WHERE rel IS NOT NULL] AS relationships
    """

    result = driver.execute_query(
        query,
        {"chunk_uids": chunk_uids, "source": SEMANTIC_SOURCE},
        database_=database,
    )
    records = result[0] if isinstance(result, tuple) else result
    context: dict[str, dict[str, Any]] = {}
    for record in records or []:
        chunk_uid = record.get("chunk_uid")
        if not chunk_uid:
            continue

        entity_nodes = record.get("entity_nodes", [])
        related_nodes = record.get("related_nodes", [])
        node_entries = [*entity_nodes, *related_nodes]
        relationship_entries = [
            entry
            for entry in record.get("relationships", [])
            if entry
        ]

        nodes_payload = []
        seen_node_ids: set[str] = set()
        for node_entry in node_entries:
            if not node_entry:
                continue
            node_dict = dict(node_entry)
            node_id = node_dict.get("id")
            if node_id is not None:
                node_id = str(node_id)
            element_id = node_dict.get("element_id")
            if element_id is not None:
                element_id = str(element_id)
            dedupe_key = element_id or node_id
            if dedupe_key in seen_node_ids:
                continue
            properties = scrub_object(node_dict.get("properties", {}))
            nodes_payload.append(
                {
                    "id": node_id,
                    "element_id": element_id,
                    "labels": list(node_dict.get("labels", [])),
                    "properties": properties,
                }
            )
            if dedupe_key is not None:
                seen_node_ids.add(dedupe_key)

        relationships_payload = []
        for rel_entry in relationship_entries:
            rel_dict = dict(rel_entry)
            rel_properties = scrub_object(rel_dict.get("properties", {}))
            relationships_payload.append(
                {
                    "type": rel_dict.get("type"),
                    "start": (str(rel_dict.get("start")) if rel_dict.get("start") is not None else None),
                    "end": (str(rel_dict.get("end")) if rel_dict.get("end") is not None else None),
                    "properties": rel_properties,
                }
            )

        if nodes_payload or relationships_payload:
            context[str(chunk_uid)] = {
                "nodes": nodes_payload,
                "relationships": relationships_payload,
            }

    return context


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
