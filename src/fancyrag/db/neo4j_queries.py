"""Shared Neo4j Cypher helpers used by FancyRAG pipeline and QA modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

try:  # pragma: no cover - exercised when neo4j optional dependency installed
    from neo4j.exceptions import Neo4jError
except ImportError:  # pragma: no cover - fallback for environments without neo4j
    class Neo4jError(Exception):
        """Fallback Neo4j error used when the driver is unavailable."""


from _compat.structlog import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ChunkMetadataLike(Protocol):
    """Protocol describing the metadata captured for each ingested chunk."""

    uid: str
    sequence: int
    index: int
    checksum: str
    relative_path: str
    git_commit: str | None


@runtime_checkable
class QaChunkRecordLike(Protocol):
    """Protocol describing the minimal QA chunk record used during rollback."""

    uid: str


@runtime_checkable
class QaSourceRecordLike(Protocol):
    """Protocol describing the QA source payload required for rollback queries."""

    chunks: Sequence[QaChunkRecordLike]
    relative_path: str
    ingest_run_key: str | None


ValueResult = Sequence[Mapping[str, Any]] | Mapping[str, Any] | Any


def reset_database(driver: Any, *, database: str | None) -> None:
    """Remove previously ingested nodes to guarantee a clean ingest for the run."""

    driver.execute_query("MATCH (n) DETACH DELETE n", database_=database)


def ensure_document_relationships(
    driver: Any,
    *,
    database: str | None,
    source_path: Path,
    relative_path: str,
    git_commit: str | None,
    document_checksum: str,
    chunks_metadata: Sequence[ChunkMetadataLike],
) -> None:
    """
    Ensure a Document node exists for the given source file and attach chunk provenance.
    """

    chunk_payload = [
        {
            "uid": meta.uid,
            "sequence": meta.sequence,
            "index": meta.index,
            "relative_path": meta.relative_path,
            "git_commit": meta.git_commit,
            "checksum": meta.checksum,
        }
        for meta in chunks_metadata
    ]

    driver.execute_query(
        """
        // Create or reuse the Document node representing this source file
        MERGE (doc:Document {source_path: $source_path})
          ON CREATE SET doc.name = $document_name,
                        doc.title = $document_name
        // Refresh document-level provenance on every ingestion
        SET doc.relative_path = $relative_path,
            doc.git_commit = $git_commit,
            doc.checksum = $document_checksum
        WITH doc
        // Process each chunk emitted by the current pipeline execution
        UNWIND $chunk_payload AS meta
        // Locate the unique chunk that matches the current payload entry using the uid assigned post-pipeline
        MATCH (chunk:Chunk {uid: meta.uid})
        WITH doc, chunk, meta
        // Update per-chunk provenance while preserving existing identifiers when re-ingesting
        SET chunk.source_path = $source_path,
            chunk.relative_path = meta.relative_path,
            chunk.git_commit = meta.git_commit,
            chunk.checksum = meta.checksum,
            chunk.chunk_id = coalesce(chunk.chunk_id, meta.sequence),
            chunk.index = coalesce(chunk.index, meta.index)
        // Ensure the Document ↔ Chunk relationship exists for this payload entry
        MERGE (doc)-[:HAS_CHUNK]->(chunk)
        """,
        {
            "source_path": str(source_path),
            "document_name": source_path.name,
            "relative_path": relative_path,
            "git_commit": git_commit,
            "document_checksum": document_checksum,
            "chunk_payload": chunk_payload,
        },
        database_=database,
    )


def rollback_ingest(
    driver: Any,
    *,
    database: str | None,
    sources: Sequence[QaSourceRecordLike],
) -> None:
    """Delete graph elements produced during the provided ingestion sources."""

    run_keys = {
        record.ingest_run_key for record in sources if record.ingest_run_key
    }
    if run_keys:
        driver.execute_query(
            """
            UNWIND $run_keys AS run_key
            MATCH ()-[rel]-()
            WHERE rel.ingest_run_key = run_key
            DELETE rel
            """,
            {"run_keys": list(run_keys)},
            database_=database,
        )
        driver.execute_query(
            """
            UNWIND $run_keys AS run_key
            MATCH (node)
            WHERE node.ingest_run_key = run_key
              AND NOT node:Document
              AND NOT node:Chunk
            DETACH DELETE node
            """,
            {"run_keys": list(run_keys)},
            database_=database,
        )

    chunk_uids = [chunk.uid for record in sources for chunk in record.chunks]
    if chunk_uids:
        driver.execute_query(
            """
            UNWIND $uids AS uid
            MATCH (c:Chunk {uid: uid})
            DETACH DELETE c
            """,
            {"uids": chunk_uids},
            database_=database,
        )

    relative_paths = {record.relative_path for record in sources}
    if relative_paths:
        driver.execute_query(
            """
            UNWIND $paths AS path
            MATCH (doc:Document {relative_path: path})
            WHERE NOT (doc)-[:HAS_CHUNK]->(:Chunk)
            DETACH DELETE doc
            """,
            {"paths": list(relative_paths)},
            database_=database,
        )


def collect_counts(driver: Any, *, database: str | None) -> Mapping[str, int]:
    """Return counts of Documents, Chunks, and HAS_CHUNK relationships from Neo4j."""

    queries = {
        "documents": "MATCH (:Document) RETURN count(*) AS value",
        "chunks": "MATCH (:Chunk) RETURN count(*) AS value",
        "relationships": "MATCH (:Document)-[:HAS_CHUNK]->(:Chunk) RETURN count(*) AS value",
    }
    counts: dict[str, int] = {}
    for key, query in queries.items():
        counts[key] = 0
        try:
            counts[key] = _execute_value_query(driver, query, {}, database)
        except Neo4jError:
            logger.warning("neo4j.count_failed", query=key)
    return counts


def count_missing_embeddings(
    driver: Any, *, database: str | None, chunk_uids: Sequence[str]
) -> int:
    """Return the number of chunks missing embedding vectors."""

    if not chunk_uids:
        return 0
    return _execute_value_query(
        driver,
        """
        UNWIND $uids AS uid
        MATCH (c:Chunk {uid: uid})
        WHERE c.embedding IS NULL OR size(c.embedding) = 0
        RETURN count(*) AS value
        """,
        {"uids": list(chunk_uids)},
        database,
    )


def count_orphan_chunks(
    driver: Any, *, database: str | None, chunk_uids: Sequence[str]
) -> int:
    """Return the number of ingested chunks that lack HAS_CHUNK relationships."""

    if not chunk_uids:
        return 0
    return _execute_value_query(
        driver,
        """
        UNWIND $uids AS uid
        MATCH (c:Chunk {uid: uid})
        WHERE NOT ( (:Document)-[:HAS_CHUNK]->(c) )
        RETURN count(*) AS value
        """,
        {"uids": list(chunk_uids)},
        database,
    )


def count_checksum_mismatches(
    driver: Any,
    *,
    database: str | None,
    chunk_rows: Sequence[Mapping[str, str]],
) -> int:
    """Return the number of chunk checksum mismatches against provided metadata."""

    if not chunk_rows:
        return 0
    return _execute_value_query(
        driver,
        """
        UNWIND $rows AS row
        MATCH (c:Chunk {uid: row.uid})
        WHERE coalesce(c.checksum, "") <> row.checksum
        RETURN count(*) AS value
        """,
        {"rows": list(chunk_rows)},
        database,
    )


def collect_semantic_counts(
    driver: Any,
    *,
    database: str | None,
    source_tag: str,
) -> Mapping[str, int]:
    """Return counts of semantic enrichment entities associated with ``source_tag``."""

    if not source_tag:
        return {"nodes_in_db": 0, "relationships_in_db": 0, "orphan_entities": 0}

    nodes = _execute_value_query(
        driver,
        "MATCH (n) WHERE n.semantic_source = $source RETURN count(*) AS value",
        {"source": source_tag},
        database,
    )
    relationships = _execute_value_query(
        driver,
        "MATCH ()-[r]->() WHERE r.semantic_source = $source RETURN count(*) AS value",
        {"source": source_tag},
        database,
    )
    orphans = _execute_value_query(
        driver,
        "MATCH (n) WHERE n.semantic_source = $source AND NOT (n)--() RETURN count(*) AS value",
        {"source": source_tag},
        database,
    )
    return {
        "nodes_in_db": nodes,
        "relationships_in_db": relationships,
        "orphan_entities": orphans,
    }


def _execute_value_query(
    driver: Any,
    query: str,
    parameters: Mapping[str, Any],
    database: str | None,
) -> int:
    result = driver.execute_query(
        query,
        parameters=dict(parameters),
        database_=database,
    )
    return _extract_value(result)


def _extract_value(result: ValueResult) -> int:
    records = _normalise_records(result)
    if not records:
        return 0
    record = records[0]
    if isinstance(record, Mapping):
        value = record.get("value")
    elif hasattr(record, "value"):
        value = getattr(record, "value")
    else:
        try:
            value = record[0]  # type: ignore[index]
        except (IndexError, TypeError):  # pragma: no cover - defensive guard
            value = None
    return int(value or 0)


def _normalise_records(result: ValueResult) -> Sequence[Mapping[str, Any]]:
    if isinstance(result, tuple):
        result = result[0]
    if hasattr(result, "records"):
        records = result.records  # type: ignore[attr-defined]
        if callable(records):
            records = records()
        if isinstance(records, Sequence) and not isinstance(
            records, (str, bytes, bytearray)
        ):
            return records
        return list(records) if records else []
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        return result
    if isinstance(result, Mapping):
        return [result]
    return []


__all__ = [
    "Neo4jError",
    "ChunkMetadataLike",
    "QaChunkRecordLike",
    "QaSourceRecordLike",
    "collect_counts",
    "collect_semantic_counts",
    "count_checksum_mismatches",
    "count_missing_embeddings",
    "count_orphan_chunks",
    "ensure_document_relationships",
    "reset_database",
    "rollback_ingest",
]
