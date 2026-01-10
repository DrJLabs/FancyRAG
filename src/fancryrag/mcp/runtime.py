"""Runtime helpers for the FastMCP hybrid server."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from fastmcp import FastMCP
from fastmcp.server.auth import AuthProvider
from fastmcp.server.auth.providers.google import GoogleProvider
from neo4j import GraphDatabase, RoutingControl
from neo4j import Driver
from neo4j.exceptions import Neo4jError
from neo4j.graph import Node
from neo4j_graphrag.retrievers import HybridCypherRetriever

from fancryrag.config import AppConfig
from fancryrag.embeddings import RetryingOpenAIEmbeddings


logger = logging.getLogger(__name__)


VECTOR_SCORE_QUERY = (
    "CALL db.index.vector.queryNodes($index_name, $limit, $query_vector) "
    "YIELD node, score RETURN elementId(node) AS element_id, score ORDER BY score DESC"
)

FULLTEXT_SCORE_QUERY = (
    "CALL db.index.fulltext.queryNodes($index_name, $query_text, {limit: $limit}) "
    "YIELD node, score RETURN elementId(node) AS element_id, score ORDER BY score DESC"
)

FETCH_NODE_QUERY = (
    "MATCH (n) WHERE elementId(n) = $element_id "
    "RETURN n AS node, labels(n) AS labels"
)


@dataclass(slots=True)
class ServerState:
    config: AppConfig
    driver: Driver
    retriever: HybridCypherRetriever


def _normalize_scores(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    items = {record["element_id"]: float(record["score"]) for record in records}
    if not items:
        return {}
    max_score = max(items.values())
    if max_score == 0:
        return {key: 0.0 for key in items}
    return {key: value / max_score for key, value in items.items()}


def _node_metadata(node: Node | None) -> Dict[str, Any]:
    if node is None:
        return {}

    if hasattr(node, "items"):
        metadata: Dict[str, Any] = dict(node.items())  # type: ignore[arg-type]
    else:  # pragma: no cover - fallback for unexpected node implementations
        metadata = {}
    metadata.pop("embedding", None)
    metadata["labels"] = list(node.labels)
    metadata["element_id"] = node.element_id
    return metadata


def create_state(config: AppConfig) -> ServerState:
    logger.info("server.initializing", extra={"neo4j_uri": config.neo4j.uri})

    driver = GraphDatabase.driver(
        config.neo4j.uri,
        auth=(config.neo4j.username, config.neo4j.password),
    )
    driver.verify_connectivity()

    embedder = RetryingOpenAIEmbeddings(
        model=config.embedding.model,
        base_url=config.embedding.base_url,
        api_key=config.embedding.api_key,
        timeout_seconds=config.embedding.timeout_seconds,
        max_retries=config.embedding.max_retries,
    )

    retriever = HybridCypherRetriever(
        driver=driver,
        vector_index_name=config.indexes.vector,
        fulltext_index_name=config.indexes.fulltext,
        retrieval_query=config.query.template,
        embedder=embedder,
        neo4j_database=config.neo4j.database,
    )

    logger.info(
        "server.initialized",
        extra={
            "vector_index": config.indexes.vector,
            "fulltext_index": config.indexes.fulltext,
        },
    )

    return ServerState(config=config, driver=driver, retriever=retriever)


def _vector_scores(
    state: ServerState, query_vector: list[float], top_k: int, ratio: int
) -> Dict[str, float]:
    if not query_vector:
        return {}

    try:
        records, _, _ = state.driver.execute_query(
            VECTOR_SCORE_QUERY,
            index_name=state.config.indexes.vector,
            limit=top_k * max(1, ratio),
            query_vector=query_vector,
            database_=state.config.neo4j.database,
            routing_=RoutingControl.READ,
        )
    except Neo4jError as error:
        logger.warning(
            "scores.vector.failed",
            extra={"error": type(error).__name__},
        )
        return {}

    plain = [
        {"element_id": record["element_id"], "score": record["score"]}
        for record in records
    ]
    return _normalize_scores(plain)


def _fulltext_scores(
    state: ServerState, query_text: str, top_k: int
) -> Dict[str, float]:
    if not query_text:
        return {}

    try:
        records, _, _ = state.driver.execute_query(
            FULLTEXT_SCORE_QUERY,
            index_name=state.config.indexes.fulltext,
            query_text=query_text,
            limit=top_k,
            database_=state.config.neo4j.database,
            routing_=RoutingControl.READ,
        )
    except Neo4jError as error:
        logger.warning(
            "scores.fulltext.failed",
            extra={"error": type(error).__name__},
        )
        return {}

    plain = [
        {"element_id": record["element_id"], "score": record["score"]}
        for record in records
    ]
    return _normalize_scores(plain)


def search_sync(
    state: ServerState, query: str, top_k: int, effective_ratio: int
) -> Dict[str, Any]:
    logger.info(
        "search.started",
        extra={
            "query": query,
            "top_k": top_k,
            "effective_ratio": effective_ratio,
        },
    )

    result = state.retriever.get_search_results(
        query_text=query,
        top_k=top_k,
        effective_search_ratio=effective_ratio,
    )

    query_vector = result.metadata.get("query_vector") if result.metadata else None
    vector_scores = _vector_scores(state, query_vector or [], top_k, effective_ratio)
    fulltext_scores = _fulltext_scores(state, query, top_k)

    items: List[Dict[str, Any]] = []
    for record in result.records:
        node: Node | None = record.get("node")
        metadata = _node_metadata(node)
        element_id = metadata.get("element_id")
        text_value = record.get("text")
        if not text_value and node is not None:
            text_value = node.get("text")
        semantic_nodes = record.get("semantic_nodes") or []
        semantic_relationships = record.get("semantic_relationships") or []

        items.append(
            {
                "text": text_value or "",
                "metadata": metadata,
                "score": float(record.get("score", 0.0)),
                "score_vector": float(vector_scores.get(element_id, 0.0))
                if element_id
                else 0.0,
                "score_fulltext": float(fulltext_scores.get(element_id, 0.0))
                if element_id
                else 0.0,
                "semantic_nodes": semantic_nodes,
                "semantic_relationships": semantic_relationships,
            }
        )

    logger.info(
        "search.completed",
        extra={
            "results": len(items),
        },
    )

    return {
        "query": query,
        "top_k": top_k,
        "effective_search_ratio": effective_ratio,
        "results": items,
    }


def fetch_sync(state: ServerState, element_id: str) -> Dict[str, Any]:
    logger.info("fetch.started", extra={"element_id": element_id})

    try:
        records, _, _ = state.driver.execute_query(
            FETCH_NODE_QUERY,
            element_id=element_id,
            database_=state.config.neo4j.database,
            routing_=RoutingControl.READ,
        )
    except Neo4jError as error:
        logger.error(
            "fetch.failed",
            extra={"element_id": element_id, "error": type(error).__name__},
        )
        return {
            "found": False,
            "element_id": element_id,
            "error": f"Database query failed: {type(error).__name__}",
        }

    if not records:
        logger.info("fetch.completed", extra={"element_id": element_id, "found": False})
        return {"found": False, "element_id": element_id}

    record = records[0]
    node: Node = record["node"]
    metadata = _node_metadata(node)
    metadata["labels"] = record.get("labels", metadata.get("labels", []))

    logger.info("fetch.completed", extra={"element_id": element_id, "found": True})
    return {
        "found": True,
        "element_id": element_id,
        "metadata": metadata,
        "text": metadata.get("text", ""),
    }


def build_server(
    state: ServerState, auth_provider: AuthProvider | None = None
) -> FastMCP:
    provider = None
    if state.config.server.auth_required:
        provider = auth_provider or GoogleProvider(
            client_id=state.config.oauth.client_id,
            client_secret=state.config.oauth.client_secret,
            base_url=state.config.server.base_url,
            required_scopes=state.config.oauth.required_scopes,
        )

    server = FastMCP(name="FancyRAG Hybrid MCP", auth=provider)

    @server.tool
    async def search(
        query: str, top_k: int = 5, effective_search_ratio: int = 1
    ) -> Dict[str, Any]:
        """Execute hybrid retrieval over Neo4j."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if effective_search_ratio <= 0:
            raise ValueError("effective_search_ratio must be greater than zero")
        return await asyncio.to_thread(
            search_sync, state, query, top_k, effective_search_ratio
        )

    @server.tool
    async def fetch(element_id: str) -> Dict[str, Any]:
        """Fetch a node by its Neo4j element id."""
        if not element_id:
            raise ValueError("element_id is required")
        return await asyncio.to_thread(fetch_sync, state, element_id)

    return server


__all__ = [
    "ServerState",
    "build_server",
    "create_state",
    "fetch_sync",
    "search_sync",
]
