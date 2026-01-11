"""Smoke test that runs inside the Compose network to validate the MCP stack."""

from __future__ import annotations

import asyncio
import os
import time
import urllib.error
import urllib.request
from typing import Any

import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from neo4j import GraphDatabase


_SEARCH_PAYLOAD = {"query": "container", "top_k": 5}
_LATENCY_BUDGET_SECONDS = float(os.environ.get("SMOKE_LATENCY_BUDGET_SECONDS", "1.5"))


def _wait_for_health(url: str, token: str, timeout_seconds: float = 180.0) -> None:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
    }
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=10):
                return
        except urllib.error.HTTPError as error:
            if error.code in {401, 406}:
                return
            time.sleep(2)
        except (urllib.error.URLError, ConnectionResetError):
            time.sleep(2)
    raise AssertionError(f"Timed out waiting for {url}")


def _seed_neo4j(uri: str, username: str, password: str, database: str) -> None:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            statements = [
                "DROP INDEX text_embeddings IF EXISTS",
                "DROP INDEX chunk_text_fulltext IF EXISTS",
                """
                CREATE VECTOR INDEX text_embeddings IF NOT EXISTS
                  FOR (n:Chunk) ON (n.embedding)
                  OPTIONS {indexConfig: {`vector.dimensions`: 3, `vector.similarity_function`: 'cosine'}}
                """.strip(),
                "CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]",
                "CALL db.awaitIndexes(120)",
                "MATCH (n:Chunk) DETACH DELETE n",
                "CREATE (:Chunk {text: 'Container smoke chunk', embedding: [0.42, 0.13, 0.88]})",
            ]
            for statement in statements:
                session.run(statement)
    finally:
        driver.close()


async def _invoke_search_async(base_url: str, token: str) -> tuple[dict[str, Any], float]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
    }
    async with streamablehttp_client(base_url, headers=headers) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            await session.list_tools()
            start_time = time.perf_counter()
            result = await session.call_tool("search", _SEARCH_PAYLOAD)
            latency = time.perf_counter() - start_time
            if result.isError:
                raise AssertionError(f"search tool returned error: {result}")

            if result.structuredContent is not None:
                return result.structuredContent, latency

            items = [item.text for item in result.content if hasattr(item, "text")]
            return {"results": items}, latency


def _invoke_search(base_url: str, token: str) -> tuple[dict[str, Any], float]:
    return asyncio.run(_invoke_search_async(base_url, token))


@pytest.mark.integration
def test_compose_smoke_stack() -> None:
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ["NEO4J_DATABASE"]
    base_url = os.environ["MCP_URL"].rstrip("/")
    token = os.environ["MCP_TOKEN"]

    _wait_for_health(f"{base_url}/health", token)
    _seed_neo4j(uri, username, password, database)

    response, latency = _invoke_search(base_url, token)
    results = response.get("results")
    assert isinstance(results, list), "search response must include results list"
    assert results, "expected at least one search result"
    assert (
        latency <= _LATENCY_BUDGET_SECONDS
    ), f"expected search latency ≤ {_LATENCY_BUDGET_SECONDS}s, got {latency:.3f}s"
