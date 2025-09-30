#!/usr/bin/env python
"""Run the SimpleKGPipeline against sample content for the minimal path workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, Neo4jError

from _compat.structlog import get_logger
from cli.openai_client import OpenAIClientError, SharedOpenAIClient
from cli.sanitizer import scrub_object
from cli.utils import ensure_embedding_dimensions
from config.settings import DEFAULT_EMBEDDING_DIMENSIONS, OpenAISettings
from fancyrag.utils import ensure_env
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import EmbeddingsGenerationError, LLMGenerationError
from neo4j_graphrag.experimental.components.schema import GraphSchema
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse

logger = get_logger(__name__)

DEFAULT_SOURCE = Path("docs/samples/pilot.txt")
DEFAULT_LOG_PATH = Path("artifacts/local_stack/kg_build.json")
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
_RAW_DEFAULT_SCHEMA = {
    "node_types": [
        {"label": "Document", "additional_properties": True},
        {"label": "Chunk", "additional_properties": True},
        {"label": "Company", "additional_properties": True},
        {"label": "Product", "additional_properties": True},
        {"label": "Operator", "additional_properties": True},
    ],
    "relationship_types": [
        {"label": "HAS_CHUNK", "additional_properties": True},
        {"label": "LAUNCHED", "additional_properties": True},
        {"label": "INGESTED_BY", "additional_properties": True},
    ],
    "patterns": [
        ["Document", "HAS_CHUNK", "Chunk"],
        ["Company", "LAUNCHED", "Product"],
        ["Chunk", "INGESTED_BY", "Operator"],
    ],
    "additional_node_types": False,
    "additional_relationship_types": False,
    "additional_patterns": False,
}

# GraphSchema defaults rely on pydantic default factories that expect a validated
# payload. Pydantic 2.9+ stopped forwarding that context, which makes the
# upstream default factories incompatible when only labels are supplied.  Eagerly
# validating here ensures the pipeline receives a ready GraphSchema instance and
# sidesteps the incompatibility without relaxing validation downstream.
DEFAULT_SCHEMA = GraphSchema.model_validate(_RAW_DEFAULT_SCHEMA)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SimpleKGPipeline against local sample content, persisting results to Neo4j with structured logging and retries.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Path to the sample content to ingest (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Character chunk size for the text splitter (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Character overlap between chunks (default: %(default)s)",
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


def _read_source(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"source file not found: {path}")
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"source file is empty: {path}")
    return content


def _ensure_positive(value: int, *, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _ensure_non_negative(value: int, *, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be zero or positive")
    return value


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_content(raw_response: Any) -> str:
    choices = getattr(raw_response, "choices", None)
    if choices is None and isinstance(raw_response, Mapping):
        choices = raw_response.get("choices")
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, Mapping):
        message = first.get("message")
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, Mapping):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for item in content:
            text_value = getattr(item, "text", None)
            if isinstance(text_value, Mapping):
                part = text_value.get("value")
            elif hasattr(text_value, "value"):
                part = getattr(text_value, "value")
            else:
                part = None
            if part:
                parts.append(str(part))
            else:
                maybe_str = getattr(item, "content", None)
                if maybe_str:
                    parts.append(str(maybe_str))
        return "".join(parts)
    return ""


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


class SharedOpenAIEmbedder(Embedder):
    """Embedder adapter that reuses the SharedOpenAIClient."""

    def __init__(self, client: SharedOpenAIClient, settings: OpenAISettings) -> None:
        self._client = client
        self._settings = settings

    def embed_query(self, text: str) -> list[float]:
        try:
            result = self._client.embedding(input_text=text)
        except OpenAIClientError as exc:
            raise EmbeddingsGenerationError(str(exc)) from exc
        vector = list(result.vector)
        ensure_embedding_dimensions(vector, settings=self._settings)
        return vector


class SharedOpenAILLM(LLMInterface):
    """LLM adapter that routes generation through SharedOpenAIClient."""

    def __init__(self, client: SharedOpenAIClient, settings: OpenAISettings) -> None:
        super().__init__(
            model_name=settings.chat_model,
            model_params={"temperature": 0.0, "max_tokens": 512},
        )
        self._client = client

    def _build_messages(
        self,
        input_text: str,
        message_history: Sequence[Mapping[str, str]] | None,
        system_instruction: str | None,
    ) -> list[Mapping[str, str]]:
        messages: list[Mapping[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        if message_history:
            messages.extend(message_history)
        messages.append({"role": "user", "content": input_text})
        return messages

    def invoke(
        self,
        input: str,
        message_history: Sequence[Mapping[str, str]] | None = None,
        system_instruction: str | None = None,
    ) -> LLMResponse:
        messages = self._build_messages(input, message_history, system_instruction)
        try:
            result = self._client.chat_completion(
                messages=messages,
                temperature=self.model_params.get("temperature", 0.0),
            )
        except OpenAIClientError as exc:
            raise LLMGenerationError(str(exc)) from exc

        content = _strip_code_fence(_extract_content(result.raw_response))
        logger.info("kg_build.llm_response", content=content)
        if not content:
            raise LLMGenerationError("OpenAI returned an empty response")
        return LLMResponse(content=content)

    async def ainvoke(
        self,
        input: str,
        message_history: Sequence[Mapping[str, str]] | None = None,
        system_instruction: str | None = None,
    ) -> LLMResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.invoke,
            input,
            message_history,
            system_instruction,
        )


def _collect_counts(driver, *, database: str | None) -> Mapping[str, int]:
    queries = {
        "documents": "MATCH (:Document) RETURN count(*) AS value",
        "chunks": "MATCH (:Chunk) RETURN count(*) AS value",
        "relationships": "MATCH (:Document)-[:HAS_CHUNK]->(:Chunk) RETURN count(*) AS value",
    }
    counts: dict[str, int] = {}
    for key, query in queries.items():
        try:
            result = driver.execute_query(query, database_=database)
            records = getattr(result, "records", result)
            if records:
                record = records[0]
                value = record.get("value") if isinstance(record, Mapping) else record[0]
                counts[key] = int(value or 0)
        except Neo4jError:
            logger.warning("kg_build.count_failed", query=key)
    return counts


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    args.chunk_size = _ensure_positive(args.chunk_size, name="chunk_size")
    args.chunk_overlap = _ensure_non_negative(args.chunk_overlap, name="chunk_overlap")

    ensure_env("OPENAI_API_KEY")
    ensure_env("NEO4J_URI")
    ensure_env("NEO4J_USERNAME")
    ensure_env("NEO4J_PASSWORD")

    source_path = Path(args.source).expanduser()
    source_text = _read_source(source_path)

    settings = OpenAISettings.load(actor="kg_build")
    shared_client = SharedOpenAIClient(settings)
    embedder = SharedOpenAIEmbedder(shared_client, settings)
    llm = SharedOpenAILLM(shared_client, settings)
    splitter = FixedSizeSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    uri = os.environ["NEO4J_URI"]
    auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])

    start = time.perf_counter()
    run_id: str | None = None
    counts: Mapping[str, int] = {}

    try:
        with GraphDatabase.driver(uri, auth=auth) as driver:
            pipeline = SimpleKGPipeline(
                llm=llm,
                driver=driver,
                embedder=embedder,
                schema=DEFAULT_SCHEMA,
                from_pdf=False,
                text_splitter=splitter,
                neo4j_database=args.database,
            )
            result = asyncio.run(pipeline.run_async(text=source_text))
            run_id = result.run_id
            counts = _collect_counts(driver, database=args.database)
    except (OpenAIClientError, LLMGenerationError, EmbeddingsGenerationError) as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc
    except (Neo4jError, ClientError) as exc:
        raise RuntimeError(f"Neo4j error: {exc}") from exc

    duration_ms = int((time.perf_counter() - start) * 1000)

    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "kg_build",
        "status": "success",
        "duration_ms": duration_ms,
        "source": str(source_path),
        "input_bytes": len(source_text.encode("utf-8")),
        "chunking": {
            "size": args.chunk_size,
            "overlap": args.chunk_overlap,
        },
        "database": args.database,
        "openai": {
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "max_attempts": settings.max_attempts,
        },
        "counts": counts,
        "run_id": run_id,
    }

    log_path = Path(args.log_path)
    _ensure_directory(log_path)
    sanitized = scrub_object(log)
    log_path.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    print(json.dumps(sanitized))
    logger.info("kg_build.completed", **sanitized)
    return log


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(argv)
        return 0
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        logger.error("kg_build.error", error=str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - final guard
        print(f"error: {exc}", file=sys.stderr)
        logger.exception("kg_build.failed", error=str(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
