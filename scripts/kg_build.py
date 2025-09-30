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

from neo4j import AsyncGraphDatabase
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
from pydantic import BaseModel, ConfigDict, Field

logger = get_logger(__name__)

DEFAULT_SOURCE = Path("docs/samples/pilot.txt")
DEFAULT_LOG_PATH = Path("artifacts/local_stack/kg_build.json")
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "config" / "kg_schema.json"


def _load_default_schema(path: Path = DEFAULT_SCHEMA_PATH) -> GraphSchema:
    """Load and validate the default GraphSchema definition from disk."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - installation defect
        raise RuntimeError(f"default schema file not found: {path}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - installation defect
        raise RuntimeError(f"invalid schema JSON: {path}") from exc
    # GraphSchema defaults rely on pydantic default factories that expect a validated
    # payload. Pydantic 2.9+ stopped forwarding that context, which makes the
    # upstream default factories incompatible when only labels are supplied.  Eagerly
    # validating here ensures the pipeline receives a ready GraphSchema instance and
    # sidesteps the incompatibility without relaxing validation downstream.
    return GraphSchema.model_validate(raw)


DEFAULT_SCHEMA = _load_default_schema()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the KG build script.
    
    Parameters:
        argv (Sequence[str] | None): Optional list of argument strings to parse (defaults to process argv when None).
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes `source`, `chunk_size`, `chunk_overlap`, `database`, and `log_path`.
    """
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
    """
    Read UTF-8 text content from a file path and ensure it is not empty.
    
    Parameters:
        path (Path): Filesystem path to the source file to read.
    
    Returns:
        str: The file contents decoded as UTF-8.
    
    Raises:
        FileNotFoundError: If the given path does not exist.
        ValueError: If the file contains only whitespace or is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"source file not found: {path}")
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"source file is empty: {path}")
    return content


def _ensure_positive(value: int, *, name: str) -> int:
    """
    Ensure the provided integer is greater than zero.
    
    Parameters:
        value (int): The integer to validate; must be greater than zero.
        name (str): Parameter name used in the error message if validation fails.
    
    Returns:
        int: The validated value.
    
    Raises:
        ValueError: If `value` is less than or equal to zero.
    """
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _ensure_non_negative(value: int, *, name: str) -> int:
    """
    Validate that `value` is zero or positive.
    
    Parameters:
    	value (int): Integer to validate.
    	name (str): Identifier used in the ValueError message when validation fails.
    
    Returns:
    	int: The same `value` when it is zero or greater.
    
    Raises:
    	ValueError: If `value` is less than zero; message will include `name`.
    """
    if value < 0:
        raise ValueError(f"{name} must be zero or positive")
    return value


def _ensure_directory(path: Path) -> None:
    """
    Ensure the parent directory of the given path exists by creating it if necessary.
    
    Parameters:
    	path (Path): The filesystem path whose parent directory will be created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _coerce_text(value: Any) -> str | None:
    """Best-effort conversion of heterogeneous content payloads to text."""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        inner = value.get("value")
        return str(inner) if inner else None
    if hasattr(value, "value"):
        inner = getattr(value, "value")
        return str(inner) if inner else None
    return None


class ChatContentPart(BaseModel):
    """Pydantic model for tool and text content entries."""

    model_config = ConfigDict(extra="allow")

    text: Any = None
    content: Any = None

    def as_text(self) -> str | None:
        text = _coerce_text(self.text)
        if text:
            return text
        return _coerce_text(self.content)


class ChatMessage(BaseModel):
    """Pydantic model for OpenAI chat messages supporting rich content."""

    model_config = ConfigDict(extra="allow")

    content: Any = None

    def parts(self) -> list[str]:
        content = self.content
        if isinstance(content, str):
            return [content]
        parts: list[str] = []
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            for item in content:
                if isinstance(item, str):
                    if item:
                        parts.append(item)
                    continue
                if isinstance(item, Mapping):
                    text = ChatContentPart.model_validate(item).as_text()
                else:
                    text = _coerce_text(item)
                if text:
                    parts.append(text)
        elif isinstance(content, Mapping):
            text = ChatContentPart.model_validate(content).as_text()
            if text:
                parts.append(text)
        else:
            text = _coerce_text(content)
            if text:
                parts.append(text)
        return parts


class ChatChoice(BaseModel):
    """Pydantic model for chat completion choices."""

    model_config = ConfigDict(extra="allow")

    message: ChatMessage | None = Field(default=None)


class ChatCompletion(BaseModel):
    """Pydantic model covering the subset of fields used by kg_build."""

    model_config = ConfigDict(extra="allow")

    choices: list[ChatChoice] = Field(default_factory=list)

    def first_text(self) -> str:
        for choice in self.choices:
            if not choice.message:
                continue
            parts = choice.message.parts()
            if parts:
                return "".join(parts)
        return ""


def _extract_content(raw_response: Any) -> str:
    """Extract textual content from a chat-completion style response."""

    completion = ChatCompletion.model_validate(raw_response)
    return completion.first_text()


def _strip_code_fence(text: str) -> str:
    """
    Remove surrounding Markdown code fences (``` blocks) from the given text.
    
    Returns:
        The text with a leading and trailing triple-backtick fence removed if present, then trimmed of leading and trailing whitespace.
    """
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
        """
        Create a SharedOpenAIEmbedder that will use the provided shared OpenAI client and settings for embedding requests.
        
        Parameters:
            client: SharedOpenAIClient used to perform embedding API calls.
            settings: OpenAISettings that configure model selection and embedding dimensionality.
        """
        self._client = client
        self._settings = settings

    def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding vector for the given text using the shared OpenAI client.
        
        Parameters:
            text (str): Text to encode into an embedding.
        
        Returns:
            list[float]: Embedding vector for the input text; dimensions are validated or adjusted per settings.
        
        Raises:
            EmbeddingsGenerationError: If the embedding request to the OpenAI client fails.
        """
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
        """
        Create a SharedOpenAILLM that routes LLM requests through a shared OpenAI client using the provided settings.
        
        Parameters:
            client (SharedOpenAIClient): Shared client used to perform chat completions.
            settings (OpenAISettings): Configuration that supplies the chat model name and any model-specific defaults; the constructor uses it to set the model name and default model parameters.
        """
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
        """
        Builds a list of chat-style message dictionaries suitable for the chat API.
        
        Parameters:
            input_text (str): The user's message content to append as the final message.
            message_history (Sequence[Mapping[str, str]] | None): Optional prior messages (each with "role" and "content") to include in order.
            system_instruction (str | None): Optional system-level instruction to prepend as the first message.
        
        Returns:
            list[Mapping[str, str]]: Ordered list of messages where each message is a mapping with keys "role" and "content".
        """
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
        """
        Generate a chat completion for the given input using the configured OpenAI client and return the extracted text.
        
        Parameters:
            input (str): The user prompt or input text to send to the LLM.
            message_history (Sequence[Mapping[str, str]] | None): Optional prior messages to include in the conversation (each message a mapping with typical keys like "role" and "content").
            system_instruction (str | None): Optional system-level instruction to prepend to the message sequence.
        
        Returns:
            LLMResponse: An LLMResponse containing the extracted and code-fence-stripped text from the model's reply.
        
        Raises:
            LLMGenerationError: If the OpenAI client fails or the model returns an empty response.
        """
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
        """
        Asynchronously invoke the LLM with the given input, optional message history, and optional system instruction.
        
        Parameters:
            input (str): User prompt or input text for the LLM.
            message_history (Sequence[Mapping[str, str]] | None): Optional sequence of prior messages where each message is a mapping with keys like `"role"` and `"content"`.
            system_instruction (str | None): Optional system-level instruction to prepend to the message stream.
        
        Returns:
            LLMResponse: The model's response containing the generated content.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.invoke,
            input,
            message_history,
            system_instruction,
        )


async def _collect_counts(driver, *, database: str | None) -> Mapping[str, int]:
    """
    Collects counts of documents, chunks, and document→chunk relationships from the specified Neo4j database.
    
    Queries the database for:
    - number of Document nodes,
    - number of Chunk nodes,
    - number of HAS_CHUNK relationships from Document to Chunk.
    
    Parameters:
        database (str | None): Name of the Neo4j database to query; pass `None` to use the driver's default.
    
    Returns:
        Mapping[str, int]: A mapping with keys "documents", "chunks", and "relationships" containing integer counts.
        Keys for which the query failed are omitted from the mapping.
    """
    queries = {
        "documents": "MATCH (:Document) RETURN count(*) AS value",
        "chunks": "MATCH (:Chunk) RETURN count(*) AS value",
        "relationships": "MATCH (:Document)-[:HAS_CHUNK]->(:Chunk) RETURN count(*) AS value",
    }
    counts: dict[str, int] = {}
    for key, query in queries.items():
        try:
            result = await driver.execute_query(query, database_=database)
            records = result
            if isinstance(result, tuple):
                records = result[0]
            else:
                records = getattr(result, "records", result)
            if records:
                record = records[0]
                value = record.get("value") if isinstance(record, Mapping) else record[0]
                counts[key] = int(value or 0)
        except Neo4jError:
            logger.warning("kg_build.count_failed", query=key)
    return counts


async def _execute_pipeline(
    *,
    uri: str,
    auth: tuple[str, str],
    source_text: str,
    database: str | None,
    embedder: Embedder,
    llm: SharedOpenAILLM,
    splitter: FixedSizeSplitter,
) -> tuple[str | None, Mapping[str, int]]:
    """Run the SimpleKGPipeline using the async Neo4j driver."""

    async with AsyncGraphDatabase.driver(uri, auth=auth) as driver:
        pipeline = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            embedder=embedder,
            schema=DEFAULT_SCHEMA,
            from_pdf=False,
            text_splitter=splitter,
            neo4j_database=database,
        )
        result = await pipeline.run_async(text=source_text)
        counts = await _collect_counts(driver, database=database)
        return result.run_id, counts


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Builds a knowledge graph from a source file and writes a structured JSON run log.
    
    Parses CLI arguments (or uses provided argv), validates environment and chunking parameters, runs the SimpleKGPipeline to ingest the source text into Neo4j using OpenAI-based embedder and LLM adapters, collects database counts, writes a sanitized JSON log to disk, and returns the log dictionary.
    
    Parameters:
        argv (Sequence[str] | None): Optional list of CLI arguments to override sys.argv; when None, defaults from the environment/argument parser are used.
    
    Returns:
        dict[str, Any]: A structured log dictionary containing keys such as:
            - timestamp: ISO 8601 UTC timestamp of completion
            - operation: the operation name ("kg_build")
            - status: operation status ("success" on normal completion)
            - duration_ms: elapsed time in milliseconds
            - source: path to the input source file
            - input_bytes: size of the input in bytes
            - chunking: mapping with "size" and "overlap" values used for splitting
            - database: Neo4j database name (or None)
            - openai: OpenAI settings used (chat_model, embedding_model, embedding_dimensions, max_attempts)
            - counts: mapping with counts of ingested entities (documents, chunks, relationships)
            - run_id: pipeline run identifier (if available)
    
    Raises:
        RuntimeError: if OpenAI requests or Neo4j operations fail.
    """
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
        run_id, counts = asyncio.run(
            _execute_pipeline(
                uri=uri,
                auth=auth,
                source_text=source_text,
                database=args.database,
                embedder=embedder,
                llm=llm,
                splitter=splitter,
            )
        )
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
    """
    Run the CLI workflow for building the knowledge graph and yield a process-style exit code.
    
    Parameters:
        argv (Sequence[str] | None): Command-line arguments to pass to the run() function; if None, the program default is used.
    
    Returns:
        exit_code (int): 0 on successful completion, 1 if an error occurred.
    """
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
