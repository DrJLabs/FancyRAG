#!/usr/bin/env python
"""Run the SimpleKGPipeline against sample content for the minimal path workflow."""

from __future__ import annotations

import argparse
import asyncio
import copy
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from neo4j_graphrag.experimental.components.kg_writer import KGWriterModel, Neo4jWriter
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphConfig
from neo4j_graphrag.experimental.components.schema import GraphSchema
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import TextChunks
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from pydantic import ValidationError, validate_call

# Maintain compatibility with earlier async driver import paths used in tests.
AsyncGraphDatabase = GraphDatabase

try:  # pragma: no cover - optional dependency in some environments
    from openai.types.chat import ChatCompletion as OpenAIChatCompletion
except Exception:  # pragma: no cover - fall back when OpenAI SDK is absent
    OpenAIChatCompletion = None  # type: ignore[assignment]

logger = get_logger(__name__)

DEFAULT_SOURCE = Path("docs/samples/pilot.txt")
DEFAULT_LOG_PATH = Path("artifacts/local_stack/kg_build.json")
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "config" / "kg_schema.json"

DEFAULT_PROFILE = "text"
PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "text": {
        "chunk_size": 600,
        "chunk_overlap": 100,
        "include": ("**/*.txt", "**/*.md", "**/*.rst"),
    },
    "markdown": {
        "chunk_size": 800,
        "chunk_overlap": 120,
        "include": ("**/*.md", "**/*.markdown", "**/*.mdx", "**/*.txt", "**/*.rst"),
    },
    "code": {
        "chunk_size": 400,
        "chunk_overlap": 40,
        "include": (
            "**/*.py",
            "**/*.ts",
            "**/*.tsx",
            "**/*.js",
            "**/*.java",
            "**/*.go",
            "**/*.rs",
            "**/*.rb",
            "**/*.php",
            "**/*.cs",
            "**/*.c",
            "**/*.cpp",
            "**/*.hpp",
            "**/*.proto",
        ),
    },
}


@dataclass
class SourceSpec:
    """Represents a resolved ingestion source."""

    path: Path
    relative_path: str
    text: str
    checksum: str


class CachingFixedSizeSplitter(FixedSizeSplitter):
    """Fixed-size splitter that caches results to avoid duplicate work."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Store cached results keyed by the raw text (or tuple of segments).  Each
        # lookup returns a deep copy so per-file provenance stays isolated even
        # when different inputs share identical content.
        self._cache: dict[str | tuple[str, ...], TextChunks] = {}

    @staticmethod
    def _cache_key(text: str | Sequence[str]) -> str | tuple[str, ...]:
        if isinstance(text, str):
            return text
        return tuple(text)

    async def run(
        self, text: str | Sequence[str], config: Any | None = None
    ) -> TextChunks:  # type: ignore[override]
        if config is not None:
            # Defer to the base implementation when custom configuration is
            # supplied; caching only targets the default execution path.  The
            # upstream splitter signature differs slightly across releases, so
            # fall back to the simplest invocation if a positional/keyword
            # mismatch occurs.
            try:
                return await super().run(text, config)
            except (TypeError, ValidationError):  # pragma: no cover - safety net
                return await super().run(text)

        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            return copy.deepcopy(cached)

        result = await super().run(text)
        self._cache[key] = copy.deepcopy(result)
        return result

    def get_cached(self, text: str | Sequence[str]) -> TextChunks | None:
        """Return the cached chunk result for ``text`` if available."""

        cached = self._cache.get(self._cache_key(text))
        if cached is not None:
            return copy.deepcopy(cached)
        return None


@dataclass
class ChunkMetadata:
    """Metadata captured for each chunk written to Neo4j/Qdrant."""

    uid: str
    sequence: int
    index: int
    checksum: str
    relative_path: str
    git_commit: str | None


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


PRIMITIVE_TYPES = (str, int, float, bool)


def _resolve_git_commit() -> str | None:
    """Resolve the current git commit SHA if available."""

    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        result = subprocess.run(
            [git_executable, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None
    commit = result.stdout.strip()
    return commit or None


@functools.lru_cache(maxsize=1)
def _resolve_repo_root() -> Path | None:
    """Return the repository root directory if git metadata is available."""

    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        result = subprocess.run(
            [git_executable, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None
    root = result.stdout.strip()
    return Path(root) if root else None


def _relative_to_repo(path: Path, *, base: Path | None = None) -> str:
    """Return a stable relative path for the provided file."""

    resolved = path.resolve()
    for candidate in (base.resolve() if base else None, _resolve_repo_root(), Path.cwd()):
        if candidate is None:
            continue
        try:
            return str(resolved.relative_to(candidate))
        except ValueError:
            continue
    try:
        return str(resolved.relative_to(resolved.anchor))
    except ValueError:  # pragma: no cover - defensive fallback
        return resolved.as_posix()


def _discover_source_files(directory: Path, patterns: Iterable[str]) -> list[Path]:
    """Return deterministically ordered files matching the given glob patterns."""

    base = directory.resolve()
    candidates: set[Path] = set()
    for pattern in patterns:
        candidates.update(p for p in base.rglob(pattern) if p.is_file())
    resolved = {path.resolve() for path in candidates}
    return sorted(resolved, key=lambda p: str(p.relative_to(base)))


def _read_directory_source(path: Path) -> str | None:
    """Read UTF-8 text from `path`, skipping binary or empty files."""

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("kg_build.skip_binary", path=str(path))
        return None
    if not content.strip():
        logger.warning("kg_build.skip_empty", path=str(path))
        return None
    return content


def _compute_checksum(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_chunk_metadata(
    chunks: Sequence[Any],
    *,
    relative_path: str,
    git_commit: str | None,
) -> list[ChunkMetadata]:
    metadata: list[ChunkMetadata] = []
    for sequence, chunk in enumerate(chunks, start=1):
        text = getattr(chunk, "text", "") or ""
        index = getattr(chunk, "index", sequence - 1)
        uid = getattr(chunk, "uid", None)
        if uid is None:
            raise ValueError("chunk object missing uid; cannot attribute metadata")
        metadata.append(
            ChunkMetadata(
                uid=uid,
                sequence=sequence,
                index=index,
                checksum=_compute_checksum(text),
                relative_path=relative_path,
                git_commit=git_commit,
            )
        )
    return metadata


def _ensure_jsonable(value: Any) -> Any:
    """
    Coerce an arbitrary Python value into a JSON-serializable structure.
    
    Parameters:
        value (Any): The input to convert. May be a primitive, mapping, sequence, or any other object.
    
    Returns:
        Any: A JSON-serializable representation of `value`:
            - primitives and `None` are returned unchanged,
            - mappings become dicts with string keys and JSONable values,
            - lists/tuples/sets become lists of JSONable items,
            - all other objects are converted to their string representation.
    """

    if value is None or isinstance(value, PRIMITIVE_TYPES):
        return value
    if isinstance(value, Mapping):
        return {str(key): _ensure_jsonable(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_ensure_jsonable(item) for item in value]
    return str(value)


def _sanitize_property_value(value: Any) -> Any:
    """
    Convert a value into a Neo4j-safe property form (a primitive, a homogeneous list of primitives, or a JSON string).
    
    If the input is None or a primitive type (str, int, float, bool), it is returned unchanged. Sequences (list/tuple/set) are recursively sanitized: if all non-None elements coerce to the same primitive type, a list of those primitives is returned; otherwise the original sequence is serialized to a JSON string. Mappings are serialized to a JSON string. Any other value is coerced to its string representation.
    
    Parameters:
        value (Any): The value to sanitize for Neo4j property storage.
    
    Returns:
        Any: One of the following:
            - None or a primitive (str, int, float, bool) when the value is already primitive or None.
            - A list of primitives when a sequence contains homogeneous primitive elements.
            - A JSON string for mappings or heterogeneous/complex sequences.
            - A string for other non-serializable objects.
    """

    if value is None or isinstance(value, PRIMITIVE_TYPES):
        return value
    if isinstance(value, (list, tuple, set)):
        sanitized_items = []
        coerced_type: type[Any] | None = None
        for item in value:
            sanitised = _sanitize_property_value(item)
            if isinstance(sanitised, list):
                return json.dumps(_ensure_jsonable(value), sort_keys=True)
            if sanitised is None:
                continue
            if coerced_type is None:
                coerced_type = type(sanitised)
            if coerced_type not in PRIMITIVE_TYPES or type(sanitised) is not coerced_type:
                return json.dumps(_ensure_jsonable(value), sort_keys=True)
            sanitized_items.append(sanitised)
        return sanitized_items
    if isinstance(value, Mapping):
        return json.dumps(_ensure_jsonable(value), sort_keys=True)
    return str(value)


class SanitizingNeo4jWriter(Neo4jWriter):
    """Neo4j writer that coerces complex properties into Neo4j-friendly primitives."""

    def _sanitize_properties(self, properties: Mapping[str, Any]) -> dict[str, Any]:
        """
        Sanitize a mapping of node/relationship properties into Neo4j-friendly primitives.
        
        Iterates over the provided properties, converts each value using _sanitize_property_value, omits entries whose sanitized value is None, and returns a dictionary with stringified keys and sanitized values.
        
        Parameters:
            properties (Mapping[str, Any]): Original property mapping to sanitize.
        
        Returns:
            dict[str, Any]: A new dictionary containing only properties with sanitized, JSON/Neo4j-compatible values and string keys.
        """
        sanitized: dict[str, Any] = {}
        for key, value in properties.items():
            clean_value = _sanitize_property_value(value)
            if clean_value is None:
                continue
            sanitized[str(key)] = clean_value
        return sanitized

    def _nodes_to_rows(self, nodes, lexical_graph_config):  # type: ignore[override]
        """
        Convert node objects into row dictionaries and sanitize each row's `properties` for JSON- and Neo4j-friendly values.
        
        Parameters:
            nodes: An iterable of node objects to be converted into rows.
            lexical_graph_config: Configuration used by the base conversion process (passed through to the superclass).
        
        Returns:
            rows (list[dict]): A list of row dictionaries as produced by the superclass, with each row's "properties" replaced by a sanitized mapping suitable for Neo4j storage and JSON serialization.
        """
        rows = super()._nodes_to_rows(nodes, lexical_graph_config)
        for row in rows:
            properties = row.get("properties") or {}
            row["properties"] = self._sanitize_properties(properties)
        return rows

    def _relationships_to_rows(self, relationships):  # type: ignore[override]
        """
        Transform relationship objects into row dictionaries and sanitize each row's `properties` mapping.
        
        Parameters:
            relationships: An iterable of relationship objects to convert into rows.
        
        Returns:
            rows (list[dict]): A list of row dictionaries for each relationship where the `properties`
            entry has been sanitized into JSON/Neo4j-friendly primitive values.
        """
        rows = super()._relationships_to_rows(relationships)
        for row in rows:
            properties = row.get("properties") or {}
            row["properties"] = self._sanitize_properties(properties)
        return rows

    @validate_call
    async def run(  # type: ignore[override]
        self,
        graph,
        lexical_graph_config: LexicalGraphConfig | None = None,
    ) -> KGWriterModel:
        """
        Run the writer against the provided lexical graph using the given configuration.
        
        Parameters:
        	lexical_graph_config (LexicalGraphConfig): Configuration that controls how lexical graph elements are translated into nodes and relationships; used to influence property/label mapping and other writer behavior.
        
        Returns:
        	KGWriterModel: Model summarizing the result of the write operation (nodes/relationships created or updated and related metadata).
        """
        if lexical_graph_config is None:
            lexical_graph_config = LexicalGraphConfig()
        return await super().run(graph, lexical_graph_config)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the KG build script.
    
    Accepts an optional list of argument strings (typically None to use sys.argv) and returns a Namespace containing the parsed options:
    - source: path to the sample content to ingest.
    - chunk_size: character chunk size for the text splitter.
    - chunk_overlap: character overlap between chunks.
    - database: optional Neo4j database name (None uses server default).
    - log_path: file path for the structured JSON run log.
    """
    parser = argparse.ArgumentParser(
        description="Run the SimpleKGPipeline against local sample content, persisting results to Neo4j with structured logging and retries.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Path to a single content file to ingest (default: %(default)s)",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Directory containing files to ingest; overrides --source when provided.",
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        dest="include_patterns",
        help="Glob pattern (relative to --source-dir) to include. Can be provided multiple times.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PRESETS.keys()),
        default=None,
        help="Chunking profile to apply (sets default chunk size, overlap, and include patterns).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Character chunk size for the text splitter (overrides profile/default).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Character overlap between chunks (overrides profile/default).",
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
    parser.add_argument(
        "--reset-database",
        action="store_true",
        help="Delete all nodes and relationships in Neo4j before ingesting (destructive).",
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
    if not content:
        logger.warning("kg_build.empty_source", path=str(path))
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

    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "text"):
        text = _coerce_text(getattr(value, "text"))
        if text:
            return text
    if hasattr(value, "input_text"):
        text = _coerce_text(getattr(value, "input_text"))
        if text:
            return text
    if hasattr(value, "value"):
        text = _coerce_text(getattr(value, "value"))
        if text:
            return text
    if hasattr(value, "content") and not isinstance(value, Mapping):
        content = getattr(value, "content")
        if isinstance(content, str) and content:
            return content
    if isinstance(value, Mapping):
        for key in ("text", "input_text", "value", "content"):
            inner = value.get(key)
            if inner is None:
                continue
            text = _coerce_text(inner)
            if text:
                return text
    return None


def _normalise_choice_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts = [part for item in content if (part := _coerce_text(item))]
        if parts:
            return "".join(parts)
    text = _coerce_text(content)
    return text or ""


def _content_from_completion(completion: Any) -> str:
    for choice in getattr(completion, "choices", []):
        message = getattr(choice, "message", None)
        if message is None:
            continue
        content = getattr(message, "content", None)
        text = _normalise_choice_content(content)
        if text:
            return text
    return ""


def _content_from_payload(payload: Any) -> str:
    if isinstance(payload, Mapping):
        choices = payload.get("choices") or []
    else:
        choices = getattr(payload, "choices", [])
    for choice in choices:
        if isinstance(choice, Mapping):
            message = choice.get("message")
        else:
            message = getattr(choice, "message", None)
        if message is None:
            continue
        if isinstance(message, Mapping):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        text = _normalise_choice_content(content)
        if text:
            return text
    return ""


def _extract_content(raw_response: Any) -> str:
    """Extract textual content from a chat-completion style response."""

    payload = raw_response
    if hasattr(raw_response, "model_dump"):
        payload = raw_response.model_dump()
    elif hasattr(raw_response, "to_dict"):
        payload = raw_response.to_dict()

    if OpenAIChatCompletion is not None:
        if isinstance(raw_response, OpenAIChatCompletion):
            text = _content_from_completion(raw_response)
            if text:
                return text
        else:
            try:
                completion = OpenAIChatCompletion.model_validate(payload)
            except ValidationError:
                completion = None
            if completion is not None:
                text = _content_from_completion(completion)
                if text:
                    return text

    return _content_from_payload(payload)


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


def _collect_counts(driver, *, database: str | None) -> Mapping[str, int]:
    """
    Return counts of Document nodes, Chunk nodes, and HAS_CHUNK relationships from the Neo4j database.
    
    Queries three counts ("documents", "chunks", "relationships") against the provided driver and returns a mapping from those keys to integer counts. Keys are included only for queries that returned a usable result; a failed query will be skipped (logged) and not appear in the result.
    
    Parameters:
        database (str | None): Optional database name to run the queries against. Use None for the driver's default database.
    
    Returns:
        Mapping[str, int]: A mapping with any of the keys `"documents"`, `"chunks"`, and `"relationships"` mapped to their respective integer counts.
    """

    queries = {
        "documents": "MATCH (:Document) RETURN count(*) AS value",
        "chunks": "MATCH (:Chunk) RETURN count(*) AS value",
        "relationships": "MATCH (:Document)-[:HAS_CHUNK]->(:Chunk) RETURN count(*) AS value",
    }
    counts: dict[str, int] = {}
    for key, query in queries.items():
        try:
            result = driver.execute_query(query, database_=database)
            records = result
            if isinstance(result, tuple):
                records = result[0]
            else:
                records = getattr(result, "records", result)
            if not records:
                continue
            record = records[0]
            if isinstance(record, Mapping):
                value = record.get("value")
            elif hasattr(record, "value"):
                value = getattr(record, "value")
            else:
                try:
                    value = record[0]  # type: ignore[index]
                except Exception:  # pragma: no cover - defensive guard
                    value = None
            counts[key] = int(value or 0)
        except Neo4jError:
            logger.warning("kg_build.count_failed", query=key)
    return counts


def _reset_database(driver, *, database: str | None) -> None:
    """Remove previously ingested nodes to guarantee a clean ingest for the run."""

    driver.execute_query("MATCH (n) DETACH DELETE n", database_=database)


def _execute_pipeline(
    *,
    uri: str,
    auth: tuple[str, str],
    source_text: str,
    database: str | None,
    embedder: Embedder,
    llm: SharedOpenAILLM,
    splitter: FixedSizeSplitter,
    reset_database: bool,
) -> str | None:
    """
    Execute the knowledge-graph pipeline against a Neo4j instance and return the pipeline run identifier.
    
    When requested, this call resets the target Neo4j database before running the pipeline and writes sanitized nodes and relationships via the provided writer and components.
    
    Parameters:
        database (str | None): Name of the Neo4j database to use; pass `None` to use the server default.
        reset_database (bool): When True, remove all nodes and relationships prior to running the pipeline.
    
    Returns:
        run_id (str | None): The pipeline run identifier if produced, `None` otherwise.
    """

    with GraphDatabase.driver(uri, auth=auth) as driver:
        if reset_database:
            _reset_database(driver, database=database)
        writer = SanitizingNeo4jWriter(driver=driver, neo4j_database=database)
        pipeline = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            embedder=embedder,
            schema=DEFAULT_SCHEMA,
            from_pdf=False,
            text_splitter=splitter,
            kg_writer=writer,
            neo4j_database=database,
        )

        async def _run() -> str | None:
            """
            Execute the configured pipeline on the prepared source text and return its run identifier.
            
            Returns:
                run_id (str | None): Identifier of the completed pipeline run, or None if a run ID was not produced.
            """
            result = await pipeline.run_async(text=source_text)
            return result.run_id

        return asyncio.run(_run())


def _ensure_document_relationships(
    driver,
    *,
    database: str | None,
    source_path: Path,
    relative_path: str,
    git_commit: str | None,
    document_checksum: str,
    chunks_metadata: Sequence[ChunkMetadata],
) -> None:
    """
    Ensure a Document node exists for the given source file and link all Chunk nodes to it with HAS_CHUNK relationships.
    
    Creates or merges a Document node with its name and title set to the source file name on creation, sets each Chunk.node's `source_path` property to the provided path, assigns a sequential `chunk_id` when one is not already present, and creates a HAS_CHUNK relationship from the Document to each Chunk.
    
    Parameters:
        database (str | None): Optional Neo4j database name to execute the query against; pass None to use the default.
        source_path (Path): Filesystem path of the source document; used as the Document.source_path value and to derive the Document.name/title.
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
        // Chunk.uid values are globally unique per pipeline execution, so we can safely match on them
        MATCH (chunk:Chunk {uid: meta.uid})
        WITH doc, chunk, meta
        // Update per-chunk provenance while preserving existing identifiers when re-ingesting
        SET chunk.source_path = $source_path,
            chunk.relative_path = meta.relative_path,
            chunk.git_commit = meta.git_commit,
            chunk.checksum = meta.checksum
        FOREACH (_ IN CASE WHEN chunk.chunk_id IS NULL THEN [1] ELSE [] END |
            SET chunk.chunk_id = meta.sequence
        )
        FOREACH (_ IN CASE WHEN chunk.index IS NULL THEN [1] ELSE [] END |
            SET chunk.index = meta.index
        )
        // Ensure the Document ↔ Chunk relationship exists
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


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Builds a knowledge graph from a source file and writes a structured JSON run log.
    
    Parses CLI arguments (or uses provided argv), validates environment and chunking parameters, ingests the source text into Neo4j using configured OpenAI clients and the pipeline, optionally resets the database when requested, ensures document-chunk relationships, collects database counts, writes a sanitized JSON log to disk, prints the log, and returns the log dictionary.
    
    Parameters:
        argv (Sequence[str] | None): Optional list of CLI arguments to override sys.argv; when None the process uses default argument parsing.
    
    Returns:
        dict[str, Any]: A structured run log containing keys including:
            - timestamp: ISO 8601 UTC timestamp of completion
            - operation: the operation name ("kg_build")
            - status: operation status ("success" on normal completion)
            - duration_ms: elapsed time in milliseconds
            - source: path to the input source file
            - input_bytes: size of the input in bytes
            - chunking: mapping with "size" and "overlap" used for splitting
            - database: Neo4j database name (or None)
            - reset_database: boolean indicating whether the destructive reset flag was provided
            - openai: OpenAI settings used (chat_model, embedding_model, embedding_dimensions, max_attempts)
            - counts: mapping with counts of ingested entities (documents, chunks, relationships)
            - run_id: pipeline run identifier (if available)
    
    Raises:
        RuntimeError: if OpenAI requests or Neo4j operations fail.
    """
    args = _parse_args(argv)

    profile = args.profile or DEFAULT_PROFILE
    preset = PROFILE_PRESETS.get(profile, PROFILE_PRESETS[DEFAULT_PROFILE])
    chunk_size = args.chunk_size if args.chunk_size is not None else preset["chunk_size"]
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else preset["chunk_overlap"]
    include_patterns = tuple(args.include_patterns) if args.include_patterns else tuple(preset.get("include", ()))

    chunk_size = _ensure_positive(chunk_size, name="chunk_size")
    chunk_overlap = _ensure_non_negative(chunk_overlap, name="chunk_overlap")

    ensure_env("OPENAI_API_KEY")
    ensure_env("NEO4J_URI")
    ensure_env("NEO4J_USERNAME")
    ensure_env("NEO4J_PASSWORD")

    git_commit = _resolve_git_commit()

    source_specs: list[SourceSpec]
    if args.source_dir:
        directory = Path(args.source_dir).expanduser()
        if not directory.is_dir():
            raise ValueError(f"source directory not found: {directory}")
        patterns = include_patterns or preset.get("include", ())
        files = _discover_source_files(directory, patterns)
        source_specs = []
        for file_path in files:
            content = _read_directory_source(file_path)
            if content is None:
                continue
            source_specs.append(
                SourceSpec(
                    path=file_path,
                    relative_path=_relative_to_repo(file_path, base=directory),
                    text=content,
                    checksum=_compute_checksum(content),
                )
            )
        if not source_specs:
            raise ValueError(
                "No ingestible files matched the supplied directory and include patterns."
            )
    else:
        source_path = Path(args.source).expanduser()
        source_text = _read_source(source_path)
        source_specs = [
            SourceSpec(
                path=source_path,
                relative_path=_relative_to_repo(source_path),
                text=source_text,
                checksum=_compute_checksum(source_text),
            )
        ]

    settings = OpenAISettings.load(actor="kg_build")
    shared_client = SharedOpenAIClient(settings)
    embedder = SharedOpenAIEmbedder(shared_client, settings)
    llm = SharedOpenAILLM(shared_client, settings)
    splitter = CachingFixedSizeSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    uri = os.environ["NEO4J_URI"]
    auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])

    start = time.perf_counter()
    run_ids: list[str | None] = []
    log_files: list[dict[str, Any]] = []
    log_chunks: list[dict[str, Any]] = []
    reset_pending = bool(args.reset_database)

    counts: Mapping[str, int] = {}
    with GraphDatabase.driver(uri, auth=auth) as driver:
        for spec in source_specs:
            chunk_result = asyncio.run(splitter.run(spec.text))
            chunk_metadata = _build_chunk_metadata(
                chunk_result.chunks,
                relative_path=spec.relative_path,
                git_commit=git_commit,
            )
            try:
                run_id = _execute_pipeline(
                    uri=uri,
                    auth=auth,
                    source_text=spec.text,
                    database=args.database,
                    embedder=embedder,
                    llm=llm,
                    splitter=splitter,
                    reset_database=reset_pending,
                )
            except (OpenAIClientError, LLMGenerationError, EmbeddingsGenerationError) as exc:
                raise RuntimeError("OpenAI request failed") from exc
            except (Neo4jError, ClientError) as exc:
                raise RuntimeError("Neo4j error") from exc
            run_ids.append(run_id)
            reset_pending = False

            _ensure_document_relationships(
                driver,
                database=args.database,
                source_path=spec.path,
                relative_path=spec.relative_path,
                git_commit=git_commit,
                document_checksum=spec.checksum,
                chunks_metadata=chunk_metadata,
            )

            log_files.append(
                {
                    "source": str(spec.path),
                    "relative_path": spec.relative_path,
                    "checksum": spec.checksum,
                    "chunks": len(chunk_metadata),
                }
            )
            for meta in chunk_metadata:
                log_chunks.append(
                    {
                        "source": str(spec.path),
                        "relative_path": meta.relative_path,
                        "chunk_index": meta.index,
                        "chunk_id": meta.sequence,
                        "checksum": meta.checksum,
                        "git_commit": meta.git_commit,
                    }
                )

        counts = _collect_counts(driver, database=args.database)

    duration_ms = int((time.perf_counter() - start) * 1000)

    total_bytes = sum(len(spec.text.encode("utf-8")) for spec in source_specs)
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "kg_build",
        "status": "success",
        "duration_ms": duration_ms,
        "source": str(Path(args.source_dir).expanduser()) if args.source_dir else str(source_specs[0].path),
        "source_mode": "directory" if args.source_dir else "file",
        "input_bytes": total_bytes,
        "chunking": {
            "size": chunk_size,
            "overlap": chunk_overlap,
            "profile": profile,
            "include_patterns": list(include_patterns),
        },
        "database": args.database,
        "reset_database": bool(args.reset_database),
        "openai": {
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "max_attempts": settings.max_attempts,
        },
        "counts": counts,
        "run_id": run_ids[-1] if run_ids else None,
        "run_ids": [run_id for run_id in run_ids if run_id],
        "files": log_files,
        "chunks": log_chunks,
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
