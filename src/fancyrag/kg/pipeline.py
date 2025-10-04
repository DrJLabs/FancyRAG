"""Pipeline orchestration utilities for FancyRAG knowledge graph ingestion."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, Neo4jError

from _compat.structlog import get_logger
from cli.openai_client import OpenAIClientError, SharedOpenAIClient
from cli.sanitizer import scrub_object
from cli.utils import ensure_embedding_dimensions
from config.settings import OpenAISettings
from fancyrag.qa import (
    IngestionQaEvaluator,
    QaChunkRecord,
    QaSourceRecord,
    QaThresholds,
    collect_counts as _collect_counts,
)
from fancyrag.splitters import CachingSplitterConfig, build_caching_splitter
from fancyrag.utils import (
    ensure_directory as _ensure_directory,
    ensure_env,
    relative_to_repo as _relative_to_repo,
    resolve_repo_root as _resolve_repo_root,
)

_PYDANTIC_AVAILABLE = importlib.util.find_spec("pydantic") is not None

if _PYDANTIC_AVAILABLE:  # pragma: no branch - import-time dependency check
    from pydantic import ValidationError, validate_call
else:  # pragma: no cover - exercised only in minimal CI environments
    class ValidationError(ValueError):  # type: ignore[no-redef]
        """Fallback validation error when pydantic is unavailable."""


    def validate_call(func=None, **_kwargs):  # type: ignore[no-redef]
        """Simplified validate_call decorator that returns the function unchanged."""

        if func is None:
            return lambda wrapped: wrapped
        return func

def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # pragma: no cover - importlib quirk when parent is module
        return False


_GRAPHRAG_MODULES = [
    "neo4j_graphrag.embeddings.base",
    "neo4j_graphrag.exceptions",
    "neo4j_graphrag.experimental.components.entity_relation_extractor",
    "neo4j_graphrag.experimental.components.kg_writer",
    "neo4j_graphrag.experimental.components.lexical_graph",
    "neo4j_graphrag.experimental.components.schema",
    "neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter",
    "neo4j_graphrag.experimental.components.types",
    "neo4j_graphrag.experimental.pipeline.kg_builder",
    "neo4j_graphrag.llm.base",
    "neo4j_graphrag.llm.types",
]

_GRAPHRAG_AVAILABLE = all(_module_available(module_name) for module_name in _GRAPHRAG_MODULES)

if _GRAPHRAG_AVAILABLE:  # pragma: no branch - import-time dependency check
    from neo4j_graphrag.embeddings.base import Embedder
    from neo4j_graphrag.exceptions import EmbeddingsGenerationError, LLMGenerationError
    from neo4j_graphrag.experimental.components.entity_relation_extractor import (
        LLMEntityRelationExtractor,
        OnError,
    )
    from neo4j_graphrag.experimental.components.kg_writer import (
        KGWriterModel,
        Neo4jWriter,
    )
    from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphConfig
    from neo4j_graphrag.experimental.components.schema import GraphSchema
    from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
        FixedSizeSplitter,
    )
    from neo4j_graphrag.experimental.components.types import (
        TextChunk,
        TextChunks,
        Neo4jGraph,
    )
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
    from neo4j_graphrag.llm.base import LLMInterface
    from neo4j_graphrag.llm.types import LLMResponse
else:  # pragma: no cover - exercised only in minimal CI environments
    class _GraphRagMissingDependency(ModuleNotFoundError):
        """Raised when neo4j_graphrag functionality is required but unavailable."""


    class Embedder:  # type: ignore[no-redef]
        """Placeholder Embedder base when neo4j_graphrag is absent."""


    class EmbeddingsGenerationError(RuntimeError):  # type: ignore[no-redef]
        """Fallback embeddings error used when neo4j_graphrag is unavailable."""


    class LLMGenerationError(RuntimeError):  # type: ignore[no-redef]
        """Fallback LLM error used when neo4j_graphrag is unavailable."""


    class LLMInterface:  # type: ignore[no-redef]
        """Minimal LLM interface placeholder for SharedOpenAILLM."""

        def __init__(
            self,
            *_args,
            model_name: str | None = None,
            model_params: Mapping[str, Any] | None = None,
            **_kwargs,
        ) -> None:
            self.model_name = model_name
            self.model_params = dict(model_params or {})


    @dataclass
    class LLMResponse:  # type: ignore[no-redef]
        """Lightweight response container mirroring neo4j_graphrag.llm.types.LLMResponse."""

        content: str | None = None


    class OnError(Enum):  # type: ignore[no-redef]
        """Subset of the neo4j_graphrag OnError enumeration."""

        RAISE = "raise"


    class LLMEntityRelationExtractor:  # type: ignore[no-redef]
        """Placeholder extractor that raises when semantic enrichment is requested without dependencies."""

        def __init__(self, *_, **__) -> None:
            raise _GraphRagMissingDependency(
                "neo4j_graphrag is required for semantic enrichment support"
            )


    @dataclass
    class KGWriterModel:  # type: ignore[no-redef]
        """Minimal writer model capturing node/relationship counts."""

        nodes_created: int = 0
        relationships_created: int = 0


    class Neo4jWriter:  # type: ignore[no-redef]
        """Fallback writer with inert behavior when neo4j_graphrag is unavailable."""

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def run(self, *_args, **_kwargs) -> KGWriterModel:
            raise _GraphRagMissingDependency(
                "neo4j_graphrag is required for Neo4j writer execution"
            )

        def _nodes_to_rows(self, *_args, **_kwargs) -> list[dict[str, Any]]:
            return []

        def _relationships_to_rows(self, *_args, **_kwargs) -> list[dict[str, Any]]:
            return []


    class LexicalGraphConfig:  # type: ignore[no-redef]
        """Placeholder lexical graph configuration object."""


    class GraphSchema:  # type: ignore[no-redef]
        """Placeholder schema returned when neo4j_graphrag is unavailable."""

        @classmethod
        def model_validate(cls, *_args, **_kwargs) -> "GraphSchema":
            return cls()


    @dataclass
    class TextChunk:  # type: ignore[no-redef]
        """Simple text chunk representation used for caching in tests."""

        text: str
        index: int
        metadata: Any | None = None
        uid: str = field(default_factory=lambda: str(uuid4()))


    @dataclass
    class TextChunks:  # type: ignore[no-redef]
        """Container matching the interface expected from neo4j_graphrag splitters."""

        chunks: list[TextChunk]


    @dataclass
    class Neo4jGraph:  # type: ignore[no-redef]
        """Lightweight graph container for semantic enrichment statistics."""

        nodes: list[Any] = field(default_factory=list)
        relationships: list[Any] = field(default_factory=list)


    class FixedSizeSplitter:  # type: ignore[no-redef]
        """Simple splitter that yields one chunk per input string when dependencies are absent."""

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def run(self, text: str | Sequence[str], *_args, **_kwargs) -> TextChunks:
            if isinstance(text, str):
                items = [text]
            else:
                items = list(text)
            chunks = [
                TextChunk(text=item, index=index, metadata=None) for index, item in enumerate(items)
            ]
            return TextChunks(chunks=chunks)


    class SimpleKGPipeline:  # type: ignore[no-redef]
        """Pipeline stub that raises when executed without neo4j_graphrag installed."""

        def __init__(self, *_args, **_kwargs) -> None:
            raise _GraphRagMissingDependency(
                "neo4j_graphrag is required for the KG builder pipeline"
            )

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
PROJECT_ROOT = _resolve_repo_root() or Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "scripts" / "config" / "kg_schema.json"

DEFAULT_PROFILE = "text"
QA_REPORT_VERSION = "ingestion-qa-report/v1"
DEFAULT_QA_DIR = Path("artifacts/ingestion")
SEMANTIC_SOURCE = "kg_build.semantic_enrichment.v1"
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


@dataclass(frozen=True)
class QaLimits:
    """Configuration for ingestion QA gating thresholds."""

    max_missing_embeddings: int = 0
    max_orphan_chunks: int = 0
    max_checksum_mismatches: int = 0
    max_semantic_failures: int = 0
    max_semantic_orphans: int = 0


@dataclass(frozen=True)
class PipelineOptions:
    """Inputs required to orchestrate the KG build pipeline."""

    source: Path
    source_dir: Path | None
    include_patterns: Sequence[str] | None
    profile: str | None
    chunk_size: int | None
    chunk_overlap: int | None
    database: str | None
    log_path: Path
    qa_report_dir: Path
    qa_limits: QaLimits
    semantic_enabled: bool
    semantic_max_concurrency: int
    reset_database: bool


@dataclass
class SourceSpec:
    """Represents a resolved ingestion source."""

    path: Path
    relative_path: str
    text: str
    checksum: str


@dataclass
class ChunkMetadata:
    """Metadata captured for each chunk written to Neo4j/Qdrant."""

    uid: str
    sequence: int
    index: int
    checksum: str
    relative_path: str
    git_commit: str | None


@dataclass
class SemanticEnrichmentStats:
    """Aggregated metrics describing semantic enrichment output."""

    chunks_processed: int = 0
    chunk_failures: int = 0
    nodes_written: int = 0
    relationships_written: int = 0


@dataclass
class SemanticQaSummary:
    """Summary passed into QA evaluation for semantic enrichment metrics."""

    enabled: bool = False
    chunks_processed: int = 0
    chunk_failures: int = 0
    nodes_written: int = 0
    relationships_written: int = 0
    source_tag: str = SEMANTIC_SOURCE



def _load_default_schema(path: Path = DEFAULT_SCHEMA_PATH) -> GraphSchema:
    """
    Load and return the validated default GraphSchema from disk.
    
    Parameters:
        path (Path): Path to a JSON file containing the GraphSchema definition (defaults to DEFAULT_SCHEMA_PATH).
    
    Returns:
        GraphSchema: A validated GraphSchema instance constructed from the file contents.
    
    Raises:
        RuntimeError: If the schema file is missing or contains invalid JSON.
    """

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
    """
    Builds per-chunk provenance metadata from a sequence of chunk-like objects.
    
    Each chunk object must have a `uid` attribute; `text` and `index` are read if present. The function computes a checksum from the chunk text and returns a list of ChunkMetadata with sequence order, index, uid, checksum, relative path, and git commit.
    
    Parameters:
        chunks: An ordered sequence of objects representing chunks. Each object must expose a `uid` attribute; `text` and `index` attributes are optional.
        relative_path: File path (relative to the repository or input base) to associate with each chunk's provenance.
        git_commit: Git commit SHA to associate with each chunk's provenance, or `None` if unavailable.
    
    Returns:
        A list of ChunkMetadata records, one per input chunk, preserving the input sequence.
    
    Raises:
        ValueError: If any chunk lacks a `uid`, or if duplicate `uid` values are encountered.
    """
    metadata: list[ChunkMetadata] = []
    seen_uids: set[str] = set()
    for sequence, chunk in enumerate(chunks, start=1):
        text = getattr(chunk, "text", "") or ""
        index = getattr(chunk, "index", sequence - 1)
        uid = getattr(chunk, "uid", None)
        if uid is None:
            raise ValueError("chunk object missing uid; cannot attribute metadata")
        if uid in seen_uids:
            raise ValueError(
                f"duplicate chunk uid detected while building metadata for ingestion: {uid}"
            )
        seen_uids.add(uid)
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


def _annotate_semantic_graph(
    graph: Neo4jGraph,
    *,
    chunk_metadata: Mapping[str, ChunkMetadata],
    relative_path: str,
    git_commit: str | None,
    document_checksum: str,
) -> None:
    """Attach provenance metadata to semantic nodes and relationships."""

    for node in graph.nodes:
        props = dict(node.properties or {})
        prefix, _, _ = node.id.partition(":")
        meta = chunk_metadata.get(node.id)
        if meta is None and prefix:
            meta = chunk_metadata.get(prefix)
        props.setdefault("relative_path", relative_path)
        if git_commit is not None:
            props.setdefault("git_commit", git_commit)
        props.setdefault("document_checksum", document_checksum)
        if meta is not None:
            props.setdefault("chunk_uid", meta.uid)
            props.setdefault("chunk_checksum", meta.checksum)
        props.setdefault("semantic_source", SEMANTIC_SOURCE)
        node.properties = props

    for relationship in graph.relationships:
        props = dict(relationship.properties or {})
        start_prefix, _, _ = relationship.start_node_id.partition(":")
        end_prefix, _, _ = relationship.end_node_id.partition(":")
        related_metas = [
            chunk_metadata[prefix]
            for prefix in (relationship.start_node_id, relationship.end_node_id, start_prefix, end_prefix)
            if prefix and prefix in chunk_metadata
        ]
        props.setdefault("relative_path", relative_path)
        if git_commit is not None:
            props.setdefault("git_commit", git_commit)
        props.setdefault("document_checksum", document_checksum)
        resolved_uids = sorted({meta.uid for meta in related_metas})
        if resolved_uids:
            props.setdefault("chunk_uids", resolved_uids)
        props.setdefault("semantic_source", SEMANTIC_SOURCE)
        relationship.properties = props


def _run_semantic_enrichment(
    *,
    driver,
    database: str | None,
    llm: SharedOpenAILLM,
    chunk_result: TextChunks,
    chunk_metadata: Sequence[ChunkMetadata],
    relative_path: str,
    git_commit: str | None,
    document_checksum: str,
    ingest_run_key: str | None,
    max_concurrency: int,
) -> SemanticEnrichmentStats:
    """Execute semantic entity extraction for the provided chunks and persist results."""

    stats = SemanticEnrichmentStats(chunks_processed=len(chunk_result.chunks))
    if not chunk_result.chunks:
        return stats

    extractor = LLMEntityRelationExtractor(
        llm=llm,
        create_lexical_graph=False,
        on_error=OnError.RAISE,
        max_concurrency=max_concurrency,
    )
    chunk_meta_lookup: dict[str, ChunkMetadata] = {}
    for meta in chunk_metadata:
        chunk_meta_lookup[meta.uid] = meta
        chunk_meta_lookup[str(meta.sequence)] = meta
        if meta.index is not None:
            chunk_meta_lookup[str(meta.index)] = meta
        prefix_uid, _, _ = str(meta.uid).partition(":")
        if prefix_uid:
            chunk_meta_lookup[prefix_uid] = meta

    async def _extract() -> tuple[Neo4jGraph, int]:
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _extract_and_process(chunk: TextChunk) -> Neo4jGraph | None:
            async with semaphore:
                try:
                    graph = await extractor.extract_for_chunk(DEFAULT_SCHEMA, "", chunk)
                except (LLMGenerationError, OpenAIClientError):
                    return None
                await extractor.post_process_chunk(graph, chunk)
                return graph

        tasks = [_extract_and_process(chunk) for chunk in chunk_result.chunks]
        results = await asyncio.gather(*tasks)

        chunk_graphs = [graph for graph in results if graph is not None]
        failures = len(results) - len(chunk_graphs)

        combined = extractor.combine_chunk_graphs(None, chunk_graphs)
        return combined, failures

    combined_graph, failures = asyncio.run(_extract())
    stats.chunk_failures = failures

    if not combined_graph.nodes and not combined_graph.relationships:
        return stats

    _annotate_semantic_graph(
        combined_graph,
        chunk_metadata=chunk_meta_lookup,
        relative_path=relative_path,
        git_commit=git_commit,
        document_checksum=document_checksum,
    )

    writer = SanitizingNeo4jWriter(driver=driver, neo4j_database=database)
    writer.set_ingest_run_key(ingest_run_key)
    asyncio.run(writer.run(combined_graph))

    stats.nodes_written = len(combined_graph.nodes)
    stats.relationships_written = len(combined_graph.relationships)
    return stats


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

    _INGEST_RUN_KEY_FIELD = "ingest_run_key"

    def set_ingest_run_key(self, run_key: str | None) -> None:
        """Attach the ingest run key used to tag nodes and relationships written by this writer."""

        self._ingest_run_key = run_key

    def _get_ingest_run_key(self) -> str | None:
        """Return the ingest run key associated with the current writer instance."""

        return getattr(self, "_ingest_run_key", None)

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
            sanitized = self._sanitize_properties(properties)
            run_key = self._get_ingest_run_key()
            if run_key:
                sanitized.setdefault(self._INGEST_RUN_KEY_FIELD, run_key)
            row["properties"] = sanitized
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
            sanitized = self._sanitize_properties(properties)
            run_key = self._get_ingest_run_key()
            if run_key:
                sanitized.setdefault(self._INGEST_RUN_KEY_FIELD, run_key)
            row["properties"] = sanitized
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
    validated = int(value)
    if validated < 0:
        raise ValueError(f"{name} must be zero or positive")
    return validated

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


_FENCE_PATTERN = re.compile(r"^```[ \t]*([^\n`]*)\s*(.*?)\s*```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Remove surrounding Markdown code fences from the given text."""

    stripped = text.strip()
    match = _FENCE_PATTERN.match(stripped)
    if not match:
        return stripped
    return match.group(2).strip()


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
        preview_len = 256
        logger.debug(
            "kg_build.llm_response",
            content_preview=content[:preview_len],
            content_length=len(content),
            truncated=len(content) > preview_len,
        )
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
    ingest_run_key: str | None = None,
) -> str | None:
    """
    Execute the knowledge-graph pipeline against a Neo4j instance and return the pipeline run identifier.

    When requested, this call resets the target Neo4j database before running the pipeline and writes sanitized nodes and relationships via the provided writer and components.

    Parameters:
        database (str | None): Name of the Neo4j database to use; pass `None` to use the server default.
        reset_database (bool): When True, remove all nodes and relationships prior to running the pipeline.
        ingest_run_key (str | None): Unique identifier applied to nodes and relationships written during this execution for rollback targeting.
    
    Returns:
        run_id (str | None): The pipeline run identifier if produced, `None` otherwise.
    """

    with GraphDatabase.driver(uri, auth=auth) as driver:
        if reset_database:
            _reset_database(driver, database=database)
        writer = SanitizingNeo4jWriter(driver=driver, neo4j_database=database)
        writer.set_ingest_run_key(ingest_run_key)
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
    Ensure a Document node exists for the given source file and attach the provided Chunk nodes to it with up-to-date provenance.
    
    Creates or reuses a Document node identified by the file path, updates document-level provenance (relative path, git commit, checksum), sets per-chunk provenance (source path, relative path, git commit, checksum) for each chunk, preserves existing chunk identifiers when present, and ensures a HAS_CHUNK relationship from the Document to each Chunk.
    
    Parameters:
        driver: Neo4j driver or wrapper used to execute the Cypher query.
        database (str | None): Optional Neo4j database name to run the query against; pass None to use the default.
        source_path (Path): Filesystem path of the source document used as the Document.source_path and to derive the document name/title.
        relative_path (str): Stable repository-relative path to store on the Document and chunks.
        git_commit (str | None): Git commit SHA to store as provenance on the Document and chunks, or None if unavailable.
        document_checksum (str): SHA-256 checksum representing the current document content/version.
        chunks_metadata (Sequence[ChunkMetadata]): Sequence of per-chunk provenance records (uid, sequence, index, relative_path, git_commit, checksum) to associate with the Document.
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


def _rollback_ingest(
    driver,
    *,
    database: str | None,
    sources: Sequence[QaSourceRecord],
) -> None:
    """
    Delete graph elements produced during the provided ingestion sources.

    Removes Chunk nodes whose `uid` values appear in `sources`, detaches and deletes Documents that become orphaned, and
    eliminates any additional nodes or relationships tagged with the ingestion run key associated with the sources. The
    deletions are performed in the specified Neo4j `database` (or the driver's default if `None`).

    Parameters:
        database (str | None): Target Neo4j database name, or `None` to use the driver's default.
        sources (Sequence[QaSourceRecord]): Sequence of QA source records describing the ingested artifacts to roll back.
    """

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


def run_pipeline(options: PipelineOptions) -> dict[str, Any]:
    """
    Orchestrates a knowledge-graph ingestion run using the provided pipeline options.
    
    Run the full pipeline for one or more source files: discover or read sources, chunk text,
    call OpenAI services for embeddings and optional semantic extraction, write chunk and
    semantic entities to Neo4j, perform ingestion QA, and produce a run log and QA reports.
    
    Parameters:
        options (PipelineOptions): Configuration for the ingestion run (sources, chunking,
            database and connection settings, QA thresholds, semantic enrichment options,
            logging and output paths).
    
    Returns:
        dict[str, Any]: A sanitized run log containing timestamps, status, duration, input
        sizes, chunking settings, OpenAI and database metadata, counts, run identifiers,
        per-file and per-chunk summaries, and optional semantic and QA sections.
    """

    profile = options.profile or DEFAULT_PROFILE
    preset = PROFILE_PRESETS.get(profile, PROFILE_PRESETS[DEFAULT_PROFILE])
    chunk_size = options.chunk_size if options.chunk_size is not None else preset["chunk_size"]
    chunk_overlap = (
        options.chunk_overlap if options.chunk_overlap is not None else preset["chunk_overlap"]
    )
    include_patterns = (
        tuple(options.include_patterns)
        if options.include_patterns
        else tuple(preset.get("include", ()))
    )

    chunk_size = _ensure_positive(chunk_size, name="chunk_size")
    chunk_overlap = _ensure_non_negative(chunk_overlap, name="chunk_overlap")
    semantic_max_concurrency = options.semantic_max_concurrency
    if options.semantic_enabled:
        semantic_max_concurrency = _ensure_positive(
            semantic_max_concurrency, name="semantic_max_concurrency"
        )

    ensure_env("OPENAI_API_KEY")
    ensure_env("NEO4J_URI")
    ensure_env("NEO4J_USERNAME")
    ensure_env("NEO4J_PASSWORD")

    git_commit = _resolve_git_commit()

    source_specs: list[SourceSpec]
    if options.source_dir:
        directory = Path(options.source_dir).expanduser()
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
        source_path = Path(options.source).expanduser()
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
    splitter_config = CachingSplitterConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitter = build_caching_splitter(splitter_config)

    uri = os.environ["NEO4J_URI"]
    auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])

    start = time.perf_counter()
    run_ids: list[str | None] = []
    log_files: list[dict[str, Any]] = []
    log_chunks: list[dict[str, Any]] = []
    qa_sources: list[QaSourceRecord] = []
    semantic_totals = SemanticEnrichmentStats()
    reset_pending = bool(options.reset_database)

    qa_section: dict[str, Any] | None = None
    counts: Mapping[str, int] | None = None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        for spec in source_specs:
            scope_token = str(spec.path.resolve())
            with splitter.scoped(scope_token):
                ingest_run_key = f"kg-build:{uuid4()}"
                try:
                    run_id = _execute_pipeline(
                        uri=uri,
                        auth=auth,
                        source_text=spec.text,
                        database=options.database,
                        embedder=embedder,
                        llm=llm,
                        splitter=splitter,
                        reset_database=reset_pending,
                        ingest_run_key=ingest_run_key,
                    )
                except (OpenAIClientError, LLMGenerationError, EmbeddingsGenerationError) as exc:
                    raise RuntimeError(f"OpenAI request failed: {exc}") from exc
                except (Neo4jError, ClientError) as exc:
                    raise RuntimeError(f"Neo4j error: {exc}") from exc
                run_ids.append(run_id)
                reset_pending = False

                chunk_result = splitter.get_cached(spec.text)
                if chunk_result is None:
                    chunk_result = asyncio.run(splitter.run(spec.text))
                chunk_metadata = _build_chunk_metadata(
                    chunk_result.chunks,
                    relative_path=spec.relative_path,
                    git_commit=git_commit,
                )

                _ensure_document_relationships(
                    driver,
                    database=options.database,
                    source_path=spec.path,
                    relative_path=spec.relative_path,
                    git_commit=git_commit,
                    document_checksum=spec.checksum,
                    chunks_metadata=chunk_metadata,
                )

                qa_chunks = [
                    QaChunkRecord(
                        uid=meta.uid,
                        checksum=meta.checksum,
                        text=getattr(chunk, "text", "") or "",
                    )
                    for meta, chunk in zip(chunk_metadata, chunk_result.chunks)
                ]
                qa_sources.append(
                    QaSourceRecord(
                        path=str(spec.path),
                        relative_path=spec.relative_path,
                        document_checksum=spec.checksum,
                        git_commit=git_commit,
                        chunks=qa_chunks,
                        ingest_run_key=ingest_run_key,
                    )
                )
                log_files.append(
                    {
                        "path": str(spec.path),
                        "relative_path": spec.relative_path,
                        "checksum": spec.checksum,
                        "chunks": len(chunk_metadata),
                    }
                )
                log_chunks.extend(
                    {
                        "path": str(spec.path),
                        "uid": chunk.uid,
                        "sequence": chunk.sequence,
                        "index": chunk.index,
                        "checksum": chunk.checksum,
                        "relative_path": chunk.relative_path,
                        "git_commit": chunk.git_commit,
                    }
                    for chunk in chunk_metadata
                )

                if options.semantic_enabled:
                    semantic_stats = _run_semantic_enrichment(
                        driver=driver,
                        database=options.database,
                        llm=llm,
                        chunk_result=chunk_result,
                        chunk_metadata=chunk_metadata,
                        relative_path=spec.relative_path,
                        git_commit=git_commit,
                        document_checksum=spec.checksum,
                        ingest_run_key=ingest_run_key,
                        max_concurrency=semantic_max_concurrency,
                    )
                    semantic_totals.chunks_processed += semantic_stats.chunks_processed
                    semantic_totals.chunk_failures += semantic_stats.chunk_failures
                    semantic_totals.nodes_written += semantic_stats.nodes_written
                    semantic_totals.relationships_written += semantic_stats.relationships_written

        thresholds = QaThresholds(
            max_missing_embeddings=options.qa_limits.max_missing_embeddings,
            max_orphan_chunks=options.qa_limits.max_orphan_chunks,
            max_checksum_mismatches=options.qa_limits.max_checksum_mismatches,
            max_semantic_failures=options.qa_limits.max_semantic_failures,
            max_semantic_orphans=options.qa_limits.max_semantic_orphans,
        )
        semantic_summary = SemanticQaSummary(
            enabled=bool(options.semantic_enabled),
            chunks_processed=semantic_totals.chunks_processed,
            chunk_failures=semantic_totals.chunk_failures,
            nodes_written=semantic_totals.nodes_written,
            relationships_written=semantic_totals.relationships_written,
        )
        evaluator = IngestionQaEvaluator(
            driver=driver,
            database=options.database,
            sources=qa_sources,
            thresholds=thresholds,
            report_root=options.qa_report_dir,
            report_version=QA_REPORT_VERSION,
            semantic_summary=semantic_summary,
        )

        qa_result = evaluator.evaluate()
        qa_section = {
            "status": qa_result.status,
            "summary": qa_result.summary,
            "report_version": qa_result.version,
            "report_json": qa_result.report_json,
            "report_markdown": qa_result.report_markdown,
            "thresholds": asdict(qa_result.thresholds),
            "metrics": qa_result.metrics,
            "anomalies": qa_result.anomalies,
            "duration_ms": qa_result.duration_ms,
        }

        if not qa_result.passed:
            _rollback_ingest(driver, database=options.database, sources=qa_sources)
            raise RuntimeError(
                "Ingestion QA gating failed; see ingestion QA report for details"
            )

        counts = qa_result.metrics.get("graph_counts", {})
        if not counts:
            counts = _collect_counts(driver, database=options.database)

    duration_ms = int((time.perf_counter() - start) * 1000)

    total_bytes = sum(len(spec.text.encode("utf-8")) for spec in source_specs)
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "kg_build",
        "status": "success",
        "duration_ms": duration_ms,
        "source": str(Path(options.source_dir).expanduser())
        if options.source_dir
        else str(source_specs[0].path),
        "source_mode": "directory" if options.source_dir else "file",
        "input_bytes": total_bytes,
        "chunking": {
            "size": chunk_size,
            "overlap": chunk_overlap,
            "profile": profile,
            "include_patterns": list(include_patterns),
        },
        "database": options.database,
        "reset_database": bool(options.reset_database),
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
    if semantic_summary.enabled:
        log["semantic"] = {
            "chunks_processed": semantic_summary.chunks_processed,
            "chunk_failures": semantic_summary.chunk_failures,
            "nodes_written": semantic_summary.nodes_written,
            "relationships_written": semantic_summary.relationships_written,
        }
    if qa_section is not None:
        log["qa"] = qa_section

    log_path = options.log_path
    _ensure_directory(log_path)
    sanitized = scrub_object(log)
    log_path.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    print(json.dumps(sanitized))
    logger.info(
        "kg_build.completed",
        status=sanitized.get("status"),
        duration_ms=sanitized.get("duration_ms"),
        source=sanitized.get("source"),
        source_mode=sanitized.get("source_mode"),
        run_id=sanitized.get("run_id"),
        files_count=len(sanitized.get("files", [])),
        chunks_count=len(sanitized.get("chunks", [])),
        qa_status=(sanitized.get("qa") or {}).get("status"),
    )
    return log
