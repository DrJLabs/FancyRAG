#!/usr/bin/env python
"""Run the SimpleKGPipeline against sample content for the minimal path workflow."""

from __future__ import annotations

import argparse
import asyncio
import copy
import functools
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
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
from fancyrag.utils import ensure_env

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
PROJECT_ROOT = Path(__file__).resolve().parents[3]
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


@dataclass
class SourceSpec:
    """Represents a resolved ingestion source."""

    path: Path
    relative_path: str
    text: str
    checksum: str


class CachingFixedSizeSplitter(FixedSizeSplitter):
    """Fixed-size splitter that caches results while yielding fresh chunk UIDs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._blueprints: dict[str | tuple[str, ...], list[dict[str, Any]]] = {}
        self._last_outputs: dict[str | tuple[str, ...], TextChunks] = {}
        self._scope_stack: list[str | None] = []

    @contextmanager
    def scoped(self, scope: str | Path | None):
        """Scope cache lookups to a specific source identifier."""

        scope_id = str(scope) if scope is not None else None
        self._scope_stack.append(scope_id)
        try:
            yield self
        finally:
            self._scope_stack.pop()

    def _current_scope(self) -> str | None:
        if not self._scope_stack:
            return None
        return self._scope_stack[-1]

    def _cache_key(self, text: str | Sequence[str]) -> str | tuple[str, ...]:
        if isinstance(text, str):
            base_key: str | tuple[str, ...] = text
        else:
            base_key = tuple(text)
        scope = self._current_scope()
        if scope is None:
            return base_key
        if isinstance(base_key, tuple):
            return (scope, *base_key)
        return (scope, base_key)

    async def run(
        self, text: str | Sequence[str], config: Any | None = None
    ) -> TextChunks:  # type: ignore[override]
        if config is not None:
            try:
                return await super().run(text, config)
            except (TypeError, ValidationError):  # pragma: no cover - fallback
                return await super().run(text)

        key = self._cache_key(text)
        blueprint = self._blueprints.get(key)
        if blueprint is None:
            result = await super().run(text)
            self._blueprints[key] = [
                {
                    "text": chunk.text,
                    "index": chunk.index,
                    "metadata": copy.deepcopy(getattr(chunk, "metadata", None)),
                }
                for chunk in result.chunks
            ]
            self._last_outputs[key] = result
            return result

        chunks: list[TextChunk] = []
        for template in blueprint:
            chunks.append(
                TextChunk(
                    text=template["text"],
                    index=template["index"],
                    metadata=copy.deepcopy(template["metadata"]),
                    uid=str(uuid4()),
                )
            )
        text_chunks = TextChunks(chunks=chunks)
        self._last_outputs[key] = text_chunks
        return text_chunks

    def get_cached(self, text: str | Sequence[str]) -> TextChunks | None:
        """Return the cached chunk result for ``text`` if available."""

        return self._last_outputs.get(self._cache_key(text))


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
class QaChunkRecord:
    """Minimal chunk data required for QA evaluation."""

    uid: str
    checksum: str
    text: str


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


@dataclass
class QaSourceRecord:
    """Aggregated ingestion artifact metadata for a single source."""

    relative_path: str
    git_commit: str | None
    document_checksum: str
    chunks: list[QaChunkRecord]
    ingest_run_key: str | None = None


@dataclass
class QaThresholds:
    """Threshold configuration controlling QA gating."""

    max_missing_embeddings: int = 0
    max_orphan_chunks: int = 0
    max_checksum_mismatches: int = 0
    max_semantic_failures: int = 0
    max_semantic_orphans: int = 0


@dataclass
class QaResult:
    """Result payload produced by the ingestion QA evaluator."""

    status: str
    summary: str
    metrics: dict[str, Any]
    anomalies: list[str]
    thresholds: QaThresholds
    report_json: str
    report_markdown: str
    timestamp: datetime
    version: str
    duration_ms: int

    @property
    def passed(self) -> bool:
        """
        Indicates whether the QA evaluation passed.
        
        @returns:
            `true` if `status` equals "pass", `false` otherwise.
        """
        return self.status == "pass"


class IngestionQaEvaluator:
    """Compute ingestion quality metrics and enforce gating thresholds."""

    TOKEN_BINS = (64, 128, 256, 512, 1024, 2048)

    def __init__(
        self,
        *,
        driver,
        database: str | None,
        sources: Sequence[QaSourceRecord],
        thresholds: QaThresholds,
        report_root: Path,
        report_version: str,
        semantic_summary: SemanticQaSummary | None = None,
    ) -> None:
        """
        Initialize the evaluator with the Neo4j driver, QA sources, thresholds, and report output settings.

        Parameters:
            database (str | None): Optional Neo4j database name to query for QA metrics; use the default database when None.
            sources (Sequence[QaSourceRecord]): Sequence of QA source records describing ingested documents and their chunks to evaluate.
            thresholds (QaThresholds): QA gating thresholds to apply when evaluating anomalies.
            report_root (Path): Filesystem directory where JSON and Markdown QA reports will be written.
            report_version (str): Version string to embed in generated QA reports.
            semantic_summary (SemanticQaSummary | None): Optional semantic enrichment metrics collected during ingestion.

        """
        self._driver = driver
        self._database = database
        self._sources = list(sources)
        self._thresholds = thresholds
        self._report_root = report_root
        self._report_version = report_version
        self._semantic_summary = semantic_summary

    def evaluate(self) -> QaResult:
        """
        Perform the ingestion QA evaluation across the provided sources and produce a QA result and report files.
        
        Queries the graph for chunk/document counts and specific anomaly counts (missing embeddings, orphan chunks, checksum mismatches), computes aggregate metrics per-source and totals, compares results against the configured thresholds, writes a sanitized JSON and Markdown QA report to the configured report directory, and records evaluation duration.
        
        Returns:
            QaResult: Aggregated QA evaluation result containing status ("pass" or "fail"), human-readable summary, metrics dictionary, list of detected anomalies, used thresholds, paths to the generated JSON and Markdown reports (relative to the repository when possible), timestamp, report version, and evaluation duration in milliseconds.
        """
        eval_start = time.perf_counter()
        timestamp = datetime.now(timezone.utc)
        metrics: dict[str, Any] = {}
        anomalies: list[str] = []

        chunk_uids = [chunk.uid for record in self._sources for chunk in record.chunks]

        counts = _collect_counts(self._driver, database=self._database)
        metrics["graph_counts"] = counts

        if chunk_uids:
            missing_embeddings = self._query_value(
                """
                UNWIND $uids AS uid
                MATCH (c:Chunk {uid: uid})
                WHERE c.embedding IS NULL OR size(c.embedding) = 0
                RETURN count(*) AS value
                """,
                {"uids": chunk_uids},
            )
            orphan_chunks = self._query_value(
                """
                UNWIND $uids AS uid
                MATCH (c:Chunk {uid: uid})
                WHERE NOT ( (:Document)-[:HAS_CHUNK]->(c) )
                RETURN count(*) AS value
                """,
                {"uids": chunk_uids},
            )
        else:
            missing_embeddings = 0
            orphan_chunks = 0

        chunk_rows = [
            {"uid": chunk.uid, "checksum": chunk.checksum}
            for record in self._sources
            for chunk in record.chunks
        ]
        if chunk_rows:
            checksum_mismatches = self._query_value(
                """
                UNWIND $chunks AS row
                MATCH (c:Chunk {uid: row.uid})
                WHERE coalesce(c.checksum, "") <> row.checksum
                RETURN count(*) AS value
                """,
                {"chunks": chunk_rows},
            )
        else:
            checksum_mismatches = 0

        metrics["missing_embeddings"] = missing_embeddings
        metrics["orphan_chunks"] = orphan_chunks
        metrics["checksum_mismatches"] = checksum_mismatches

        totals = self._compute_totals()
        metrics.update(totals)

        if self._semantic_summary and self._semantic_summary.enabled:
            semantic_metrics: dict[str, Any] = {
                "chunks_processed": self._semantic_summary.chunks_processed,
                "chunk_failures": self._semantic_summary.chunk_failures,
                "nodes_written": self._semantic_summary.nodes_written,
                "relationships_written": self._semantic_summary.relationships_written,
            }
            semantic_metrics.update(self._collect_semantic_counts())
            metrics["semantic"] = semantic_metrics
            if (
                semantic_metrics.get("chunk_failures", 0)
                > self._thresholds.max_semantic_failures
            ):
                anomalies.append(
                    "semantic_chunk_failures="
                    f"{semantic_metrics.get('chunk_failures', 0)} exceeds max"
                    f" {self._thresholds.max_semantic_failures}"
                )
            if (
                semantic_metrics.get("orphan_entities", 0)
                > self._thresholds.max_semantic_orphans
            ):
                anomalies.append(
                    "semantic_orphan_entities="
                    f"{semantic_metrics.get('orphan_entities', 0)} exceeds max"
                    f" {self._thresholds.max_semantic_orphans}"
                )

        per_file = [
            {
                "relative_path": record.relative_path,
                "git_commit": record.git_commit,
                "document_checksum": record.document_checksum,
                "chunks": len(record.chunks),
            }
            for record in self._sources
        ]
        metrics["files"] = per_file

        if missing_embeddings > self._thresholds.max_missing_embeddings:
            anomalies.append(
                f"missing_embeddings={missing_embeddings} exceeds max {self._thresholds.max_missing_embeddings}"
            )
        if orphan_chunks > self._thresholds.max_orphan_chunks:
            anomalies.append(
                f"orphan_chunks={orphan_chunks} exceeds max {self._thresholds.max_orphan_chunks}"
            )
        if checksum_mismatches > self._thresholds.max_checksum_mismatches:
            anomalies.append(
                f"checksum_mismatches={checksum_mismatches} exceeds max {self._thresholds.max_checksum_mismatches}"
            )

        status = "pass" if not anomalies else "fail"
        summary = (
            f"QA {status.upper()}: missing_embeddings={missing_embeddings}, "
            f"orphan_chunks={orphan_chunks}, checksum_mismatches={checksum_mismatches}"
        )

        report_dir = self._report_root / timestamp.strftime("%Y%m%dT%H%M%S")
        json_path = report_dir / "quality_report.json"
        md_path = report_dir / "quality_report.md"
        _ensure_directory(json_path)

        payload = {
            "version": self._report_version,
            "generated_at": timestamp.isoformat(),
            "status": status,
            "summary": summary,
            "metrics": metrics,
            "thresholds": asdict(self._thresholds),
            "anomalies": anomalies,
        }
        eval_duration_ms = int((time.perf_counter() - eval_start) * 1000)
        metrics["qa_evaluation_ms"] = eval_duration_ms

        sanitized_payload = scrub_object(payload)
        json_path.write_text(json.dumps(sanitized_payload, indent=2), encoding="utf-8")
        md_path.write_text(self._render_markdown(sanitized_payload), encoding="utf-8")

        report_json_rel = _relative_to_repo(json_path)
        report_md_rel = _relative_to_repo(md_path)

        return QaResult(
            status=status,
            summary=summary,
            metrics=metrics,
            anomalies=anomalies,
            thresholds=self._thresholds,
            report_json=report_json_rel,
            report_markdown=report_md_rel,
            timestamp=timestamp,
            version=self._report_version,
            duration_ms=eval_duration_ms,
        )

    def _collect_semantic_counts(self) -> dict[str, int]:
        """Collect semantic enrichment counts from Neo4j when enrichment is enabled."""

        if not self._semantic_summary or not self._semantic_summary.enabled:
            return {}
        tag = self._semantic_summary.source_tag
        node_count = self._query_value(
            "MATCH (n) WHERE n.semantic_source = $source RETURN count(*) AS value",
            {"source": tag},
        )
        relationship_count = self._query_value(
            (
                "MATCH ()-[r]->() WHERE r.semantic_source = $source "
                "RETURN count(*) AS value"
            ),
            {"source": tag},
        )
        orphan_count = self._query_value(
            (
                "MATCH (n) WHERE n.semantic_source = $source AND NOT (n)--() "
                "RETURN count(*) AS value"
            ),
            {"source": tag},
        )
        return {
            "nodes_in_db": node_count,
            "relationships_in_db": relationship_count,
            "orphan_entities": orphan_count,
        }

    def _compute_totals(self) -> dict[str, Any]:
        """
        Compute aggregate counts and summary statistics for all provided QA sources.
        
        Returns:
            totals (dict[str, Any]): Aggregate metrics including:
                - files_processed (int): number of source records processed.
                - chunks_processed (int): total number of chunks across all sources.
                - token_estimate (dict): token-based statistics with keys:
                    - total (int): sum of estimated tokens for all chunks.
                    - max (int): largest token estimate for a single chunk.
                    - mean (float): mean token estimate across chunks.
                    - histogram (list[tuple[int, int]]): token histogram buckets as produced by `_token_histogram`.
                - char_lengths (dict): character-length statistics with keys:
                    - total (int): sum of character lengths for all chunks.
                    - max (int): largest character length for a single chunk.
                    - mean (float): mean character length across chunks.
        """
        chunk_texts = [chunk.text for record in self._sources for chunk in record.chunks]
        chunk_lengths = [len(text) for text in chunk_texts]
        token_estimates = [self._estimate_tokens(text) for text in chunk_texts]

        histogram = self._token_histogram(token_estimates)

        return {
            "files_processed": len(self._sources),
            "chunks_processed": len(chunk_texts),
            "token_estimate": {
                "total": sum(token_estimates),
                "max": max(token_estimates) if token_estimates else 0,
                "mean": statistics.fmean(token_estimates) if token_estimates else 0.0,
                "histogram": histogram,
            },
            "char_lengths": {
                "total": sum(chunk_lengths),
                "max": max(chunk_lengths) if chunk_lengths else 0,
                "mean": statistics.fmean(chunk_lengths) if chunk_lengths else 0.0,
            },
        }

    def _token_histogram(self, token_counts: Sequence[int]) -> dict[str, int]:
        """
        Create a histogram of token counts grouped into the evaluator's predefined bins.
        
        Parameters:
            token_counts (Sequence[int]): Sequence of token counts to bin.
        
        Returns:
            dict[str, int]: Mapping from bucket label to the number of counts in that bucket.
                Bucket labels represent inclusive upper-bound ranges for each configured bin
                and a final open-ended bucket for counts greater than the highest bin.
        """
        buckets = {self._bucket_label(None, upper): 0 for upper in self.TOKEN_BINS}
        buckets[self._bucket_label(self.TOKEN_BINS[-1], None)] = 0
        for count in token_counts:
            placed = False
            for upper in self.TOKEN_BINS:
                if count <= upper:
                    buckets[self._bucket_label(None, upper)] += 1
                    placed = True
                    break
            if not placed:
                buckets[self._bucket_label(self.TOKEN_BINS[-1], None)] += 1
        return buckets

    @staticmethod
    def _bucket_label(lower: int | None, upper: int | None) -> str:
        """
        Format a human-readable label for an integer bucket defined by optional lower and upper bounds.
        
        Parameters:
            lower (int | None): Lower bound of the bucket (None means unbounded below).
            upper (int | None): Upper bound of the bucket (None means unbounded above).
        
        Returns:
            str: A label describing the bucket:
                - "`<= {upper}`" if only `upper` is provided,
                - "`> {lower}`" if only `lower` is provided,
                - `"unknown"` if both bounds are `None`,
                - "`{lower+1}-{upper}`" when both bounds are provided.
        """
        if lower is None and upper is not None:
            return f"<= {upper}"
        if lower is not None and upper is None:
            return f"> {lower}"
        if lower is None and upper is None:
            return "unknown"
        return f"{lower + 1}-{upper}"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate token count for a text using a simple character-to-token heuristic.
        
        Uses an approximate ratio of 4 characters per token and ensures non-empty text maps to at least 1 token.
        
        Returns:
            Estimated number of tokens as an integer; `0` for empty input.
        """
        if not text:
            return 0
        # Rough heuristic assuming ~4 characters per token for mixed-language corpora.
        return max(1, math.ceil(len(text) / 4))

    def _query_value(self, query: str, parameters: Mapping[str, Any]) -> int:
        """
        Execute the given query against the evaluator's driver and return the first numeric result as an integer.
        
        Parameters:
            query (str): Query string to execute.
            parameters (Mapping[str, Any]): Parameters to pass to the query.
        
        Returns:
            int: The integer value of the first result row's `"value"` field, or the first column if unnamed; returns 0 when there are no configured sources, no rows, or when the retrieved value is missing or not coercible to an integer.
        """
        if not self._sources:
            return 0
        result = self._driver.execute_query(query, parameters, database_=self._database)
        records = result[0] if isinstance(result, tuple) else result
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
            except (IndexError, TypeError):
                value = None
        return int(value or 0)

    def _render_markdown(self, payload: Mapping[str, Any]) -> str:
        """
        Render an ingestion QA payload as a Markdown-formatted report.
        
        Parameters:
            payload (Mapping[str, Any]): QA report payload containing keys such as
                `version`, `generated_at`, `status`, `summary`, `metrics`, `anomalies`,
                and `thresholds`. `metrics` may include `token_estimate`, `files_processed`,
                `chunks_processed`, `missing_embeddings`, `orphan_chunks`, `checksum_mismatches`,
                and a `files` sequence with per-file entries (`relative_path`, `chunks`,
                `document_checksum`, `git_commit`).
        
        Returns:
            str: A Markdown string summarizing the QA report, including top-line status,
            metrics, an optional token histogram, findings (anomalies), thresholds, and a
            per-file table when available.
        """
        lines: list[str] = []
        lines.append(f"# Ingestion QA Report ({payload['version']})")
        lines.append("")
        lines.append(f"- Generated: {payload['generated_at']}")
        lines.append(f"- Status: {payload['status'].upper()}")
        lines.append(f"- Summary: {payload['summary']}")
        lines.append("")
        metrics_obj = payload.get("metrics", {})
        metrics = metrics_obj if isinstance(metrics_obj, Mapping) else {}
        lines.append("## Metrics")
        lines.append("")
        token_stats_obj = metrics.get("token_estimate", {})
        token_stats = token_stats_obj if isinstance(token_stats_obj, Mapping) else {}
        lines.append(
            f"- Files processed: {metrics.get('files_processed', 0)}"
        )
        lines.append(
            f"- Chunks processed: {metrics.get('chunks_processed', 0)}"
        )
        lines.append(
            f"- Token estimate total: {token_stats.get('total', 0)}"
        )
        lines.append(
            f"- Token estimate mean: {round(token_stats.get('mean', 0.0), 2)}"
        )
        lines.append(
            f"- Missing embeddings: {metrics.get('missing_embeddings', 0)}"
        )
        lines.append(
            f"- Orphan chunks: {metrics.get('orphan_chunks', 0)}"
        )
        lines.append(
            f"- Checksum mismatches: {metrics.get('checksum_mismatches', 0)}"
        )
        lines.append("")
        histogram = token_stats.get("histogram", {})
        if isinstance(histogram, Mapping) and histogram:
            lines.append("### Token Histogram")
            lines.append("")
            lines.append("| Bucket | Count |")
            lines.append("| --- | ---: |")
            for bucket, count in histogram.items():
                lines.append(f"| {bucket} | {count} |")
            lines.append("")
        anomalies = payload.get("anomalies", [])
        lines.append("## Findings")
        lines.append("")
        if anomalies:
            for item in anomalies:
                lines.append(f"- ❌ {item}")
        else:
            lines.append("- ✅ All thresholds satisfied")
        lines.append("")
        thresholds = payload.get("thresholds", {})
        if isinstance(thresholds, Mapping) and thresholds:
            lines.append("## Thresholds")
            lines.append("")
            for key, value in thresholds.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
        files = metrics.get("files", [])
        if isinstance(files, Sequence) and not isinstance(files, (str, bytes)) and files:
            lines.append("## Files")
            lines.append("")
            lines.append("| Relative Path | Chunks | Checksum | Git Commit |")
            lines.append("| --- | ---: | --- | --- |")
            for entry in files:
                if isinstance(entry, Mapping):
                    lines.append(
                        f"| {entry.get('relative_path', '***')} | {entry.get('chunks', '***')} | {entry.get('document_checksum', '***')} | {entry.get('git_commit') or '-'} |"
                    )
            lines.append("")
        return "\n".join(lines)
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the KG build CLI.
    
    Parameters:
        argv (Sequence[str] | None): Optional list of argument strings to parse; when None the process's command-line (sys.argv) is used.
    
    Returns:
        argparse.Namespace: Parsed options including source selection (--source, --source-dir, --include-pattern), chunking/profile settings (--profile, --chunk-size, --chunk-overlap), Neo4j target (--database), logging path (--log-path), QA configuration (--qa-report-dir, --qa-max-missing-embeddings, --qa-max-orphan-chunks, --qa-max-checksum-mismatches), and control flags such as --reset-database.
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
        "--qa-report-dir",
        default=str(DEFAULT_QA_DIR),
        help="Directory for ingestion QA reports (default: %(default)s)",
    )
    parser.add_argument(
        "--qa-max-missing-embeddings",
        type=int,
        default=0,
        help="Maximum allowed chunks missing embeddings before failing QA (default: %(default)s)",
    )
    parser.add_argument(
        "--qa-max-orphan-chunks",
        type=int,
        default=0,
        help="Maximum allowed orphan chunks (default: %(default)s)",
    )
    parser.add_argument(
        "--qa-max-checksum-mismatches",
        type=int,
        default=0,
        help="Maximum allowed checksum mismatches (default: %(default)s)",
    )
    parser.add_argument(
        "--qa-max-semantic-failures",
        type=int,
        default=0,
        help="Maximum allowed semantic extraction failures (default: %(default)s)",
    )
    parser.add_argument(
        "--qa-max-semantic-orphans",
        type=int,
        default=0,
        help="Maximum allowed semantic orphan entities (default: %(default)s)",
    )
    parser.add_argument(
        "--enable-semantic",
        dest="semantic_enabled",
        action="store_true",
        help="Enable semantic enrichment using the GraphRAG entity-relation extractor.",
    )
    parser.add_argument(
        "--semantic-max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent semantic extraction requests (default: %(default)s)",
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


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Builds a knowledge graph from the provided source file(s) and produces a structured run log.
    
    Reads CLI arguments (or uses the supplied argv), validates environment and chunking settings, ingests source text into Neo4j using configured OpenAI clients and the pipeline, runs ingestion QA, writes a sanitized JSON run log to disk, and returns the assembled log dictionary.
    
    Parameters:
        argv (Sequence[str] | None): Optional list of CLI arguments to override sys.argv. When None the process parses arguments from the environment/command line.
    
    Returns:
        dict[str, Any]: A structured run log containing metadata about the run, including:
            - timestamp: ISO 8601 UTC timestamp of completion
            - operation: the operation name ("kg_build")
            - status: operation status ("success" on normal completion)
            - duration_ms: elapsed time in milliseconds
            - source: path to the input source or directory
            - source_mode: "file" or "directory"
            - input_bytes: total size of input in bytes
            - chunking: chunking parameters used (size, overlap, profile, include_patterns)
            - database: Neo4j database name (or None)
            - reset_database: boolean indicating whether the destructive reset flag was provided
            - openai: OpenAI configuration used (chat_model, embedding_model, embedding_dimensions, max_attempts)
            - counts: mapping of graph entity counts (documents, chunks, relationships)
            - run_id / run_ids: pipeline run identifier(s)
            - files: per-file metadata (path, checksum, chunks)
            - chunks: per-chunk metadata (source, index, id, checksum, git_commit)
            - qa: QA evaluation summary and report locations (when present)
    
    Raises:
        RuntimeError: If OpenAI requests, Neo4j operations, or ingestion QA gating fail.
    """
    args = _parse_args(argv)

    profile = args.profile or DEFAULT_PROFILE
    preset = PROFILE_PRESETS.get(profile, PROFILE_PRESETS[DEFAULT_PROFILE])
    chunk_size = args.chunk_size if args.chunk_size is not None else preset["chunk_size"]
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else preset["chunk_overlap"]
    include_patterns = tuple(args.include_patterns) if args.include_patterns else tuple(preset.get("include", ()))

    chunk_size = _ensure_positive(chunk_size, name="chunk_size")
    chunk_overlap = _ensure_non_negative(chunk_overlap, name="chunk_overlap")
    semantic_max_concurrency = args.semantic_max_concurrency
    if args.semantic_enabled:
        semantic_max_concurrency = _ensure_positive(
            semantic_max_concurrency, name="semantic_max_concurrency"
        )

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
    qa_sources: list[QaSourceRecord] = []
    semantic_totals = SemanticEnrichmentStats()
    reset_pending = bool(args.reset_database)

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
                        database=args.database,
                        embedder=embedder,
                        llm=llm,
                        splitter=splitter,
                        reset_database=reset_pending,
                        ingest_run_key=ingest_run_key,
                    )
                except (OpenAIClientError, LLMGenerationError, EmbeddingsGenerationError) as exc:  # noqa: TRY003
                    raise RuntimeError(f"OpenAI request failed: {exc}") from exc
                except (Neo4jError, ClientError) as exc:  # noqa: TRY003
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

                qa_sources.append(
                    QaSourceRecord(
                        relative_path=spec.relative_path,
                        git_commit=git_commit,
                        document_checksum=spec.checksum,
                        chunks=[
                            QaChunkRecord(
                                uid=meta.uid,
                                checksum=meta.checksum,
                                text=getattr(chunk_result.chunks[i], "text", ""),
                            )
                            for i, meta in enumerate(chunk_metadata)
                        ],
                        ingest_run_key=ingest_run_key,
                    )
                )

                if args.semantic_enabled:
                    semantic_stats = _run_semantic_enrichment(
                        driver=driver,
                        database=args.database,
                        llm=llm,
                        chunk_result=chunk_result,
                        chunk_metadata=chunk_metadata,
                        relative_path=spec.relative_path,
                        git_commit=git_commit,
                        document_checksum=spec.checksum,
                        ingest_run_key=ingest_run_key,
                        max_concurrency=semantic_max_concurrency,
                    )
                    semantic_totals.chunks_processed += (
                        semantic_stats.chunks_processed
                    )
                    semantic_totals.chunk_failures += semantic_stats.chunk_failures
                    semantic_totals.nodes_written += semantic_stats.nodes_written
                    semantic_totals.relationships_written += (
                        semantic_stats.relationships_written
                    )

        thresholds = QaThresholds(
            max_missing_embeddings=args.qa_max_missing_embeddings,
            max_orphan_chunks=args.qa_max_orphan_chunks,
            max_checksum_mismatches=args.qa_max_checksum_mismatches,
            max_semantic_failures=args.qa_max_semantic_failures,
            max_semantic_orphans=args.qa_max_semantic_orphans,
        )
        semantic_summary = SemanticQaSummary(
            enabled=bool(args.semantic_enabled),
            chunks_processed=semantic_totals.chunks_processed,
            chunk_failures=semantic_totals.chunk_failures,
            nodes_written=semantic_totals.nodes_written,
            relationships_written=semantic_totals.relationships_written,
        )
        evaluator = IngestionQaEvaluator(
            driver=driver,
            database=args.database,
            sources=qa_sources,
            thresholds=thresholds,
            report_root=Path(args.qa_report_dir),
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
            _rollback_ingest(driver, database=args.database, sources=qa_sources)
            raise RuntimeError(
                "Ingestion QA gating failed; see ingestion QA report for details"
            )

        counts = qa_result.metrics.get("graph_counts", {})
        if not counts:
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
    if semantic_summary.enabled:
        log["semantic"] = {
            "chunks_processed": semantic_summary.chunks_processed,
            "chunk_failures": semantic_summary.chunk_failures,
            "nodes_written": semantic_summary.nodes_written,
            "relationships_written": semantic_summary.relationships_written,
        }
    if qa_section is not None:
        log["qa"] = qa_section

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
