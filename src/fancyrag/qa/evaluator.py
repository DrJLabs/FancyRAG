"""Ingestion QA evaluation helpers extracted from the KG pipeline."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from _compat.structlog import get_logger
from cli.sanitizer import scrub_object
from fancyrag.db.neo4j_queries import (
    collect_counts,
    collect_semantic_counts,
    count_checksum_mismatches,
    count_missing_embeddings,
    count_orphan_chunks,
    Neo4jError,
)
from fancyrag.qa.report import render_markdown, write_ingestion_report

logger = get_logger(__name__)


class SemanticSummary(Protocol):
    """Protocol describing the semantic QA summary required by the evaluator."""

    enabled: bool
    chunks_processed: int
    chunk_failures: int
    nodes_written: int
    relationships_written: int
    source_tag: str


@dataclass
class QaChunkRecord:
    """Minimal chunk data required for QA evaluation."""

    uid: str
    checksum: str
    text: str


@dataclass
class QaSourceRecord:
    """Aggregated ingestion artifact metadata for a single source."""

    path: str
    relative_path: str
    document_checksum: str
    git_commit: str | None
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
        """Return True when the QA evaluation passed."""

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
        semantic_summary: SemanticSummary | None = None,
    ) -> None:
        self._driver = driver
        self._database = database
        self._sources = list(sources)
        self._thresholds = thresholds
        self._report_root = report_root
        self._report_version = report_version
        self._semantic_summary = semantic_summary

    def evaluate(self) -> QaResult:
        """Evaluate ingestion QA metrics and emit reports."""

        eval_start = time.perf_counter()
        timestamp = datetime.now(timezone.utc)
        metrics: dict[str, Any] = {}
        anomalies: list[str] = []

        chunk_uids = [chunk.uid for record in self._sources for chunk in record.chunks]

        try:
            counts = collect_counts(self._driver, database=self._database)
        except Neo4jError as exc:
            logger.warning("qa.metrics.counts_failed", error=str(exc))
            counts = {"documents": 0, "chunks": 0, "relationships": 0}
        metrics["graph_counts"] = counts

        if chunk_uids:
            try:
                missing_embeddings = count_missing_embeddings(
                    self._driver,
                    database=self._database,
                    chunk_uids=chunk_uids,
                )
            except Neo4jError as exc:
                logger.warning("qa.metrics.missing_embeddings_failed", error=str(exc))
                missing_embeddings = 0
            try:
                orphan_chunks = count_orphan_chunks(
                    self._driver,
                    database=self._database,
                    chunk_uids=chunk_uids,
                )
            except Neo4jError as exc:
                logger.warning("qa.metrics.orphan_chunks_failed", error=str(exc))
                orphan_chunks = 0
        else:
            missing_embeddings = 0
            orphan_chunks = 0

        chunk_rows = [
            {"uid": chunk.uid, "checksum": chunk.checksum}
            for record in self._sources
            for chunk in record.chunks
        ]
        if chunk_rows:
            try:
                checksum_mismatches = count_checksum_mismatches(
                    self._driver,
                    database=self._database,
                    chunk_rows=chunk_rows,
                )
            except Neo4jError as exc:
                logger.warning("qa.metrics.checksum_mismatches_failed", error=str(exc))
                checksum_mismatches = 0
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
        report_json_rel, report_md_rel = write_ingestion_report(
            sanitized_payload=sanitized_payload,
            report_root=self._report_root,
            timestamp=timestamp,
        )

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

    def _render_markdown(self, payload: Mapping[str, Any]) -> str:
        """Legacy helper retained for existing evaluator tests and tooling."""

        return render_markdown(payload)

    def _collect_semantic_counts(self) -> dict[str, int]:
        """Collect semantic enrichment counts from Neo4j when enrichment is enabled."""

        if not self._semantic_summary or not self._semantic_summary.enabled:
            return {}
        tag = self._semantic_summary.source_tag
        try:
            return dict(
                collect_semantic_counts(
                    self._driver, database=self._database, source_tag=tag
                )
            )
        except Neo4jError as exc:
            logger.warning(
                "qa.metrics.semantic_counts_failed",
                error=str(exc),
                source_tag=tag,
            )
            return {
                "nodes_in_db": 0,
                "relationships_in_db": 0,
                "orphan_entities": 0,
            }

    def _compute_totals(self) -> dict[str, Any]:
        """Compute aggregate counts and summary statistics for all provided QA sources."""

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
        """Create a histogram of token counts grouped into predefined bins."""

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
        """Format a human-readable label for a histogram bucket."""

        if lower is None and upper is not None:
            return f"<= {upper}"
        if lower is not None and upper is None:
            return f"> {lower}"
        if lower is None and upper is None:
            return "unknown"
        return f"{lower + 1}-{upper}"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count using a simple character ratio heuristic."""

        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))
