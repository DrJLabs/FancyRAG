"""Command-line entry point for the FancyRAG knowledge-graph pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from _compat.structlog import get_logger
from fancyrag.kg import (
    DEFAULT_LOG_PATH,
    DEFAULT_PROFILE,
    DEFAULT_QA_DIR,
    DEFAULT_SOURCE,
    PROFILE_PRESETS,
    PipelineOptions,
    QaLimits,
    run_pipeline,
)

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the FancyRAG SimpleKGPipeline against local content, persisting"
        " results to Neo4j with structured logging and QA gating.",
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
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    return parser.parse_args(argv)


def _build_options(args: argparse.Namespace) -> PipelineOptions:
    include_patterns = tuple(args.include_patterns) if args.include_patterns else None
    qa_limits = QaLimits(
        max_missing_embeddings=args.qa_max_missing_embeddings,
        max_orphan_chunks=args.qa_max_orphan_chunks,
        max_checksum_mismatches=args.qa_max_checksum_mismatches,
        max_semantic_failures=args.qa_max_semantic_failures,
        max_semantic_orphans=args.qa_max_semantic_orphans,
    )
    source_dir = Path(args.source_dir).expanduser() if args.source_dir else None
    return PipelineOptions(
        source=Path(args.source).expanduser(),
        source_dir=source_dir,
        include_patterns=include_patterns,
        profile=args.profile,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        database=args.database,
        log_path=Path(args.log_path).expanduser(),
        qa_report_dir=Path(args.qa_report_dir).expanduser(),
        qa_limits=qa_limits,
        semantic_enabled=bool(args.semantic_enabled),
        semantic_max_concurrency=args.semantic_max_concurrency,
        reset_database=bool(args.reset_database),
    )


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Parse CLI arguments and execute the FancyRAG pipeline orchestrator."""

    args = _parse_args(argv)
    options = _build_options(args)
    return run_pipeline(options)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the pipeline and convert exceptions into process-style exit codes."""

    try:
        run(argv)
        return 0
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        logger.exception("kg_build.error", error=str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - final guard
        print(f"error: {exc}", file=sys.stderr)
        logger.exception("kg_build.failed", error=str(exc))
        return 1


__all__ = ["run", "main"]
