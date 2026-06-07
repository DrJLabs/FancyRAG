from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fancyrag.qa import QaChunkRecord, QaSourceRecord, QaThresholds


@dataclass(frozen=True)
class ResolvedSettings:
    """Resolved pipeline settings derived from presets and explicit overrides."""

    profile: str
    chunk_size: int
    chunk_overlap: int
    include_patterns: tuple[str, ...]
    semantic_max_concurrency: int


@dataclass(frozen=True)
class SourceDiscoveryResult:
    """Outcome of resolving pipeline sources."""

    sources: Sequence[Any]
    source_mode: str
    source_root: Path


@dataclass(frozen=True)
class ClientBundle:
    """Collection of shared OpenAI client adapters plus configured splitter."""

    shared_client: Any
    embedder: Any
    llm: Any
    semantic_llm: Any
    splitter: Any


@dataclass
class IngestionArtifacts:
    """Artifacts produced while ingesting a single source."""

    run_id: str | None
    qa_source: QaSourceRecord
    log_entry: dict[str, Any]
    chunk_entries: list[dict[str, Any]]
    semantic_stats: Any


@dataclass
class QaOutcome:
    """Aggregated QA outcome and supporting metrics."""

    qa_section: dict[str, Any] | None
    counts: Mapping[str, int] | None
    semantic_summary: Any


def resolve_settings(
    *,
    profile: str | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
    include_patterns_override: Sequence[str] | None,
    semantic_enabled: bool,
    semantic_max_concurrency: int,
    profile_presets: Mapping[str, Mapping[str, Any]],
    default_profile: str,
    ensure_positive: Callable[[int, str], int],
    ensure_non_negative: Callable[[int, str], int],
) -> ResolvedSettings:
    """
    Resolve effective chunking and semantic concurrency settings for the pipeline.

    The helper consolidates profile presets with explicit overrides, validating numeric
    parameters before returning a ResolvedSettings record that downstream helpers can
    consume without consulting global state.
    """

    resolved_profile = profile or default_profile
    preset = profile_presets.get(resolved_profile, profile_presets[default_profile])

    resolved_chunk_size = (
        chunk_size if chunk_size is not None else int(preset.get("chunk_size", 0))
    )
    resolved_chunk_overlap = (
        chunk_overlap if chunk_overlap is not None else int(preset.get("chunk_overlap", 0))
    )
    if include_patterns_override is not None:
        resolved_patterns = tuple(include_patterns_override)
    else:
        include_patterns = preset.get("include", ())
        resolved_patterns = tuple(include_patterns) if include_patterns else ()

    resolved_chunk_size = ensure_positive(resolved_chunk_size, name="chunk_size")
    resolved_chunk_overlap = ensure_non_negative(resolved_chunk_overlap, name="chunk_overlap")

    resolved_semantic_max = semantic_max_concurrency
    if semantic_enabled:
        resolved_semantic_max = ensure_positive(
            resolved_semantic_max,
            name="semantic_max_concurrency",
        )

    return ResolvedSettings(
        profile=resolved_profile,
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
        include_patterns=resolved_patterns,
        semantic_max_concurrency=resolved_semantic_max,
    )


def discover_sources(
    *,
    source: Path,
    source_dir: Path | None,
    include_patterns: tuple[str, ...],
    relative_to_repo: Callable[[Path, Path | None], str],
    read_source: Callable[[Path], str],
    read_directory_source: Callable[[Path], str | None],
    discover_source_files: Callable[[Path, Sequence[str]], Sequence[Path]],
    compute_checksum: Callable[[str], str],
    source_spec_factory: Callable[..., Any],
) -> SourceDiscoveryResult:
    """
    Resolve ingestion sources from either a single file or a directory tree.

    Returns a SourceDiscoveryResult containing the discovered SourceSpec collection,
    the mode ("file" or "directory"), and the root path associated with the discovery.
    """

    if source_dir:
        directory = Path(source_dir).expanduser()
        if not directory.is_dir():
            raise ValueError(f"source directory not found: {directory}")
        patterns = include_patterns or ()
        files = discover_source_files(directory, patterns)
        specs: list[Any] = []
        for file_path in files:
            content = read_directory_source(file_path)
            if content is None:
                continue
            specs.append(
                source_spec_factory(
                    path=file_path,
                    relative_path=relative_to_repo(file_path, base=directory),
                    text=content,
                    checksum=compute_checksum(content),
                )
            )
        if not specs:
            raise ValueError(
                "No ingestible files matched the supplied directory and include patterns."
            )
        return SourceDiscoveryResult(
            sources=tuple(specs),
            source_mode="directory",
            source_root=directory,
        )

    source_path = Path(source).expanduser()
    source_text = read_source(source_path)
    spec = source_spec_factory(
        path=source_path,
        relative_path=relative_to_repo(source_path),
        text=source_text,
        checksum=compute_checksum(source_text),
    )
    return SourceDiscoveryResult(
        sources=(spec,),
        source_mode="file",
        source_root=source_path,
    )


def build_clients(
    *,
    settings: Any,
    chunk_size: int,
    chunk_overlap: int,
    shared_client_factory: Callable[[Any], Any],
    embedder_factory: Callable[[Any, Any], Any],
    llm_factory: Callable[[Any, Any], Any],
    semantic_llm_factory: Callable[..., Any],
    schema_factory: Callable[[], dict[str, Any]],
    splitter_config_factory: Callable[..., Any],
    splitter_factory: Callable[[Any], Any],
) -> ClientBundle:
    """
    Construct shared OpenAI client adapters and the configured caching splitter.

    Returns a ClientBundle containing the shared client, embedder, LLM adapter, and
    splitter so downstream helpers can orchestrate ingestion without recalculating
    configuration.
    """

    shared_client = shared_client_factory(settings)
    embedder = embedder_factory(shared_client, settings)
    llm = llm_factory(shared_client, settings)
    semantic_schema = schema_factory()
    semantic_llm = semantic_llm_factory(shared_client, settings, schema=semantic_schema)
    splitter_config = splitter_config_factory(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitter = splitter_factory(splitter_config)
    return ClientBundle(
        shared_client=shared_client,
        embedder=embedder,
        llm=llm,
        semantic_llm=semantic_llm,
        splitter=splitter,
    )


def ingest_source(
    *,
    spec: Any,
    options: Any,
    uri: str,
    auth: tuple[str, str],
    driver: Any,
    clients: ClientBundle,
    semantic_settings: Any,
    git_commit: str | None,
    reset_database: bool,
    execute_pipeline: Callable[..., str | None],
    build_chunk_metadata: Callable[[Sequence[Any], str, str | None], Sequence[Any]],
    ensure_document_relationships: Callable[..., None],
    semantic_enabled: bool,
    semantic_max_concurrency: int,
    run_semantic_enrichment: Callable[..., Any],
    semantic_stats_factory: Callable[[], Any],
    ingest_run_key_factory: Callable[[], str],
) -> IngestionArtifacts:
    """
    Process a single source specification: execute the Neo4j ingestion pipeline,
    capture chunk metadata, assemble QA source records, and optionally perform
    semantic enrichment. Returns IngestionArtifacts for downstream aggregation.
    """

    splitter = clients.splitter
    scope_token = str(Path(spec.path).resolve())
    semantic_stats = semantic_stats_factory()

    with splitter.scoped(scope_token):
        ingest_run_key = ingest_run_key_factory()
        run_id = execute_pipeline(
            uri=uri,
            auth=auth,
            source_text=spec.text,
            database=options.database,
            embedder=clients.embedder,
            llm=clients.llm,
            splitter=splitter,
            reset_database=reset_database,
            ingest_run_key=ingest_run_key,
        )

        chunk_result = splitter.get_cached(spec.text)
        if chunk_result is None:
            chunk_result = splitter.run(spec.text)
        if asyncio.iscoroutine(chunk_result):
            chunk_result = asyncio.run(chunk_result)

        chunk_metadata = list(
            build_chunk_metadata(
                chunk_result.chunks,
                relative_path=spec.relative_path,
                git_commit=git_commit,
            )
        )

        ensure_document_relationships(
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
                uid=getattr(meta, "uid"),
                checksum=getattr(meta, "checksum"),
                text=getattr(chunk, "text", "") or "",
            )
            for meta, chunk in zip(chunk_metadata, chunk_result.chunks)
        ]
        qa_source = QaSourceRecord(
            path=str(spec.path),
            relative_path=spec.relative_path,
            document_checksum=spec.checksum,
            git_commit=git_commit,
            chunks=qa_chunks,
            ingest_run_key=ingest_run_key,
        )

        log_entry = {
            "path": str(spec.path),
            "relative_path": spec.relative_path,
            "checksum": spec.checksum,
            "chunks": len(chunk_metadata),
        }
        chunk_entries = [
            {
                "path": str(spec.path),
                "uid": getattr(meta, "uid"),
                "sequence": getattr(meta, "sequence"),
                "index": getattr(meta, "index"),
                "checksum": getattr(meta, "checksum"),
                "relative_path": getattr(meta, "relative_path"),
                "git_commit": getattr(meta, "git_commit"),
            }
            for meta in chunk_metadata
        ]

        if semantic_enabled:
            semantic_max_retries = getattr(semantic_settings, "semantic_max_retries", 0)
            semantic_failure_artifacts = bool(
                getattr(semantic_settings, "semantic_failure_artifacts", False)
            )
            stats = run_semantic_enrichment(
                driver=driver,
                database=options.database,
                llm=clients.semantic_llm,
                chunk_result=chunk_result,
                chunk_metadata=chunk_metadata,
                relative_path=spec.relative_path,
                git_commit=git_commit,
                document_checksum=spec.checksum,
                ingest_run_key=ingest_run_key,
                max_concurrency=semantic_max_concurrency,
                max_retries=semantic_max_retries,
                failure_artifacts_enabled=semantic_failure_artifacts,
                failure_artifacts_root=options.qa_report_dir,
            )
            semantic_stats.chunks_processed = getattr(stats, "chunks_processed", 0)
            semantic_stats.chunk_failures = getattr(stats, "chunk_failures", 0)
            semantic_stats.nodes_written = getattr(stats, "nodes_written", 0)
            semantic_stats.relationships_written = getattr(
                stats,
                "relationships_written",
                0,
            )

    return IngestionArtifacts(
        run_id=run_id,
        qa_source=qa_source,
        log_entry=log_entry,
        chunk_entries=chunk_entries,
        semantic_stats=semantic_stats,
    )


def perform_qa(
    *,
    driver: Any,
    database: str | None,
    qa_sources: Sequence[QaSourceRecord],
    semantic_enabled: bool,
    semantic_totals: Any,
    qa_limits: Any,
    qa_report_dir: Path,
    qa_report_version: str,
    qa_evaluator_factory: Callable[..., Any],
    collect_counts: Callable[..., Mapping[str, int]],
    rollback_ingest: Callable[..., None],
    semantic_summary_factory: Callable[..., Any],
) -> QaOutcome:
    """
    Execute ingestion QA and return the serialized QA section, graph counts, and
    semantic summary used downstream by run_pipeline.
    """

    thresholds = QaThresholds(
        max_missing_embeddings=qa_limits.max_missing_embeddings,
        max_orphan_chunks=qa_limits.max_orphan_chunks,
        max_checksum_mismatches=qa_limits.max_checksum_mismatches,
        max_semantic_failures=qa_limits.max_semantic_failures,
        max_semantic_orphans=qa_limits.max_semantic_orphans,
    )
    semantic_summary = semantic_summary_factory(
        enabled=bool(semantic_enabled),
        chunks_processed=getattr(semantic_totals, "chunks_processed", 0),
        chunk_failures=getattr(semantic_totals, "chunk_failures", 0),
        nodes_written=getattr(semantic_totals, "nodes_written", 0),
        relationships_written=getattr(semantic_totals, "relationships_written", 0),
    )

    evaluator = qa_evaluator_factory(
        driver=driver,
        database=database,
        sources=qa_sources,
        thresholds=thresholds,
        report_root=qa_report_dir,
        report_version=qa_report_version,
        semantic_summary=semantic_summary,
    )
    qa_result = evaluator.evaluate()
    qa_section = {
        "status": qa_result.status,
        "summary": qa_result.summary,
        "report_version": qa_result.version,
        "report_json": qa_result.report_json,
        "report_markdown": qa_result.report_markdown,
        "thresholds": asdict(qa_result.thresholds)
        if hasattr(qa_result.thresholds, "__dataclass_fields__")
        else qa_result.thresholds,
        "metrics": qa_result.metrics,
        "anomalies": qa_result.anomalies,
        "duration_ms": qa_result.duration_ms,
    }

    if not qa_result.passed:
        rollback_ingest(driver, database=database, sources=qa_sources)
        raise RuntimeError("Ingestion QA gating failed; see ingestion QA report for details")

    counts = qa_result.metrics.get("graph_counts", {})
    if not counts:
        counts = collect_counts(driver, database=database)

    return QaOutcome(
        qa_section=qa_section,
        counts=counts,
        semantic_summary=semantic_summary,
    )
