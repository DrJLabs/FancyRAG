"""QA utilities for FancyRAG."""

from fancyrag.db.neo4j_queries import collect_counts
from .evaluator import (
    IngestionQaEvaluator,
    QaChunkRecord,
    QaResult,
    QaSourceRecord,
    QaThresholds,
)
from .report import (
    REPORT_JSON_FILENAME,
    REPORT_MARKDOWN_FILENAME,
    render_markdown,
    write_ingestion_report,
)

__all__ = [
    "IngestionQaEvaluator",
    "QaChunkRecord",
    "QaResult",
    "QaSourceRecord",
    "QaThresholds",
    "collect_counts",
    "REPORT_JSON_FILENAME",
    "REPORT_MARKDOWN_FILENAME",
    "render_markdown",
    "write_ingestion_report",
]
