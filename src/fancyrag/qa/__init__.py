"""QA utilities for FancyRAG."""

from .evaluator import (
    IngestionQaEvaluator,
    QaChunkRecord,
    QaResult,
    QaSourceRecord,
    QaThresholds,
    collect_counts,
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
