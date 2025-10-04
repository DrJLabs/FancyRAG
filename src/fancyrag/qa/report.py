"""Report rendering helpers for ingestion QA outputs."""

from __future__ import annotations

import json
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from fancyrag.utils.paths import ensure_directory, relative_to_repo

REPORT_JSON_FILENAME = "quality_report.json"
REPORT_MARKDOWN_FILENAME = "quality_report.md"


def write_ingestion_report(
    *,
    sanitized_payload: Mapping[str, Any],
    report_root: Path,
    timestamp: datetime,
) -> tuple[str, str]:
    """Persist an ingestion QA report payload to JSON and Markdown files.

    Parameters
    ----------
    sanitized_payload:
        Payload that has already been scrubbed of sensitive values.
    report_root:
        Root directory where the timestamped report folder should be created.
    timestamp:
        Timestamp used to version the report directory.

    Returns
    -------
    tuple[str, str]
        Relative paths (from the repository root when available) to the
        generated JSON and Markdown artifacts.
    """

    stamp = timestamp.strftime("%Y%m%dT%H%M%S")
    report_dir = report_root / stamp
    suffix = 1
    while report_dir.exists():
        report_dir = report_root / f"{stamp}-{suffix:02d}"
        suffix += 1
    json_path = report_dir / REPORT_JSON_FILENAME
    markdown_path = report_dir / REPORT_MARKDOWN_FILENAME
    ensure_directory(json_path)

    payload_dict = copy.deepcopy(sanitized_payload)
    _scrub_relative_paths(payload_dict)
    json_path.write_text(json.dumps(payload_dict, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(payload_dict), encoding="utf-8")

    return (
        relative_to_repo(json_path),
        relative_to_repo(markdown_path),
    )


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render the ingestion QA payload as Markdown."""

    lines: list[str] = []
    lines.append(f"# Ingestion QA Report ({payload.get('version')})")
    lines.append("")
    lines.append(f"- Generated: {payload.get('generated_at')}")
    lines.append(f"- Status: {str(payload.get('status', '')).upper()}")
    lines.append(f"- Summary: {payload.get('summary')}")
    lines.append("")

    metrics_obj = payload.get("metrics", {})
    metrics = metrics_obj if isinstance(metrics_obj, Mapping) else {}
    lines.append("## Metrics")
    lines.append("")
    token_stats_obj = metrics.get("token_estimate", {})
    token_stats = token_stats_obj if isinstance(token_stats_obj, Mapping) else {}
    lines.append(f"- Files processed: {metrics.get('files_processed', 0)}")
    lines.append(f"- Chunks processed: {metrics.get('chunks_processed', 0)}")
    lines.append(f"- Token estimate total: {token_stats.get('total', 0)}")
    lines.append(f"- Token estimate mean: {round(token_stats.get('mean', 0.0), 2)}")
    lines.append(f"- Missing embeddings: {metrics.get('missing_embeddings', 0)}")
    lines.append(f"- Orphan chunks: {metrics.get('orphan_chunks', 0)}")
    lines.append(f"- Checksum mismatches: {metrics.get('checksum_mismatches', 0)}")
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
    if _is_sequence(files):
        lines.append("## Files")
        lines.append("")
        lines.append("| Relative Path | Chunks | Checksum | Git Commit |")
        lines.append("| --- | ---: | --- | --- |")
        for entry in files:
            if isinstance(entry, Mapping):
                lines.append(
                    "| {relative} | {chunks} | {checksum} | {commit} |".format(
                        relative=entry.get("relative_path", "***"),
                        chunks=entry.get("chunks", "***"),
                        checksum=entry.get("document_checksum", "***"),
                        commit=entry.get("git_commit") or "-",
                    )
                )
        lines.append("")

    return "\n".join(lines)


def _is_sequence(value: Any) -> bool:
    """Return True when `value` is a non-string sequence."""

    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _scrub_relative_paths(payload: Mapping[str, Any]) -> None:
    """Remove absolute filesystem paths from the payload's file listings."""

    metrics_obj = payload.get("metrics")
    if not isinstance(metrics_obj, Mapping):
        return
    files = metrics_obj.get("files")
    if not isinstance(files, Sequence):
        return
    for entry in files:
        if not isinstance(entry, dict):
            continue
        relative = entry.get("relative_path")
        if isinstance(relative, str) and Path(relative).is_absolute():
            try:
                entry["relative_path"] = Path(relative).name
            except ValueError:  # pragma: no cover - defensive fallback
                entry["relative_path"] = "***"


__all__ = [
    "REPORT_JSON_FILENAME",
    "REPORT_MARKDOWN_FILENAME",
    "render_markdown",
    "write_ingestion_report",
]
