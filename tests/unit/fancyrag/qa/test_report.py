from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fancyrag.qa.report import render_markdown, write_ingestion_report


def test_write_ingestion_report_creates_timestamped_artifacts(tmp_path):
    payload = {
        "version": "ingestion-qa-report/v1",
        "generated_at": datetime(2025, 10, 4, tzinfo=timezone.utc).isoformat(),
        "status": "pass",
        "summary": "QA PASS: all thresholds satisfied",
        "metrics": {
            "files_processed": 2,
            "chunks_processed": 5,
            "missing_embeddings": 0,
            "orphan_chunks": 0,
            "checksum_mismatches": 0,
            "token_estimate": {
                "total": 1200,
                "mean": 240.0,
                "histogram": {"<= 256": 5},
            },
            "files": [
                {
                    "relative_path": "docs/a.txt",
                    "chunks": 3,
                    "document_checksum": "abc",
                    "git_commit": "deadbeef",
                }
            ],
        },
        "thresholds": {"max_missing_embeddings": 0},
        "anomalies": [],
    }

    timestamp = datetime(2025, 10, 4, 12, 30, 45, tzinfo=timezone.utc)
    json_rel, md_rel = write_ingestion_report(
        sanitized_payload=payload,
        report_root=tmp_path,
        timestamp=timestamp,
    )

    report_dir = tmp_path / "20251004T123045"
    json_file = report_dir / "quality_report.json"
    markdown_file = report_dir / "quality_report.md"

    assert json_file.exists()
    assert markdown_file.exists()

    assert Path(json_rel).is_absolute()
    assert Path(md_rel).is_absolute()

    assert json_file.read_text(encoding="utf-8").strip().startswith("{")
    assert markdown_file.read_text(encoding="utf-8").splitlines()[0] == (
        "# Ingestion QA Report (ingestion-qa-report/v1)"
    )


def test_render_markdown_handles_anomalies_and_thresholds():
    payload = {
        "version": "ingestion-qa-report/v1",
        "generated_at": "2025-10-04T00:00:00+00:00",
        "status": "fail",
        "summary": "QA FAIL: threshold breach",
        "metrics": {
            "files_processed": 1,
            "chunks_processed": 2,
            "missing_embeddings": 1,
            "orphan_chunks": 0,
            "checksum_mismatches": 0,
            "token_estimate": {"total": 100, "mean": 50.0, "histogram": {"<= 64": 2}},
            "files": [
                {
                    "relative_path": "docs/sample.txt",
                    "chunks": 2,
                    "document_checksum": "fedcba",
                    "git_commit": None,
                }
            ],
        },
        "thresholds": {"max_missing_embeddings": 0},
        "anomalies": ["missing_embeddings=1 exceeds max 0"],
    }

    markdown = render_markdown(payload)
    lines = markdown.splitlines()

    assert "- ❌ missing_embeddings=1 exceeds max 0" in lines
    assert "| docs/sample.txt | 2 | fedcba | - |" in lines
    assert "## Thresholds" in lines


def test_write_ingestion_report_sanitises_absolute_relative_paths(tmp_path):
    payload = {
        "version": "ingestion-qa-report/v1",
        "generated_at": datetime(2025, 10, 4, tzinfo=timezone.utc).isoformat(),
        "status": "pass",
        "summary": "QA PASS",
        "metrics": {
            "files_processed": 1,
            "chunks_processed": 1,
            "missing_embeddings": 0,
            "orphan_chunks": 0,
            "checksum_mismatches": 0,
            "token_estimate": {},
            "files": [
                {
                    "relative_path": str(tmp_path / "sensitive" / "payload.txt"),
                    "chunks": 1,
                    "document_checksum": "abc",
                    "git_commit": None,
                }
            ],
        },
        "thresholds": {},
        "anomalies": [],
    }

    timestamp = datetime(2025, 10, 4, 15, 0, 0, tzinfo=timezone.utc)
    json_rel, md_rel = write_ingestion_report(
        sanitized_payload=payload,
        report_root=tmp_path,
        timestamp=timestamp,
    )

    json_text = Path(json_rel).read_text(encoding="utf-8")
    md_text = Path(md_rel).read_text(encoding="utf-8")

    assert str(tmp_path) not in json_text
    assert str(tmp_path) not in md_text
    assert "payload.txt" in json_text
    assert "payload.txt" in md_text


def test_write_ingestion_report_generates_unique_directories(tmp_path):
    payload = {
        "version": "ingestion-qa-report/v1",
        "generated_at": datetime(2025, 10, 4, tzinfo=timezone.utc).isoformat(),
        "status": "pass",
        "summary": "QA PASS",
        "metrics": {
            "files_processed": 0,
            "chunks_processed": 0,
            "missing_embeddings": 0,
            "orphan_chunks": 0,
            "checksum_mismatches": 0,
            "token_estimate": {},
            "files": [],
        },
        "thresholds": {},
        "anomalies": [],
    }

    timestamp = datetime(2025, 10, 4, 18, 0, 0, tzinfo=timezone.utc)
    first_json, _ = write_ingestion_report(
        sanitized_payload=payload,
        report_root=tmp_path,
        timestamp=timestamp,
    )
    second_json, _ = write_ingestion_report(
        sanitized_payload=payload,
        report_root=tmp_path,
        timestamp=timestamp,
    )

    assert Path(first_json).parent != Path(second_json).parent
