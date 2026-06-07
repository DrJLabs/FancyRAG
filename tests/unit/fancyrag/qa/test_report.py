from __future__ import annotations

import json
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


# --- Additional comprehensive tests for report.py ---

def test_render_markdown_with_empty_metrics():
    """Test render_markdown with minimal/empty metrics."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {},
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "# Ingestion QA Report (v1)" in markdown
    assert "Generated: 2025-01-01T00:00:00Z" in markdown
    assert "Status: PASS" in markdown


def test_render_markdown_with_missing_keys():
    """Test render_markdown handles missing keys gracefully."""
    payload = {
        "version": "v1",
    }
    
    markdown = render_markdown(payload)
    # Should not crash, handle missing keys gracefully
    assert "# Ingestion QA Report (v1)" in markdown


def test_render_markdown_with_non_mapping_metrics():
    """Test render_markdown when metrics is not a Mapping."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": None,  # Not a Mapping
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "## Metrics" in markdown
    # Should handle gracefully without crashing


def test_render_markdown_with_non_mapping_token_estimate():
    """Test render_markdown when token_estimate is not a Mapping."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {
            "token_estimate": None,  # Not a Mapping
        },
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "## Metrics" in markdown


def test_render_markdown_with_empty_histogram():
    """Test render_markdown with empty histogram."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {
            "token_estimate": {
                "histogram": {},
            },
        },
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "## Metrics" in markdown
    # Empty histogram should not render table


def test_render_markdown_with_non_mapping_histogram():
    """Test render_markdown when histogram is not a Mapping."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {
            "token_estimate": {
                "histogram": ["not", "a", "dict"],
            },
        },
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "## Metrics" in markdown


def test_render_markdown_histogram_multiple_buckets():
    """Test render_markdown with multiple histogram buckets."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {
            "token_estimate": {
                "histogram": {
                    "<= 64": 10,
                    "<= 128": 5,
                    "<= 256": 3,
                    "> 2048": 1,
                },
            },
        },
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "### Token Histogram" in markdown
    assert "| <= 64 | 10 |" in markdown
    assert "| <= 128 | 5 |" in markdown
    assert "| <= 256 | 3 |" in markdown
    assert "| > 2048 | 1 |" in markdown


def test_render_markdown_with_empty_files_list():
    """Test render_markdown with empty files list."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {
            "files": [],
        },
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "## Files" in markdown


def test_render_markdown_with_non_sequence_files():
    """Test render_markdown when files is not a sequence."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {
            "files": {"not": "a list"},
        },
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "## Metrics" in markdown


def test_render_markdown_with_string_files():
    """Test render_markdown when files is a string (should be ignored)."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {
            "files": "not a list",
        },
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    # String should not be treated as a sequence
    assert "## Metrics" in markdown


def test_render_markdown_files_with_non_mapping_entries():
    """Test render_markdown when file entries are not Mappings."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {
            "files": [
                "not a dict",
                123,
                None,
            ],
        },
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    # Should handle gracefully
    assert "## Files" in markdown


def test_render_markdown_files_with_missing_keys():
    """Test render_markdown when file entries are missing keys."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {
            "files": [
                {
                    # Missing all keys
                },
                {
                    "relative_path": "file.txt",
                    # Missing other keys
                },
            ],
        },
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "## Files" in markdown
    assert "| *** |" in markdown  # Default for missing values


def test_render_markdown_with_multiple_anomalies():
    """Test render_markdown with multiple anomalies."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "fail",
        "summary": "Multiple issues",
        "metrics": {},
        "thresholds": {},
        "anomalies": [
            "Issue 1: Something wrong",
            "Issue 2: Another problem",
            "Issue 3: Yet another issue",
        ],
    }
    
    markdown = render_markdown(payload)
    assert "## Findings" in markdown
    assert "- ❌ Issue 1: Something wrong" in markdown
    assert "- ❌ Issue 2: Another problem" in markdown
    assert "- ❌ Issue 3: Yet another issue" in markdown


def test_render_markdown_with_empty_anomalies():
    """Test render_markdown with empty anomalies list."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "All good",
        "metrics": {},
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "## Findings" in markdown
    assert "- ✅ All thresholds satisfied" in markdown


def test_render_markdown_with_empty_thresholds():
    """Test render_markdown with empty thresholds."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {},
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    # Empty thresholds should not render section
    assert "## Metrics" in markdown


def test_render_markdown_with_non_mapping_thresholds():
    """Test render_markdown when thresholds is not a Mapping."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {},
        "thresholds": None,
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "## Metrics" in markdown


def test_render_markdown_with_multiple_thresholds():
    """Test render_markdown with multiple threshold entries."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Test",
        "metrics": {},
        "thresholds": {
            "max_missing_embeddings": 0,
            "max_orphan_chunks": 5,
            "max_checksum_mismatches": 10,
        },
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "## Thresholds" in markdown
    assert "- max_missing_embeddings: 0" in markdown
    assert "- max_orphan_chunks: 5" in markdown
    assert "- max_checksum_mismatches: 10" in markdown


def test_write_ingestion_report_collision_handling(tmp_path):
    """Test write_ingestion_report handles directory name collisions."""
    payload = {
        "version": "v1",
        "generated_at": datetime(2025, 10, 4, tzinfo=timezone.utc).isoformat(),
        "status": "pass",
        "summary": "Test",
        "metrics": {"files_processed": 0, "chunks_processed": 0, "missing_embeddings": 0, "orphan_chunks": 0, "checksum_mismatches": 0, "token_estimate": {}, "files": []},
        "thresholds": {},
        "anomalies": [],
    }
    
    timestamp = datetime(2025, 10, 4, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create first report
    json1, md1 = write_ingestion_report(
        sanitized_payload=payload,
        report_root=tmp_path,
        timestamp=timestamp,
    )
    
    # Create second report with same timestamp
    json2, md2 = write_ingestion_report(
        sanitized_payload=payload,
        report_root=tmp_path,
        timestamp=timestamp,
    )
    
    # Create third report with same timestamp
    json3, md3 = write_ingestion_report(
        sanitized_payload=payload,
        report_root=tmp_path,
        timestamp=timestamp,
    )
    
    # All paths should be different
    assert Path(json1).parent != Path(json2).parent
    assert Path(json2).parent != Path(json3).parent
    assert Path(json1).parent != Path(json3).parent
    
    # All files should exist
    assert Path(json1).exists()
    assert Path(json2).exists()
    assert Path(json3).exists()


def test_write_ingestion_report_creates_nested_directory(tmp_path):
    """Test write_ingestion_report creates nested directory structure."""
    nested_root = tmp_path / "deep" / "nested" / "qa"
    payload = {
        "version": "v1",
        "generated_at": datetime(2025, 10, 4, tzinfo=timezone.utc).isoformat(),
        "status": "pass",
        "summary": "Test",
        "metrics": {"files_processed": 0, "chunks_processed": 0, "missing_embeddings": 0, "orphan_chunks": 0, "checksum_mismatches": 0, "token_estimate": {}, "files": []},
        "thresholds": {},
        "anomalies": [],
    }
    
    timestamp = datetime(2025, 10, 4, 12, 0, 0, tzinfo=timezone.utc)
    json_rel, md_rel = write_ingestion_report(
        sanitized_payload=payload,
        report_root=nested_root,
        timestamp=timestamp,
    )
    
    assert Path(json_rel).exists()
    assert Path(md_rel).exists()


def test_write_ingestion_report_preserves_payload_structure(tmp_path):
    """Test write_ingestion_report preserves complex payload structure."""
    payload = {
        "version": "v1",
        "generated_at": datetime(2025, 10, 4, tzinfo=timezone.utc).isoformat(),
        "status": "pass",
        "summary": "Complex test",
        "metrics": {
            "files_processed": 5,
            "chunks_processed": 20,
            "missing_embeddings": 0,
            "orphan_chunks": 0,
            "checksum_mismatches": 0,
            "token_estimate": {
                "total": 5000,
                "mean": 250.0,
                "max": 500,
                "histogram": {
                    "<= 64": 2,
                    "<= 256": 10,
                    "> 2048": 1,
                },
            },
            "char_lengths": {
                "total": 20000,
                "mean": 1000.0,
                "max": 2000,
            },
            "files": [
                {
                    "relative_path": "doc1.txt",
                    "chunks": 10,
                    "document_checksum": "abc123",
                    "git_commit": "commit1",
                },
                {
                    "relative_path": "doc2.txt",
                    "chunks": 10,
                    "document_checksum": "def456",
                    "git_commit": None,
                },
            ],
        },
        "thresholds": {
            "max_missing_embeddings": 0,
            "max_orphan_chunks": 0,
        },
        "anomalies": [],
    }
    
    timestamp = datetime(2025, 10, 4, 15, 30, 0, tzinfo=timezone.utc)
    json_rel, _ = write_ingestion_report(
        sanitized_payload=payload,
        report_root=tmp_path,
        timestamp=timestamp,
    )
    
    # Read back and verify structure
    json_content = Path(json_rel).read_text(encoding="utf-8")
    loaded = json.loads(json_content)
    
    assert loaded["version"] == "v1"
    assert loaded["status"] == "pass"
    assert loaded["metrics"]["files_processed"] == 5
    assert loaded["metrics"]["token_estimate"]["histogram"]["<= 64"] == 2
    assert len(loaded["metrics"]["files"]) == 2


def test_write_ingestion_report_does_not_modify_original_payload(tmp_path):
    """Test write_ingestion_report does not modify the input payload."""
    original_payload = {
        "version": "v1",
        "generated_at": datetime(2025, 10, 4, tzinfo=timezone.utc).isoformat(),
        "status": "pass",
        "summary": "Test",
        "metrics": {
            "files": [
                {
                    "relative_path": str(tmp_path / "abs_path.txt"),
                    "chunks": 1,
                }
            ]
        },
        "thresholds": {},
        "anomalies": [],
    }
    
    import copy
    payload_copy = copy.deepcopy(original_payload)
    
    timestamp = datetime(2025, 10, 4, 12, 0, 0, tzinfo=timezone.utc)
    write_ingestion_report(
        sanitized_payload=original_payload,
        report_root=tmp_path / "qa",
        timestamp=timestamp,
    )
    
    # Original should not be modified (we passed it as sanitized_payload)
    # But note: the function does copy.deepcopy internally
    assert original_payload == payload_copy


def test_scrub_relative_paths_with_absolute_paths(tmp_path):
    """Test _scrub_relative_paths converts absolute paths to basenames."""
    from fancyrag.qa.report import _scrub_relative_paths
    
    payload = {
        "metrics": {
            "files": [
                {
                    "relative_path": str(tmp_path / "deep" / "nested" / "file.txt"),
                    "chunks": 5,
                }
            ]
        }
    }
    
    _scrub_relative_paths(payload)
    
    # Should be scrubbed to basename
    assert payload["metrics"]["files"][0]["relative_path"] == "file.txt"


def test_scrub_relative_paths_with_relative_paths():
    """Test _scrub_relative_paths preserves relative paths."""
    from fancyrag.qa.report import _scrub_relative_paths
    
    payload = {
        "metrics": {
            "files": [
                {
                    "relative_path": "docs/file.txt",
                    "chunks": 5,
                }
            ]
        }
    }
    
    _scrub_relative_paths(payload)
    
    # Should not be modified
    assert payload["metrics"]["files"][0]["relative_path"] == "docs/file.txt"


def test_scrub_relative_paths_with_non_string_path():
    """Test _scrub_relative_paths handles non-string paths."""
    from fancyrag.qa.report import _scrub_relative_paths
    
    payload = {
        "metrics": {
            "files": [
                {
                    "relative_path": 123,
                    "chunks": 5,
                }
            ]
        }
    }
    
    _scrub_relative_paths(payload)
    
    # Should not crash, leaves non-string as-is
    assert payload["metrics"]["files"][0]["relative_path"] == 123


def test_scrub_relative_paths_with_missing_metrics():
    """Test _scrub_relative_paths handles missing metrics key."""
    from fancyrag.qa.report import _scrub_relative_paths
    
    payload = {}
    _scrub_relative_paths(payload)  # Should not crash


def test_scrub_relative_paths_with_non_mapping_metrics():
    """Test _scrub_relative_paths handles non-Mapping metrics."""
    from fancyrag.qa.report import _scrub_relative_paths
    
    payload = {
        "metrics": "not a dict",
    }
    _scrub_relative_paths(payload)  # Should not crash


def test_scrub_relative_paths_with_missing_files():
    """Test _scrub_relative_paths handles missing files key."""
    from fancyrag.qa.report import _scrub_relative_paths
    
    payload = {
        "metrics": {},
    }
    _scrub_relative_paths(payload)  # Should not crash


def test_scrub_relative_paths_with_non_sequence_files():
    """Test _scrub_relative_paths handles non-sequence files."""
    from fancyrag.qa.report import _scrub_relative_paths
    
    payload = {
        "metrics": {
            "files": "not a list",
        }
    }
    _scrub_relative_paths(payload)  # Should not crash


def test_scrub_relative_paths_with_non_dict_entries():
    """Test _scrub_relative_paths handles non-dict file entries."""
    from fancyrag.qa.report import _scrub_relative_paths
    
    payload = {
        "metrics": {
            "files": [
                "not a dict",
                123,
                None,
            ]
        }
    }
    _scrub_relative_paths(payload)  # Should not crash


def test_scrub_relative_paths_multiple_files():
    """Test _scrub_relative_paths with multiple files."""
    from fancyrag.qa.report import _scrub_relative_paths
    
    payload = {
        "metrics": {
            "files": [
                {"relative_path": "/abs/path1.txt", "chunks": 1},
                {"relative_path": "rel/path2.txt", "chunks": 2},
                {"relative_path": "/abs/path3.txt", "chunks": 3},
            ]
        }
    }
    
    _scrub_relative_paths(payload)
    
    assert payload["metrics"]["files"][0]["relative_path"] == "path1.txt"
    assert payload["metrics"]["files"][1]["relative_path"] == "rel/path2.txt"
    assert payload["metrics"]["files"][2]["relative_path"] == "path3.txt"


def test_is_sequence_helper():
    """Test _is_sequence helper function."""
    from fancyrag.qa.report import _is_sequence
    
    # Sequences
    assert _is_sequence([1, 2, 3]) is True
    assert _is_sequence((1, 2, 3)) is True
    assert _is_sequence(range(5)) is True
    
    # Non-sequences
    assert _is_sequence("string") is False
    assert _is_sequence(b"bytes") is False
    assert _is_sequence(123) is False
    assert _is_sequence(None) is False
    assert _is_sequence({"key": "value"}) is False


def test_write_ingestion_report_with_unicode_content(tmp_path):
    """Test write_ingestion_report handles Unicode content correctly."""
    payload = {
        "version": "v1",
        "generated_at": datetime(2025, 10, 4, tzinfo=timezone.utc).isoformat(),
        "status": "pass",
        "summary": "Unicode test: 你好世界 🚀",
        "metrics": {
            "files": [
                {
                    "relative_path": "文档.txt",
                    "chunks": 1,
                    "document_checksum": "abc",
                    "git_commit": "sha",
                }
            ],
            "files_processed": 1,
            "chunks_processed": 1,
            "missing_embeddings": 0,
            "orphan_chunks": 0,
            "checksum_mismatches": 0,
            "token_estimate": {},
        },
        "thresholds": {},
        "anomalies": ["异常: Unicode 问题"],
    }
    
    timestamp = datetime(2025, 10, 4, 12, 0, 0, tzinfo=timezone.utc)
    json_rel, md_rel = write_ingestion_report(
        sanitized_payload=payload,
        report_root=tmp_path,
        timestamp=timestamp,
    )
    
    # Read and verify Unicode is preserved
    json_content = Path(json_rel).read_text(encoding="utf-8")
    md_content = Path(md_rel).read_text(encoding="utf-8")
    
    assert "你好世界" in json_content
    assert "🚀" in json_content
    assert "文档.txt" in json_content
    assert "Unicode test: 你好世界 🚀" in md_content
    assert "异常: Unicode 问题" in md_content


def test_render_markdown_with_zero_values():
    """Test render_markdown displays zero values correctly."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "pass",
        "summary": "Zero values test",
        "metrics": {
            "files_processed": 0,
            "chunks_processed": 0,
            "missing_embeddings": 0,
            "orphan_chunks": 0,
            "checksum_mismatches": 0,
            "token_estimate": {
                "total": 0,
                "mean": 0.0,
                "histogram": {},
            },
        },
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "- Files processed: 0" in markdown
    assert "- Chunks processed: 0" in markdown
    assert "- Token estimate total: 0" in markdown
    assert "- Token estimate mean: 0.0" in markdown
    assert "- Missing embeddings: 0" in markdown


def test_render_markdown_status_case_conversion():
    """Test render_markdown converts status to uppercase."""
    payload = {
        "version": "v1",
        "generated_at": "2025-01-01T00:00:00Z",
        "status": "fail",
        "summary": "Test",
        "metrics": {},
        "thresholds": {},
        "anomalies": [],
    }
    
    markdown = render_markdown(payload)
    assert "- Status: FAIL" in markdown
    
    payload["status"] = "pass"
    markdown = render_markdown(payload)
    assert "- Status: PASS" in markdown


def test_constants_defined():
    """Test module constants are defined correctly."""
    from fancyrag.qa.report import REPORT_JSON_FILENAME, REPORT_MARKDOWN_FILENAME
    
    assert REPORT_JSON_FILENAME == "quality_report.json"
    assert REPORT_MARKDOWN_FILENAME == "quality_report.md"


def test_write_ingestion_report_uses_correct_filenames(tmp_path):
    """Test write_ingestion_report uses the module constants for filenames."""
    from fancyrag.qa.report import REPORT_JSON_FILENAME, REPORT_MARKDOWN_FILENAME
    
    payload = {
        "version": "v1",
        "generated_at": datetime(2025, 10, 4, tzinfo=timezone.utc).isoformat(),
        "status": "pass",
        "summary": "Test",
        "metrics": {"files_processed": 0, "chunks_processed": 0, "missing_embeddings": 0, "orphan_chunks": 0, "checksum_mismatches": 0, "token_estimate": {}, "files": []},
        "thresholds": {},
        "anomalies": [],
    }
    
    timestamp = datetime(2025, 10, 4, 12, 0, 0, tzinfo=timezone.utc)
    json_rel, md_rel = write_ingestion_report(
        sanitized_payload=payload,
        report_root=tmp_path,
        timestamp=timestamp,
    )
    
    assert Path(json_rel).name == REPORT_JSON_FILENAME
    assert Path(md_rel).name == REPORT_MARKDOWN_FILENAME
