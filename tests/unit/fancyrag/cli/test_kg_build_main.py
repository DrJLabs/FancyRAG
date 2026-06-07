from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import fancyrag.cli.kg_build_main as kg


def test_parse_args_defaults():
    args = kg._parse_args([])
    assert Path(args.source) == kg.DEFAULT_SOURCE
    assert args.source_dir is None
    assert args.include_patterns is None
    assert args.profile is None
    assert args.chunk_size is None
    assert args.chunk_overlap is None
    assert args.semantic_enabled is False
    assert args.semantic_max_concurrency == 5
    assert Path(args.log_path) == kg.DEFAULT_LOG_PATH
    assert Path(args.qa_report_dir) == kg.DEFAULT_QA_DIR


def test_parse_args_accepts_multiple_include_patterns():
    args = kg._parse_args([
        "--include-pattern",
        "**/*.txt",
        "--include-pattern",
        "**/*.md",
    ])
    assert args.include_patterns == ["**/*.txt", "**/*.md"]


def test_run_delegates_to_pipeline(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    def fake_run_pipeline(options):
        captured["options"] = options
        return {"status": "success"}

    monkeypatch.setattr(kg, "run_pipeline", fake_run_pipeline)

    log_path = tmp_path / "kg-log.json"
    result = kg.run([
        "--source",
        str(tmp_path / "input.txt"),
        "--log-path",
        str(log_path),
        "--qa-report-dir",
        str(tmp_path),
        "--qa-max-checksum-mismatches",
        "2",
        "--semantic-max-concurrency",
        "7",
    ])

    options = captured["options"]
    assert isinstance(options, kg.PipelineOptions)
    assert options.semantic_max_concurrency == 7
    assert options.qa_limits.max_checksum_mismatches == 2
    assert options.log_path == log_path.expanduser()
    assert options.source == (tmp_path / "input.txt").expanduser()
    assert result["status"] == "success"


def test_main_translates_runtime_errors(monkeypatch, capsys):
    monkeypatch.setattr(kg, "run", lambda _argv=None: (_ for _ in ()).throw(RuntimeError("boom")))
    exit_code = kg.main([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "boom" in captured.err


def test_main_success_path(monkeypatch):
    monkeypatch.setattr(kg, "run", lambda _argv=None: {"status": "success"})
    assert kg.main([]) == 0
