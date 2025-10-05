"""Validation tests for QA gate YAML files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest


def test_gate_yaml_exists():
    """Test that the 4.7 gate YAML file exists."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    assert gate_file.exists(), f"Gate file not found: {gate_file}"


def test_gate_yaml_is_readable():
    """Test that the gate YAML file can be read."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    assert len(content) > 0, "Gate file is empty"


def test_gate_yaml_has_required_fields():
    """Test that gate YAML contains all required fields."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    required_fields = [
        "schema:",
        "story:",
        "story_title:",
        "gate:",
        "status_reason:",
        "reviewer:",
        "updated:",
        "quality_score:",
    ]
    
    for field in required_fields:
        assert field in content, f"Required field missing: {field}"


def test_gate_yaml_has_valid_gate_status():
    """Test that gate status is one of the valid values."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    # Extract gate value
    for line in content.split("\n"):
        if line.startswith("gate:"):
            gate_value = line.split(":", 1)[1].strip().strip('"')
            assert gate_value in ["PASS", "FAIL", "WAIVED", "PENDING"], \
                f"Invalid gate status: {gate_value}"
            break


def test_gate_yaml_has_valid_story_number():
    """Test that story number is in expected format."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    for line in content.split("\n"):
        if line.startswith("story:"):
            story_value = line.split(":", 1)[1].strip().strip('"')
            # Should be in format like "4.7"
            assert "." in story_value, "Story number should contain a dot"
            parts = story_value.split(".")
            assert len(parts) >= 2, "Story number should have major.minor format"
            break


def test_gate_yaml_quality_score_is_numeric():
    """Test that quality_score is a valid number."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    for line in content.split("\n"):
        if line.startswith("quality_score:"):
            score = line.split(":", 1)[1].strip()
            try:
                score_value = int(score)
                assert 0 <= score_value <= 100, "Quality score should be 0-100"
            except ValueError:
                pytest.fail(f"Quality score is not numeric: {score}")
            break


def test_gate_yaml_has_risk_summary_section():
    """Test that risk_summary section exists."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    assert "risk_summary:" in content, "risk_summary section missing"
    assert "totals:" in content, "risk_summary.totals missing"
    assert "critical:" in content, "risk counts missing"


def test_gate_yaml_has_evidence_section():
    """Test that evidence section exists with required subsections."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    assert "evidence:" in content, "evidence section missing"
    assert "tests_reviewed:" in content or "trace:" in content, \
        "evidence should document tests or trace"


def test_gate_yaml_has_nfr_validation():
    """Test that NFR (non-functional requirements) validation exists."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    assert "nfr_validation:" in content, "nfr_validation section missing"
    # Check for common NFR categories
    nfr_categories = ["security:", "performance:", "reliability:", "maintainability:"]
    found_categories = sum(1 for cat in nfr_categories if cat in content)
    assert found_categories > 0, "No NFR categories found"


def test_gate_yaml_has_references_section():
    """Test that references section exists."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    assert "references:" in content, "references section missing"


def test_gate_yaml_updated_timestamp_format():
    """Test that updated timestamp is in ISO format."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    for line in content.split("\n"):
        if line.startswith("updated:"):
            timestamp = line.split(":", 1)[1].strip().strip('"')
            # Should be parseable as ISO format
            try:
                # Remove timezone for parsing
                if "T" in timestamp:
                    date_part = timestamp.split("T")[0]
                    assert len(date_part.split("-")) == 3, "Date should be YYYY-MM-DD"
            except AssertionError as e:
                pytest.fail(f"Invalid timestamp format: {timestamp}, error: {e}")
            break


def test_gate_yaml_references_point_to_valid_paths():
    """Test that referenced files in YAML actually exist."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    # Extract reference paths
    reference_paths = []
    in_references = False
    for line in content.split("\n"):
        if line.strip().startswith("references:"):
            in_references = True
            continue
        if in_references:
            if line.strip() and not line.startswith(" "):
                break  # End of references section
            if ":" in line and "docs/" in line:
                # Extract path from line like "  story: docs/stories/4.7.schema-and-env.md"
                path = line.split(":", 1)[1].strip()
                if path and not path.startswith("{"):
                    reference_paths.append(path)
    
    for ref_path in reference_paths:
        path_obj = Path(ref_path)
        assert path_obj.exists(), f"Referenced file does not exist: {ref_path}"


def test_gate_yaml_has_reviewer_info():
    """Test that reviewer information is present."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    for line in content.split("\n"):
        if line.startswith("reviewer:"):
            reviewer = line.split(":", 1)[1].strip().strip('"')
            assert len(reviewer) > 0, "Reviewer field is empty"
            break


def test_gate_yaml_waiver_field_exists():
    """Test that waiver field exists and has active status."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    assert "waiver:" in content, "waiver field missing"
    # Should have 'active' field in waiver
    if "waiver:" in content:
        waiver_section = False
        for line in content.split("\n"):
            if "waiver:" in line:
                waiver_section = True
            if waiver_section and "active:" in line:
                active_value = line.split(":", 1)[1].strip()
                assert active_value.lower() in ["true", "false"], \
                    f"waiver.active should be boolean: {active_value}"
                break


def test_gate_yaml_has_recommendations():
    """Test that recommendations section exists."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    assert "recommendations:" in content, "recommendations section missing"


def test_gate_yaml_line_endings():
    """Test that YAML file uses consistent line endings."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_bytes()
    
    # Check for mixed line endings
    has_crlf = b"\r\n" in content
    has_lf_only = b"\n" in content
    
    if has_crlf and has_lf_only:
        # Count occurrences
        crlf_count = content.count(b"\r\n")
        lf_only_count = content.count(b"\n") - crlf_count
        
        # Allow if one type dominates (>90%)
        total = crlf_count + lf_only_count
        if total > 0:
            dominant_ratio = max(crlf_count, lf_only_count) / total
            assert dominant_ratio > 0.9, "Mixed line endings detected"


def test_gate_yaml_no_tab_characters():
    """Test that YAML uses spaces, not tabs."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    lines_with_tabs = [i + 1 for i, line in enumerate(content.split("\n")) if "\t" in line]
    assert len(lines_with_tabs) == 0, f"Tabs found on lines: {lines_with_tabs}"


def test_gate_yaml_proper_indentation():
    """Test that YAML has consistent indentation (2 or 4 spaces)."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    content = gate_file.read_text(encoding="utf-8")
    
    indent_counts = {}
    for line in content.split("\n"):
        if line and line[0] == " ":
            # Count leading spaces
            spaces = len(line) - len(line.lstrip())
            indent_counts[spaces] = indent_counts.get(spaces, 0) + 1
    
    # Check that indents are multiples of 2
    if indent_counts:
        for indent in indent_counts:
            assert indent % 2 == 0, f"Non-standard indentation: {indent} spaces"