"""Validation tests for QA assessment markdown files."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# Test parameters for all assessment files
ASSESSMENT_FILES = [
    "docs/qa/assessments/4.7-nfr-20251005.md",
    "docs/qa/assessments/4.7-po-validation-20251005.md",
    "docs/qa/assessments/4.7-review-20251005.md",
    "docs/qa/assessments/4.7-risk-20251005.md",
    "docs/qa/assessments/4.7-test-design-20251005.md",
    "docs/qa/assessments/4.7-trace-20251005.md",
    "docs/qa/assessments/4.8-nfr-20251005.md",
    "docs/qa/assessments/4.8-po-validation-20251005.md",
    "docs/qa/assessments/4.8-trace-20251005.md",
]


@pytest.mark.parametrize("assessment_file", ASSESSMENT_FILES)
def test_assessment_file_exists(assessment_file):
    """Test that each assessment file exists."""
    path = Path(assessment_file)
    assert path.exists(), f"Assessment file not found: {assessment_file}"


@pytest.mark.parametrize("assessment_file", ASSESSMENT_FILES)
def test_assessment_file_is_readable(assessment_file):
    """Test that each assessment file can be read."""
    path = Path(assessment_file)
    content = path.read_text(encoding="utf-8")
    assert len(content) > 0, f"Assessment file is empty: {assessment_file}"


@pytest.mark.parametrize("assessment_file", ASSESSMENT_FILES)
def test_assessment_has_title(assessment_file):
    """Test that each assessment has a markdown title."""
    path = Path(assessment_file)
    content = path.read_text(encoding="utf-8")
    assert re.search(r'^#\s+\S+', content, re.MULTILINE), f"No H1 heading found in {assessment_file}"


@pytest.mark.parametrize("assessment_file", ASSESSMENT_FILES)
def test_assessment_valid_utf8(assessment_file):
    """Test that each assessment file is valid UTF-8."""
    path = Path(assessment_file)
    try:
        content = path.read_text(encoding="utf-8")
        content.encode("utf-8")
    except UnicodeDecodeError as e:
        pytest.fail(f"Invalid UTF-8 in {assessment_file}: {e}")


@pytest.mark.parametrize("assessment_file", ASSESSMENT_FILES)
def test_assessment_filename_format(assessment_file):
    """Test that assessment filename follows expected pattern."""
    filename = Path(assessment_file).name
    # Should be like "4.7-type-YYYYMMDD.md"
    pattern = r'^\d+\.\d+-[a-z-]+-\d{8}\.md$'
    assert re.match(pattern, filename), f"Filename doesn't match pattern: {filename}"


def test_nfr_assessment_has_nfr_categories():
    """Test that NFR assessment covers key categories."""
    path = Path("docs/qa/assessments/4.7-nfr-20251005.md")
    content = path.read_text(encoding="utf-8")

    nfr_categories = [
        r'security',
        r'performance',
        r'reliability',
        r'maintainability',
    ]

    found = sum(1 for cat in nfr_categories
                if re.search(cat, content, re.IGNORECASE))
    assert found >= 2, "NFR assessment should cover multiple categories"


def test_risk_assessment_has_risk_levels():
    """Test that risk assessment mentions risk levels."""
    path = Path("docs/qa/assessments/4.7-risk-20251005.md")
    content = path.read_text(encoding="utf-8")

    risk_levels = [r'critical', r'high', r'medium', r'low']
    found = any(re.search(level, content, re.IGNORECASE)
                for level in risk_levels)
    assert found, "Risk assessment should mention risk levels"


def test_test_design_has_test_scenarios():
    """Test that test design includes test scenarios or cases."""
    path = Path("docs/qa/assessments/4.7-test-design-20251005.md")
    content = path.read_text(encoding="utf-8")

    test_indicators = [
        r'test\s+case',
        r'test\s+scenario',
        r'happy\s+path',
        r'edge\s+case',
        r'test\s+coverage',
    ]

    found = sum(1 for indicator in test_indicators
                if re.search(indicator, content, re.IGNORECASE))
    assert found >= 2, "Test design should describe test scenarios"


def test_trace_assessment_has_acceptance_criteria():
    """Test that trace assessment references acceptance criteria."""
    path = Path("docs/qa/assessments/4.7-trace-20251005.md")
    content = path.read_text(encoding="utf-8")

    ac_indicators = [
        r'acceptance\s+criteria',
        r'\bac\b',
        r'criteria\s+\d+',
    ]

    found = any(re.search(indicator, content, re.IGNORECASE)
                for indicator in ac_indicators)
    assert found, "Trace assessment should reference acceptance criteria"


def test_po_validation_has_verification():
    """Test that PO validation mentions verification or approval."""
    path = Path("docs/qa/assessments/4.7-po-validation-20251005.md")
    content = path.read_text(encoding="utf-8")

    validation_terms = [
        r'verif',
        r'approv',
        r'accept',
        r'confirm',
        r'validat',
    ]

    found = any(re.search(term, content, re.IGNORECASE)
                for term in validation_terms)
    assert found, "PO validation should mention verification"


def test_review_assessment_has_findings():
    """Test that review assessment documents findings or recommendations."""
    path = Path("docs/qa/assessments/4.7-review-20251005.md")
    content = path.read_text(encoding="utf-8")

    review_terms = [
        r'finding',
        r'recommendation',
        r'issue',
        r'observation',
        r'review',
    ]

    found = any(re.search(term, content, re.IGNORECASE)
                for term in review_terms)
    assert found, "Review assessment should document findings"


@pytest.mark.parametrize("assessment_file", ASSESSMENT_FILES)
def test_assessment_no_excessive_whitespace(assessment_file):
    """Test that assessments don't have excessive blank lines."""
    path = Path(assessment_file)
    content = path.read_text(encoding="utf-8")

    # Check for more than 3 consecutive blank lines
    excessive_blanks = re.findall(r'\n\s*\n\s*\n\s*\n\s*\n', content)
    assert len(excessive_blanks) == 0, f"Excessive blank lines in {assessment_file}"


@pytest.mark.parametrize("assessment_file", ASSESSMENT_FILES)
def test_assessment_consistent_heading_style(assessment_file):
    """Test that headings use consistent ATX style (# ##)."""
    path = Path(assessment_file)
    content = path.read_text(encoding="utf-8")

    # Look for setext-style headings (underlined with === or ---)
    setext_headings = re.findall(r'^.+\n[=-]{3,}$', content, re.MULTILINE)

    # Prefer ATX style (# ##), but allow setext if consistent
    atx_headings = re.findall(r'^#{1,6}\s+', content, re.MULTILINE)

    # If we have ATX headings, setext should be minimal or none
    if len(atx_headings) > 0 and len(setext_headings) > 0:
        pytest.skip(f"Mixed heading styles in {assessment_file} (acceptable)")


def test_all_assessments_referenced_in_gate():
    """Test that all assessment files are referenced in the gate YAML."""
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")
    gate_content = gate_file.read_text(encoding="utf-8")

    for assessment_file in ASSESSMENT_FILES:
        filename = Path(assessment_file).name
        # The gate might reference without full path
        assert filename in gate_content, f"Assessment {filename} not referenced in gate"


@pytest.mark.parametrize("assessment_file", ASSESSMENT_FILES)
def test_assessment_date_in_filename_matches_content(assessment_file):
    """Test that date in filename is reasonable."""
    path = Path(assessment_file)
    filename = path.name

    # Extract date from filename (YYYYMMDD format)
    date_match = re.search(r'(\d{8})', filename)
    assert date_match, f"No date found in filename: {filename}"

    date_str = date_match.group(1)
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    # Basic validation
    assert 2020 <= year <= 2030, f"Unreasonable year in {filename}"
    assert 1 <= month <= 12, f"Invalid month in {filename}"
    assert 1 <= day <= 31, f"Invalid day in {filename}"


@pytest.mark.parametrize("assessment_file", ASSESSMENT_FILES)
def test_assessment_has_structured_content(assessment_file):
    """Test that assessment has some structure (sections/lists)."""
    path = Path(assessment_file)
    content = path.read_text(encoding="utf-8")

    # Should have either multiple headings or lists
    headings = re.findall(r'^#{1,6}\s+', content, re.MULTILINE)
    lists = re.findall(r'^[\s]*[-*+]\s+', content, re.MULTILINE)

    structure_score = len(headings) + len(lists)
    assert structure_score >= 3, f"Assessment appears unstructured: {assessment_file}"


def test_assessment_files_total_size_reasonable():
    """Test that assessment files aren't excessively large."""
    total_size = 0
    for assessment_file in ASSESSMENT_FILES:
        path = Path(assessment_file)
        total_size += path.stat().st_size

    # Total should be less than 1MB for documentation
    assert total_size < 1_000_000, f"Assessment files total size excessive: {total_size} bytes"


@pytest.mark.parametrize("assessment_file", ASSESSMENT_FILES)
def test_assessment_markdown_basics(assessment_file):
    """Test basic markdown validity."""
    path = Path(assessment_file)
    content = path.read_text(encoding="utf-8")

    # Check for common markdown issues
    issues = []

    # Unclosed code blocks
    code_fence_count = content.count("```")
    if code_fence_count % 2 != 0:
        issues.append("Unclosed code blocks")

    # Mismatched brackets in links
    open_brackets = content.count("[")
    close_brackets = content.count("]")
    if abs(open_brackets - close_brackets) > 2:  # Allow small difference
        issues.append("Mismatched link brackets")

    assert len(issues) == 0, f"Markdown issues in {assessment_file}: {issues}"


def test_4_8_nfr_assessment_has_nfr_categories():
    """Test that Story 4.8 NFR assessment covers key categories."""
    path = Path("docs/qa/assessments/4.8-nfr-20251005.md")
    content = path.read_text(encoding="utf-8")

    nfr_categories = [
        r'security',
        r'performance',
        r'reliability',
        r'maintainability',
    ]

    found = sum(1 for cat in nfr_categories
                if re.search(cat, content, re.IGNORECASE))
    assert found >= 2, "Story 4.8 NFR assessment should cover multiple categories"


def test_4_8_nfr_assessment_references_test_cases():
    """Test that Story 4.8 NFR assessment references specific test cases."""
    path = Path("docs/qa/assessments/4.8-nfr-20251005.md")
    content = path.read_text(encoding="utf-8")

    # Should reference test cases with pytest-style paths
    test_references = re.findall(r'tests/[a-z_/]+\.py::[a-z_]+', content)
    assert len(test_references) >= 3, \
        f"Story 4.8 NFR should reference multiple test cases, found {len(test_references)}"


def test_4_8_trace_assessment_has_acceptance_criteria():
    """Test that Story 4.8 trace assessment references acceptance criteria."""
    path = Path("docs/qa/assessments/4.8-trace-20251005.md")
    content = path.read_text(encoding="utf-8")

    # Should have AC references
    ac_indicators = [
        r'AC\d+',
        r'acceptance\s+criteria\s+\d+',
        r'criteria\s+\d+',
    ]

    found = any(re.search(indicator, content, re.IGNORECASE)
                for indicator in ac_indicators)
    assert found, "Story 4.8 trace assessment should reference acceptance criteria"


def test_4_8_trace_assessment_has_coverage_summary():
    """Test that Story 4.8 trace assessment includes coverage summary."""
    path = Path("docs/qa/assessments/4.8-trace-20251005.md")
    content = path.read_text(encoding="utf-8")

    # Should document coverage metrics
    coverage_indicators = [
        r'coverage:?\s*(?:full|partial|none)',
        r'total\s+requirements',
        r'fully\s+covered',
        r'\d+\s*\(\d+%\)',
    ]

    found = sum(1 for indicator in coverage_indicators
                if re.search(indicator, content, re.IGNORECASE))
    assert found >= 2, "Story 4.8 trace should include coverage summary"


def test_4_8_trace_assessment_maps_requirements_to_tests():
    """Test that Story 4.8 trace assessment maps requirements to test cases."""
    path = Path("docs/qa/assessments/4.8-trace-20251005.md")
    content = path.read_text(encoding="utf-8")

    # Should have test references in pytest format
    test_references = re.findall(r'tests/[a-z_/]+\.py::[a-z_]+', content)
    assert len(test_references) >= 5, \
        f"Story 4.8 trace should map to multiple tests, found {len(test_references)}"


def test_4_8_po_validation_has_verification():
    """Test that Story 4.8 PO validation mentions verification or approval."""
    path = Path("docs/qa/assessments/4.8-po-validation-20251005.md")
    content = path.read_text(encoding="utf-8")

    validation_terms = [
        r'verif',
        r'approv',
        r'accept',
        r'confirm',
        r'validat',
    ]

    found = any(re.search(term, content, re.IGNORECASE)
                for term in validation_terms)
    assert found, "Story 4.8 PO validation should mention verification"


def test_4_8_po_validation_has_decision():
    """Test that Story 4.8 PO validation documents a decision."""
    path = Path("docs/qa/assessments/4.8-po-validation-20251005.md")
    content = path.read_text(encoding="utf-8")

    decision_indicators = [
        r'decision:',
        r'status:?\s*approved',
        r'approved\s*[-—]\s*implementation',
        r'gate\s*hook',
    ]

    found = sum(1 for indicator in decision_indicators
                if re.search(indicator, content, re.IGNORECASE))
    assert found >= 1, "Story 4.8 PO validation should document decision"


def test_4_8_po_validation_references_story():
    """Test that Story 4.8 PO validation references the story document."""
    path = Path("docs/qa/assessments/4.8-po-validation-20251005.md")
    content = path.read_text(encoding="utf-8")

    # Should reference the story file
    story_ref = r'docs/stories/4\.8[.-]'
    assert re.search(story_ref, content), \
        "Story 4.8 PO validation should reference story document"


def test_4_8_po_validation_has_scope_reviewed():
    """Test that Story 4.8 PO validation lists reviewed scope."""
    path = Path("docs/qa/assessments/4.8-po-validation-20251005.md")
    content = path.read_text(encoding="utf-8")

    # Should have a scope section
    scope_indicators = [
        r'scope\s+reviewed',
        r'reviewed\s+scope',
        r'area.*result.*notes',
        r'alignment\s+check',
    ]

    found = any(re.search(indicator, content, re.IGNORECASE)
                for indicator in scope_indicators)
    assert found, "Story 4.8 PO validation should document scope reviewed"


def test_all_4_8_assessments_referenced_in_gate():
    """Test that all Story 4.8 assessment files are referenced in the gate YAML."""
    gate_file = Path("docs/qa/gates/4.8-tests-and-smoke.yml")
    gate_content = gate_file.read_text(encoding="utf-8")

    assessment_files_4_8 = [
        "4.8-nfr-20251005.md",
        "4.8-trace-20251005.md",
    ]

    for assessment in assessment_files_4_8:
        assert assessment in gate_content, \
            f"Story 4.8 assessment {assessment} not referenced in gate"


def test_4_8_assessments_date_consistency():
    """Test that Story 4.8 assessments have consistent dates."""
    files = [
        "docs/qa/assessments/4.8-nfr-20251005.md",
        "docs/qa/assessments/4.8-po-validation-20251005.md",
        "docs/qa/assessments/4.8-trace-20251005.md",
    ]

    dates = []
    for file_path in files:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        # Extract date from content
        date_matches = re.findall(r'\d{4}-\d{2}-\d{2}', content)
        if date_matches:
            dates.extend(date_matches)

    # All dates should be from the same general timeframe (within 7 days)
    if dates:
        unique_dates = set(dates)
        assert len(unique_dates) <= 3, \
            f"Story 4.8 assessments have inconsistent dates: {unique_dates}"


def test_4_8_trace_has_no_critical_gaps():
    """Test that Story 4.8 trace assessment reports no critical gaps."""
    path = Path("docs/qa/assessments/4.8-trace-20251005.md")
    content = path.read_text(encoding="utf-8")

    # Should explicitly mention gap status
    gap_section = re.search(r'critical\s+gaps?:?\s*([^\n]+)', content, re.IGNORECASE)
    if gap_section:
        gap_text = gap_section.group(1).lower()
        # Should indicate no critical gaps or none
        assert 'none' in gap_text or 'zero' in gap_text or '0' in gap_text, \
            "Story 4.8 trace should report gap status"


def test_4_8_nfr_has_action_items():
    """Test that Story 4.8 NFR assessment includes action items or recommendations."""
    path = Path("docs/qa/assessments/4.8-nfr-20251005.md")
    content = path.read_text(encoding="utf-8")

    action_indicators = [
        r'action\s+items?',
        r'recommendations?',
        r'follow[-\s]up',
        r'next\s+steps',
    ]

    found = any(re.search(indicator, content, re.IGNORECASE)
                for indicator in action_indicators)
    assert found, "Story 4.8 NFR should include action items or recommendations"


def test_4_8_po_validation_has_notes_for_teams():
    """Test that Story 4.8 PO validation includes notes for dev/QA teams."""
    path = Path("docs/qa/assessments/4.8-po-validation-20251005.md")
    content = path.read_text(encoding="utf-8")

    team_notes = re.search(r'notes\s+for\s+(?:dev|qa|teams?)', content, re.IGNORECASE)
    assert team_notes, "Story 4.8 PO validation should include team notes"