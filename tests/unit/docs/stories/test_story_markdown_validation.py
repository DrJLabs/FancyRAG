"""Validation tests for story markdown documentation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def test_story_markdown_exists():
    """Test that the 4.7 story markdown file exists."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    assert story_file.exists(), f"Story file not found: {story_file}"


def test_story_markdown_is_readable():
    """Test that the story markdown file can be read."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")
    assert len(content) > 0, "Story file is empty"


def test_story_markdown_has_title():
    """Test that story has a proper markdown title."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")

    # Should have at least one H1 heading
    assert re.search(r'^#\s+\S+', content, re.MULTILINE), "No H1 heading found"


def test_story_markdown_has_sections():
    """Test that story has standard sections."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")

    # Common story sections
    expected_patterns = [
        r'^#{1,2}\s+.*(?:Status|Overview|Description)',
        r'^#{1,2}\s+.*(?:Acceptance|Criteria)',
    ]

    found_sections = sum(
        1
        for pattern in expected_patterns
        if re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
    )
    assert found_sections > 0, "No standard story sections found"


def test_story_markdown_has_code_blocks():
    """Test that story has code examples (if applicable)."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")

    # Look for code blocks with triple backticks
    code_blocks = re.findall(r'```[\s\S]*?```', content)
    # This story is about schema and env, so should have code examples
    assert len(code_blocks) > 0, "No code blocks found in technical story"


def test_story_markdown_code_blocks_have_language():
    """Test that code blocks specify a language."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")

    # Find all code blocks
    code_blocks = re.findall(r'```(\w*)\n', content)
    unlabeled = [i for i, lang in enumerate(code_blocks) if not lang]

    # Allow some unlabeled blocks, but most should have language
    if code_blocks:
        labeled_ratio = (len(code_blocks) - len(unlabeled)) / len(code_blocks)
        assert labeled_ratio >= 0.5, \
            f"Less than 50% of code blocks have language specified"


def test_story_markdown_no_broken_internal_links():
    """Test that internal links to other files are valid."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")
    repo_root = story_file.parent.parent.parent

    # Find markdown links: [text](path)
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)

    for link_text, link_path in links:
        # Skip external links
        if link_path.startswith(('http://', 'https://', 'mailto:', '#')):
            continue

        # Resolve relative path
        if link_path.startswith('/'):
            target = repo_root / link_path.lstrip('/')
        else:
            target = story_file.parent / link_path

        # Remove anchor
        if '#' in link_path:
            link_path = link_path.split('#')[0]
            if link_path:  # Only check if there's a file path
                target = story_file.parent / link_path if not link_path.startswith('/') \
                    else repo_root / link_path.lstrip('/')

        if link_path and not link_path.startswith('#'):
            assert target.exists(), f"Broken link: {link_path} -> {target}"


def test_story_markdown_proper_line_length():
    """Test that lines are not excessively long (readability check)."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")

    # Check for very long lines (excluding code blocks and tables)
    in_code_block = False
    long_lines = []

    for i, line in enumerate(content.split("\n"), 1):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue

        if not in_code_block and not line.strip().startswith("|"):
            if len(line) > 120:
                long_lines.append((i, len(line)))

    # Allow some long lines, but flag if too many
    assert len(long_lines) < 5, \
        f"Too many excessively long lines (>120 chars): {long_lines[:5]}"


def test_story_markdown_has_proper_headings_hierarchy():
    """Test that headings follow proper hierarchy (no H3 before H2, etc.)."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")

    # Extract all headings with their levels
    headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)

    prev_level = 0
    for heading_marks, heading_text in headings:
        level = len(heading_marks)
        # Allow going up one level or staying same, but not jumping multiple levels
        if level > prev_level + 1:
            pytest.fail(f"Heading hierarchy jump: H{prev_level} to H{level} ({heading_text})")
        prev_level = level


def test_story_markdown_no_trailing_whitespace():
    """Test that lines don't have trailing whitespace."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")

    lines_with_trailing = [
        i + 1
        for i, line in enumerate(content.split("\n"))
        if line and line != line.rstrip()
    ]

    # Allow a few lines with trailing space (markdown line breaks use 2 spaces)
    assert len(lines_with_trailing) < 10, \
        f"Many lines with trailing whitespace: {lines_with_trailing[:5]}"


def test_story_markdown_consistent_list_formatting():
    """Test that lists use consistent formatting."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")

    # Check for mixed list styles in close proximity
    lines = content.split("\n")
    list_markers = []

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(('- ', '* ', '+ ')):
            list_markers.append((i, stripped[0]))

    # Check that we don't mix - and * in same list
    if len(list_markers) > 1:
        marker_types = set(m[1] for m in list_markers)
        # Allow both if they're in different sections (>5 lines apart)
        if len(marker_types) > 1:
            # Check for mixed markers within 5 lines
            for idx in range(len(list_markers) - 1):
                curr_idx, curr_marker = list_markers[idx]
                next_idx, next_marker = list_markers[idx + 1]
                if next_idx - curr_idx < 5 and next_marker != curr_marker:
                    pytest.fail(f"Mixed list markers near line {curr_idx}")


def test_story_markdown_valid_utf8():
    """Test that the file is valid UTF-8."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    try:
        content = story_file.read_text(encoding="utf-8")
        # Try encoding back to ensure no issues
        content.encode("utf-8")
    except UnicodeDecodeError as e:
        pytest.fail(f"Invalid UTF-8 in file: {e}")


def test_story_markdown_no_html_tags():
    """Test that markdown uses markdown syntax, not HTML (where possible)."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")

    # Look for common HTML tags that should be markdown
    problematic_html = re.findall(r'<(b|i|strong|em|h[1-6])>', content, re.IGNORECASE)

    assert len(problematic_html) == 0, \
        f"Found HTML tags that should be markdown: {set(problematic_html)}"


def test_story_markdown_has_acceptance_criteria():
    """Test that story has acceptance criteria section."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    content = story_file.read_text(encoding="utf-8")

    # Look for acceptance criteria section
    ac_patterns = [
        r'acceptance\s+criteria',
        r'definition\s+of\s+done',
        r'success\s+criteria',
    ]

    found = any(re.search(pattern, content, re.IGNORECASE) for pattern in ac_patterns)
    assert found, "No acceptance criteria section found"


def test_story_markdown_references_match_gate():
    """Test that story is referenced in the gate YAML."""
    story_file = Path("docs/stories/4.7.schema-and-env.md")
    gate_file = Path("docs/qa/gates/4.7-schema-and-env.yml")

    if gate_file.exists():
        gate_content = gate_file.read_text(encoding="utf-8")
        story_filename = story_file.name
        assert story_filename in gate_content, \
            f"Story {story_filename} not referenced in gate file"