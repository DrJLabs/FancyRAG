import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cli import stories


def _write_story(tmp_path: Path, name: str, status: str) -> Path:
    path = tmp_path / name
    path.write_text(
        textwrap.dedent(
            f"""
            # Story Placeholder

            ## Status
            {status}

            ## Dev Notes
            Initial notes.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_guard_blocks_when_previous_story_not_done(tmp_path, monkeypatch):
    stories_dir = tmp_path / "docs" / "stories"
    stories_dir.mkdir(parents=True)
    _write_story(stories_dir, "1.3.example.md", "Ready for Review")
    monkeypatch.setenv("BMAD_STORY_OVERRIDE_LOG", str(tmp_path / "override-log.md"))

    with pytest.raises(stories.StoryValidationError):
        stories.guard_next_story(
            stories_dir=stories_dir,
            new_story=None,
            override=False,
            reason="",
            actor="tester",
            timestamp=datetime.now(timezone.utc),
        )


def test_guard_records_override_and_inserts_note(tmp_path, monkeypatch):
    stories_dir = tmp_path / "docs" / "stories"
    stories_dir.mkdir(parents=True)
    _write_story(stories_dir, "1.3.example.md", "Ready for Review")
    monkeypatch.setenv("BMAD_STORY_OVERRIDE_LOG", str(tmp_path / "override-log.md"))

    stories.guard_next_story(
        stories_dir=stories_dir,
        new_story=stories_dir / "1.4.override.md",
        override=True,
        reason="Manual approval by PO",
        actor="sam",
        timestamp=datetime(2025, 9, 25, tzinfo=timezone.utc),
    )

    log_path = Path(tmp_path / "override-log.md")
    assert log_path.exists()
    log_contents = log_path.read_text(encoding="utf-8")
    assert "sam" in log_contents
    assert "1.3" in log_contents

    story_text = (stories_dir / "1.4.override.md").read_text(encoding="utf-8")
    assert "override" in story_text.lower()
    assert "Manual approval" in story_text


def test_guard_creates_story_if_missing(tmp_path, monkeypatch):
    stories_dir = tmp_path / "docs" / "stories"
    stories_dir.mkdir(parents=True)
    _write_story(stories_dir, "1.3.example.md", "Done")
    monkeypatch.setenv("BMAD_STORY_OVERRIDE_LOG", str(tmp_path / "override-log.md"))

    new_story = stories_dir / "1.4.generated.md"

    stories.guard_next_story(
        stories_dir=stories_dir,
        new_story=new_story,
        override=True,
        reason="Testing new creation",
        actor="casey",
        timestamp=datetime(2025, 9, 25, tzinfo=timezone.utc),
    )

    assert new_story.exists()
    contents = new_story.read_text(encoding="utf-8")
    assert "## Dev Notes" in contents
    assert "override" in contents.lower()
