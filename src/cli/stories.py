"""Story management utilities for BMAD workflows."""

from __future__ import annotations

import argparse
import getpass
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_STORY_DIR = Path("docs") / "stories"
def override_log_path() -> Path:
    env_override = os.getenv("BMAD_STORY_OVERRIDE_LOG")
    if env_override:
        return Path(env_override)
    return Path("docs") / "bmad" / "story-overrides.md"


@dataclass
class StoryInfo:
    path: Path
    identifier: str
    status: str


class StoryValidationError(RuntimeError):
    """Raised when the next-story guard fails."""


def _discover_stories(stories_dir: Path, exclude: Optional[Path] = None) -> list[Path]:
    if not stories_dir.exists():
        raise StoryValidationError(f"Stories directory not found: {stories_dir}")
    stories = sorted(stories_dir.glob("*.md"))
    if exclude is not None:
        stories = [path for path in stories if path.resolve() != exclude.resolve()]
    return stories


def _read_status(story_path: Path) -> str:
    status_heading = "## Status"
    with story_path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]
    for idx, line in enumerate(lines):
        if line.strip() == status_heading and idx + 1 < len(lines):
            return lines[idx + 1].strip()
    raise StoryValidationError(f"Unable to determine status for {story_path}")


def _story_identifier(story_path: Path) -> str:
    return story_path.stem.split(".")[0:2][0] if "." not in story_path.stem else ".".join(story_path.stem.split(".")[0:2])


def _load_latest_story(stories_dir: Path, *, exclude: Optional[Path] = None) -> StoryInfo:
    stories = _discover_stories(stories_dir, exclude=exclude)
    if not stories:
        raise StoryValidationError("No existing stories found.")
    latest = stories[-1]
    status = _read_status(latest)
    identifier = _story_identifier(latest)
    return StoryInfo(path=latest, identifier=identifier, status=status)


def _insert_override_note(story_path: Path, note: str) -> None:
    content = story_path.read_text(encoding="utf-8")
    anchor = "## Dev Notes"
    if anchor not in content:
        story_path.write_text(content + "\n\n" + note + "\n", encoding="utf-8")
        return
    head, tail = content.split(anchor, maxsplit=1)
    updated = f"{head}{anchor}\n\n{note}\n{tail.lstrip()}"
    story_path.write_text(updated, encoding="utf-8")


def _ensure_log_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Story Override Log\n\n"
        "| Timestamp (UTC) | Actor | Prior Story | Prior Status | Reason |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    path.write_text(header, encoding="utf-8")


def _append_log_entry(path: Path, *, timestamp: datetime, actor: str, prior: StoryInfo, reason: str) -> None:
    _ensure_log_header(path)
    line = f"| {timestamp.isoformat()} | {actor} | {prior.identifier} | {prior.status} | {reason} |\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _build_override_note(*, timestamp: datetime, actor: str, prior: StoryInfo, reason: str) -> str:
    return (
        "<!-- override-note -->\n"
        + "Warning: override of incomplete story status executed.\n"
        + f"- Timestamp (UTC): {timestamp.isoformat()}\n"
        + f"- Actor: {actor}\n"
        + f"- Prior Story: {prior.identifier}\n"
        + f"- Prior Status: {prior.status}\n"
        + f"- Reason: {reason}\n"
    )


def guard_next_story(
    *,
    stories_dir: Path,
    new_story: Optional[Path],
    override: bool,
    reason: str,
    actor: str,
    timestamp: datetime,
) -> None:
    prior = _load_latest_story(stories_dir, exclude=new_story)
    if prior.status.lower() != "done".lower() and not override:
        raise StoryValidationError(
            f"Prior story {prior.identifier} status is '{prior.status}'. Use --override-incomplete to proceed consciously."
        )

    if not override:
        return

    reason_text = reason or "Not provided"
    log_path = override_log_path()
    _append_log_entry(log_path, timestamp=timestamp, actor=actor, prior=prior, reason=reason_text)

    if new_story is not None:
        if not new_story.exists():
            new_story.parent.mkdir(parents=True, exist_ok=True)
            new_story.write_text("# Pending Story\n\n## Dev Notes\n", encoding="utf-8")
        note = _build_override_note(timestamp=timestamp, actor=actor, prior=prior, reason=reason_text)
        _insert_override_note(new_story, note)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage BMAD story creation guardrails.")
    parser.add_argument(
        "--stories-dir",
        type=Path,
        default=DEFAULT_STORY_DIR,
        help="Directory containing story markdown documents (default: docs/stories)",
    )
    parser.add_argument(
        "--new-story",
        type=Path,
        help="Path to the story being created; used to record override acknowledgements.",
    )
    parser.add_argument(
        "--override-incomplete",
        action="store_true",
        help="Allow proceeding even if the previous story is not Done.",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="",
        help="Reason for invoking the override when prior story is incomplete.",
    )
    parser.add_argument(
        "--actor",
        type=str,
        default=getpass.getuser(),
        help="Name of the operator performing the action (defaults to the current user).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    timestamp = datetime.now(timezone.utc)
    try:
        guard_next_story(
            stories_dir=args.stories_dir,
            new_story=args.new_story,
            override=args.override_incomplete,
            reason=args.reason,
            actor=args.actor,
            timestamp=timestamp,
        )
    except StoryValidationError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
