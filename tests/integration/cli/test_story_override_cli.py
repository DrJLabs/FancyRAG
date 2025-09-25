import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_story(path: Path, status: str) -> None:
    path.write_text(
        f"# Story\n\n## Status\n{status}\n\n## Dev Notes\nplaceholder\n",
        encoding="utf-8",
    )


def _initialise_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "src", repo / "src")
    shutil.copytree(REPO_ROOT / ".bmad-core", repo / ".bmad-core")
    (repo / "docs" / "bmad" ).mkdir(parents=True, exist_ok=True)
    return repo


def _run_cli(repo: Path, stories_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["BMAD_STORY_OVERRIDE_LOG"] = str(repo / "docs" / "bmad" / "story-overrides.md")
    full_args = [sys.executable, "-m", "cli.stories", "--stories-dir", str(stories_dir), *args]
    return subprocess.run(full_args, cwd=repo, capture_output=True, text=True, env=env)


def test_cli_requires_override_when_previous_story_not_done(tmp_path):
    repo = _initialise_repo(tmp_path)
    stories_dir = repo / "docs" / "stories"
    stories_dir.mkdir(parents=True)
    _make_story(stories_dir / "1.3.prev.md", "In Progress")

    result = _run_cli(repo, stories_dir, [])

    assert result.returncode == 1
    assert "override" in result.stdout.lower()


def test_cli_override_updates_story_and_log(tmp_path):
    repo = _initialise_repo(tmp_path)
    stories_dir = repo / "docs" / "stories"
    stories_dir.mkdir(parents=True)
    _make_story(stories_dir / "1.3.prev.md", "Ready for Review")
    new_story = stories_dir / "1.4.new.md"

    result = _run_cli(
        repo,
        stories_dir,
        [
            "--new-story",
            str(new_story),
            "--override-incomplete",
            "--reason",
            "PO approved",
            "--actor",
            "ci",
        ],
    )

    assert result.returncode == 0
    log_path = repo / "docs" / "bmad" / "story-overrides.md"
    assert log_path.exists()
    log = log_path.read_text(encoding="utf-8")
    assert "PO approved" in log

    story_text = new_story.read_text(encoding="utf-8")
    assert "PO approved" in story_text
    assert "## Dev Notes" in story_text
