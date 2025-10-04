from __future__ import annotations

import types
from pathlib import Path

import pytest

from fancyrag.utils.paths import ensure_directory, relative_to_repo, resolve_repo_root
import fancyrag.utils.paths as paths_mod


def test_ensure_directory_creates_parents(tmp_path):
    target = tmp_path / "nested" / "deeper" / "file.json"
    parent = target.parent
    assert not parent.exists()
    ensure_directory(target)
    assert parent.exists() and parent.is_dir()


def test_relative_to_repo_prefers_base(tmp_path):
    base = tmp_path / "base"
    inner = base / "inner"
    inner.mkdir(parents=True, exist_ok=True)
    file_path = inner / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    rel = relative_to_repo(file_path, base=base)
    assert rel.replace("\\", "/") == "inner/file.txt"


def test_relative_to_repo_uses_repo_root_when_available(monkeypatch, tmp_path):
    repo_root = tmp_path
    monkeypatch.setattr(paths_mod, "resolve_repo_root", lambda: repo_root)
    file_path = repo_root / "p" / "q.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("ok", encoding="utf-8")
    rel = relative_to_repo(file_path)
    assert rel.replace("\\", "/") == "p/q.txt"


def test_relative_to_repo_falls_back_to_absolute(monkeypatch, tmp_path):
    # Simulate no repo root available
    monkeypatch.setattr(paths_mod, "resolve_repo_root", lambda: None)
    outside = tmp_path / "x" / "y.txt"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("ok", encoding="utf-8")
    rel = relative_to_repo(outside)
    # Not relative to CWD or repo root -> absolute path string
    assert Path(rel).is_absolute()
    assert Path(rel) == outside.resolve()


def test_resolve_repo_root_none_when_git_missing(monkeypatch):
    # shutil.which("git") -> None
    monkeypatch.setattr(paths_mod.shutil, "which", lambda _name: None)
    assert resolve_repo_root() is None


def test_resolve_repo_root_parses_git_output(monkeypatch, tmp_path):
    # Simulate git present and rev-parse returns tmp_path
    monkeypatch.setattr(paths_mod.shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(
        paths_mod.subprocess,
        "run",
        lambda *_, **__: types.SimpleNamespace(stdout=str(tmp_path) + "\n"),
    )
    root = resolve_repo_root()
    assert isinstance(root, Path)
    assert root == tmp_path