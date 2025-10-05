from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import fancyrag.config.schema as schema


def test_resolve_schema_path_returns_existing_default(monkeypatch, tmp_path):
    schema_path = tmp_path / schema.DEFAULT_SCHEMA_FILENAME
    schema_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(schema, "DEFAULT_SCHEMA_PATH", schema_path)

    assert schema.resolve_schema_path() == schema_path


def test_resolve_schema_path_defaults_to_known_location(monkeypatch, tmp_path):
    monkeypatch.setattr(schema, "DEFAULT_SCHEMA_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError) as excinfo:
        schema.resolve_schema_path()
    assert "default schema file does not exist" in str(excinfo.value)


def test_resolve_schema_path_handles_relative_repo_path(monkeypatch, tmp_path):
    monkeypatch.setattr(schema, "_PROJECT_ROOT", tmp_path)
    target = tmp_path / "schemas" / "repo.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")

    resolved = schema.resolve_schema_path(Path("schemas/repo.json"))

    assert resolved == target


def test_resolve_schema_path_handles_relative_cwd(monkeypatch, tmp_path):
    target = tmp_path / "cwd.json"
    target.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    resolved = schema.resolve_schema_path("cwd.json")

    assert resolved == target


def test_load_schema_invokes_model_validate(monkeypatch, tmp_path):
    payload = {"nodes": []}
    schema_path = tmp_path / "custom_schema.json"
    schema_path.write_text(json.dumps(payload), encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeGraphSchema:
        @classmethod
        def model_validate(cls, data: object):
            captured["data"] = data
            return cls()

    monkeypatch.setattr(schema, "GraphSchema", FakeGraphSchema)

    result = schema.load_schema(schema_path)

    assert isinstance(result, FakeGraphSchema)
    assert captured["data"] == payload


def test_load_schema_missing_file_raises(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError) as excinfo:
        schema.load_schema(missing)
    assert "absolute schema path does not exist" in str(excinfo.value)


def test_load_default_schema_missing_raises(monkeypatch, tmp_path):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(schema, "DEFAULT_SCHEMA_PATH", missing)
    with pytest.raises(FileNotFoundError) as excinfo:
        schema.load_default_schema()
    assert "default schema file does not exist" in str(excinfo.value)


def test_load_schema_invalid_json(monkeypatch, tmp_path):
    schema_path = tmp_path / "invalid.json"
    schema_path.write_text("{ invalid", encoding="utf-8")

    class FakeGraphSchema:
        @classmethod
        def model_validate(cls, data: object):
            return cls()

    monkeypatch.setattr(schema, "GraphSchema", FakeGraphSchema)

    with pytest.raises(RuntimeError) as excinfo:
        schema.load_schema(schema_path)

    assert "invalid schema JSON" in str(excinfo.value)


def test_load_default_schema_returns_graph_schema(monkeypatch, tmp_path):
    payload = {"nodes": []}
    schema_path = tmp_path / schema.DEFAULT_SCHEMA_FILENAME
    schema_path.write_text(json.dumps(payload), encoding="utf-8")

    class FakeGraphSchema:
        @classmethod
        def model_validate(cls, data: object):
            assert data == payload
            return cls()

    monkeypatch.setattr(schema, "GraphSchema", FakeGraphSchema)
    monkeypatch.setattr(schema, "DEFAULT_SCHEMA_PATH", schema_path)

    result = schema.load_default_schema()

    assert isinstance(result, FakeGraphSchema)


def test_graph_schema_fallback_used_when_dependency_missing(monkeypatch):
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    reloaded = importlib.reload(schema)
    try:
        result = reloaded.GraphSchema.model_validate({})
        assert isinstance(result, reloaded.GraphSchema)
    finally:
        monkeypatch.setattr(importlib.util, "find_spec", original_find_spec)
        importlib.reload(reloaded)
