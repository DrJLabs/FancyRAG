from __future__ import annotations

import json
from pathlib import Path

import pytest

import fancyrag.config.schema as schema


def test_resolve_schema_path_defaults_to_known_location():
    assert schema.resolve_schema_path() == schema.DEFAULT_SCHEMA_PATH


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
    with pytest.raises(RuntimeError) as excinfo:
        schema.load_schema(missing)
    assert f"schema file not found: {missing}" in str(excinfo.value)


def test_load_default_schema_missing_raises(monkeypatch, tmp_path):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(schema, "DEFAULT_SCHEMA_PATH", missing)
    with pytest.raises(RuntimeError) as excinfo:
        schema.load_default_schema()
    assert f"default schema file not found: {missing}" in str(excinfo.value)


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
