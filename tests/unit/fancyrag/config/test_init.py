from __future__ import annotations

import pytest

import fancyrag.config as config_module


def test_all_exports_are_available():
    """Test that all items in __all__ are importable."""
    for name in config_module.__all__:
        assert hasattr(config_module, name), f"{name} not available in config module"


def test_default_schema_is_exported():
    """Test that DEFAULT_SCHEMA is available from config module."""
    assert hasattr(config_module, "DEFAULT_SCHEMA")
    assert config_module.DEFAULT_SCHEMA is not None


def test_default_schema_filename_is_exported():
    """Test that DEFAULT_SCHEMA_FILENAME is available."""
    assert hasattr(config_module, "DEFAULT_SCHEMA_FILENAME")
    assert config_module.DEFAULT_SCHEMA_FILENAME == "kg_schema.json"


def test_default_schema_path_is_exported():
    """Test that DEFAULT_SCHEMA_PATH is available."""
    assert hasattr(config_module, "DEFAULT_SCHEMA_PATH")
    assert config_module.DEFAULT_SCHEMA_PATH.name == "kg_schema.json"


def test_graphschema_class_is_exported():
    """Test that GraphSchema class is available."""
    assert hasattr(config_module, "GraphSchema")
    assert config_module.GraphSchema is not None


def test_load_default_schema_is_exported():
    """Test that load_default_schema function is available."""
    assert hasattr(config_module, "load_default_schema")
    assert callable(config_module.load_default_schema)


def test_load_schema_is_exported():
    """Test that load_schema function is available."""
    assert hasattr(config_module, "load_schema")
    assert callable(config_module.load_schema)


def test_resolve_schema_path_is_exported():
    """Test that resolve_schema_path function is available."""
    assert hasattr(config_module, "resolve_schema_path")
    assert callable(config_module.resolve_schema_path)


def test_all_list_matches_actual_exports():
    """Test that __all__ list matches actual public exports."""
    expected = {
        "DEFAULT_SCHEMA",
        "DEFAULT_SCHEMA_FILENAME",
        "DEFAULT_SCHEMA_PATH",
        "GraphSchema",
        "load_default_schema",
        "load_schema",
        "resolve_schema_path",
    }
    actual = set(config_module.__all__)
    assert actual == expected, f"__all__ mismatch: {actual ^ expected}"


def test_imports_from_schema_module():
    """Test that config module imports from schema submodule correctly."""
    from fancyrag.config.schema import (
        DEFAULT_SCHEMA,
        DEFAULT_SCHEMA_FILENAME,
        DEFAULT_SCHEMA_PATH,
    )
    
    assert config_module.DEFAULT_SCHEMA is DEFAULT_SCHEMA
    assert config_module.DEFAULT_SCHEMA_FILENAME == DEFAULT_SCHEMA_FILENAME
    assert config_module.DEFAULT_SCHEMA_PATH == DEFAULT_SCHEMA_PATH


def test_module_level_functions_callable():
    """Test that all exported functions are callable."""
    assert callable(config_module.load_schema)
    assert callable(config_module.load_default_schema)
    assert callable(config_module.resolve_schema_path)


def test_no_private_exports_in_all():
    """Test that __all__ doesn't export private members."""
    for name in config_module.__all__:
        assert not name.startswith("_"), f"Private member {name} in __all__"


def test_graphschema_has_model_validate():
    """Test that exported GraphSchema has model_validate method."""
    assert hasattr(config_module.GraphSchema, "model_validate")
    assert callable(config_module.GraphSchema.model_validate)