from __future__ import annotations

import builtins
import importlib

import pytest


def test_structured_output_defers_graphrag_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import fancyrag.kg.structured_output as structured_output

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("neo4j_graphrag"):
            raise ModuleNotFoundError("neo4j_graphrag is not installed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.reload(structured_output)

    with pytest.raises(RuntimeError, match="neo4j_graphrag"):
        module.build_neo4j_graph_schema()
