from __future__ import annotations

import ast
import inspect
from pathlib import Path

import fancyrag.kg.phases as phases


def _get_phase_functions_from_pipeline() -> set[str]:
    repo_root = Path(__file__).resolve().parents[4]
    pipeline_source = (repo_root / "src/fancyrag/kg/pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(pipeline_source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == ".phases":
            for alias in node.names:
                attr = getattr(phases, alias.name, None)
                if inspect.isfunction(attr):
                    names.add(alias.asname or alias.name)
    return names


def test_architecture_doc_lists_pipeline_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    doc_path = repo_root / "docs/architecture/projects/fancyrag-kg-build-refactor.md"
    content = doc_path.read_text(encoding="utf-8")
    helper_functions = _get_phase_functions_from_pipeline()
    missing = sorted(name for name in helper_functions if name not in content)
    assert not missing, f"Missing helper references in architecture doc: {missing}"
