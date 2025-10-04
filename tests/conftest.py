from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from fancyrag.utils.env import load_project_dotenv
from fancyrag.utils.paths import resolve_repo_root


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
STUBS_DIR = ROOT_DIR / "stubs"


def _ensure_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


_ensure_path(SRC_DIR)
load_project_dotenv()


def _load_pandas_stub() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "pandas", STUBS_DIR / "pandas" / "__init__.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError("Unable to load pandas stub module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module  # type: ignore[return-value]


if "pandas" not in sys.modules:
    sys.modules["pandas"] = _load_pandas_stub()


@pytest.fixture(autouse=True)
def _clear_resolve_repo_root_cache():
    """Ensure resolve_repo_root cache does not leak between tests."""

    resolve_repo_root.cache_clear()
    yield
    resolve_repo_root.cache_clear()
