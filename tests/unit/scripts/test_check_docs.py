from __future__ import annotations

import json
from importlib import util
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys

import pytest

MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_docs.py"
SPEC = util.spec_from_file_location("scripts.check_docs", MODULE_PATH)
check_docs = util.module_from_spec(SPEC)  # type: ignore[arg-type]
assert isinstance(SPEC, ModuleSpec) and SPEC.loader  # pragma: no cover
sys.modules.setdefault("scripts.check_docs", check_docs)
SPEC.loader.exec_module(check_docs)  # type: ignore[union-attr]


@pytest.fixture()
def temporary_docs(tmp_path: Path) -> Path:
    """Create a temporary docs tree populated with the mandatory files."""

    docs_root = tmp_path / "docs" / "architecture"
    docs_root.mkdir(parents=True)

    (docs_root / "overview.md").write_text(
        "\n".join(
            (
                "Run composed stack",
                "scripts/check_local_stack.sh --config",
                "scripts/check_local_stack.sh --up",
                "QdrantNeo4jRetriever ensures retriever wiring",
                "Use scripts/check_docs.py to keep documentation aligned.",
            )
        ),
        encoding="utf-8",
    )
    (docs_root / "source-tree.md").write_text(
        "scripts/check_docs.py appears in the tree",
        encoding="utf-8",
    )
    return tmp_path


def test_main_passes_when_all_tokens_present(temporary_docs: Path, tmp_path: Path) -> None:
    json_output = tmp_path / "result.json"

    exit_code = check_docs.main(
        ["--root", str(temporary_docs), "--json-output", str(json_output)]
    )

    assert exit_code == 0
    payload = json_output.read_text(encoding="utf-8")
    assert json.loads(payload)["status"] == "pass"


def test_main_fails_when_retriever_reference_missing(temporary_docs: Path, tmp_path: Path) -> None:
    overview = temporary_docs / "docs" / "architecture" / "overview.md"
    overview.write_text(
        "\n".join(
            (
                "scripts/check_local_stack.sh --config",
                "scripts/check_local_stack.sh --up",
                "Use scripts/check_docs.py to keep documentation aligned.",
            )
        ),
        encoding="utf-8",
    )

    exit_code = check_docs.main(
        ["--root", str(temporary_docs), "--json-output", str(tmp_path / "result.json")]
    )

    assert exit_code == 1


def test_main_reports_missing_file(tmp_path: Path) -> None:
    root = tmp_path
    docs_root = root / "docs" / "architecture"
    docs_root.mkdir(parents=True)
    (docs_root / "overview.md").write_text(
        "\n".join(
            (
                "scripts/check_local_stack.sh --config",
                "scripts/check_local_stack.sh --up",
                "QdrantNeo4jRetriever ensures retriever wiring",
                "Use scripts/check_docs.py to keep documentation aligned.",
            )
        ),
        encoding="utf-8",
    )

    exit_code = check_docs.main(
        ["--root", str(root), "--json-output", str(tmp_path / "result.json")]
    )

    assert exit_code == 1
