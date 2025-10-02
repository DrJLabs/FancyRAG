from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.integration
def test_check_docs_cli_passes(tmp_path: Path) -> None:
    """
    Run the check_docs CLI against the project root and assert it reports a passing status.
    
    Executes the scripts.check_docs module and writes JSON output to a temporary file; fails the test if the process exits non-zero and asserts that the JSON payload's "status" field equals "pass".
    """
    json_output = tmp_path / "lint.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_docs",
            "--root",
            str(PROJECT_ROOT),
            "--json-output",
            str(json_output),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if result.returncode != 0:
        pytest.fail(f"check_docs failed: {result.stdout}")

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
