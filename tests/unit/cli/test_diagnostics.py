import importlib
from types import SimpleNamespace

import pytest

from cli import diagnostics
from cli.sanitizer import sanitize_text


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure secrets are cleared between tests."""
    for key in (
        "OPENAI_API_KEY",
        "QDRANT_API_KEY",
        "NEO4J_PASSWORD",
        "NEO4J_BOLT_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)


def _mock_modules(monkeypatch, versions=None):
    versions = versions or {
        "neo4j_graphrag": "0.9.0",
        "neo4j": "5.23.0",
        "qdrant_client": "1.10.4",
        "openai": "1.40.0",
        "structlog": "24.1.0",
    }
    original_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "pytest":
            return original_import(name)
        if name in versions:
            return SimpleNamespace(__version__=versions[name])
        return original_import(name, package=package)

    monkeypatch.setattr(importlib, "import_module", fake_import)


def test_dependency_import_success(tmp_path, monkeypatch):
    """All required modules import and versions are captured."""
    _mock_modules(monkeypatch)
    lock = tmp_path / "requirements.lock"
    lock.write_text("neo4j-graphrag==0.9.1\nneo4j==5.23.1\n", encoding="utf-8")
    report = diagnostics.build_report(tmp_path)
    modules = {pkg["module"] for pkg in report["packages"]}
    assert modules == {module for module, _ in diagnostics.MODULES}
    assert all(pkg["version"] for pkg in report["packages"])


def test_missing_dependency_triggers_error(tmp_path, monkeypatch):
    """Missing module results in non-zero exit code and actionable output."""
    original_import = importlib.import_module

    def broken_import(name, package=None):
        if name == "neo4j_graphrag":
            raise ModuleNotFoundError
        return original_import(name, package=package)

    monkeypatch.setattr(importlib, "import_module", broken_import)
    rc = diagnostics.run_workspace(tmp_path, write=False, output=None)
    assert rc == 1


def test_output_redacts_secret(monkeypatch):
    """Secrets are redacted from diagnostic text."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-456")
    message = sanitize_text("API key sk-test-456 should not leak")
    assert "sk-test-456" not in message
    assert "***" in message


def test_report_contains_expected_metadata(tmp_path, monkeypatch):
    """Report includes lockfile checksum and python metadata."""
    _mock_modules(monkeypatch)
    lock = tmp_path / "requirements.lock"
    lock.write_text("neo4j-graphrag==0.9.0\n", encoding="utf-8")
    report = diagnostics.build_report(tmp_path)
    lock_info = report["lockfile"]
    assert isinstance(lock_info["sha256"], str)
    assert len(lock_info["sha256"]) == 64
    assert report["python"]["version"]
    assert report["git"] == {"sha": None}
