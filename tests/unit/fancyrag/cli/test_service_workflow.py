from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.settings import ServiceSettings
from fancyrag.cli import service_workflow as workflow_module
from fancyrag.cli.service_workflow import RunOverrides, ServiceWorkflow


@pytest.fixture
def workflow_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    check_script = scripts_dir / "check_local_stack.sh"
    check_script.write_text("#!/bin/sh\n", encoding="utf-8")

    samples_dir = tmp_path / "docs" / "samples"
    samples_dir.mkdir(parents=True)
    (samples_dir / "pilot.txt").write_text("sample", encoding="utf-8")

    service_settings = ServiceSettings(
        preset="smoke",
        dataset_path="docs/samples/pilot.txt",
        dataset_dir=None,
        include_patterns=("**/*.txt",),
        profile="text",
        telemetry="console",
        semantic_enabled=False,
        evaluation_enabled=True,
        vector_index="chunks_vec",
        collection="chunks_main",
    )

    class StubSettings:
        def __init__(self) -> None:
            self.service = service_settings
            self.neo4j = SimpleNamespace(uri="bolt://localhost:7687", database=None)
            self.neo4j.auth = lambda: ("neo4j", "pass")
            self.qdrant = SimpleNamespace(client_kwargs=lambda: {})
            self.openai = SimpleNamespace()

        def export_environment(self) -> dict[str, str]:
            return {
                "OPENAI_API_KEY": "test",
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USERNAME": "neo4j",
                "NEO4J_PASSWORD": "pass",
                "QDRANT_URL": "http://localhost:6333",
            }

    stub_settings = StubSettings()
    call_counter = {"count": 0}

    def fake_get_settings(require: set[str] | frozenset[str]):
        call_counter["count"] += 1
        return stub_settings

    monkeypatch.setattr("fancyrag.cli.service_workflow.get_settings", fake_get_settings)

    calls: list[dict[str, object]] = []

    def fake_invoke(self, command, env):  # noqa: ANN001 - signature mirrors real method
        calls.append({"command": list(command), "env": dict(env)})
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(ServiceWorkflow, "_invoke", fake_invoke, raising=False)

    workflow = ServiceWorkflow(repo_root=tmp_path)

    def make_overrides(**kwargs) -> RunOverrides:
        preset = kwargs.pop("preset", None)
        dataset_path = kwargs.pop("dataset_path", None)
        dataset_dir = kwargs.pop("dataset_dir", None)
        include_patterns = tuple(kwargs.pop("include_patterns", ()))
        profile = kwargs.pop("profile", None)
        semantic_enabled = kwargs.pop("semantic_enabled", None)
        evaluation_enabled = kwargs.pop("evaluation_enabled", None)
        telemetry = kwargs.pop("telemetry", None)
        skip_teardown = kwargs.pop("skip_teardown", False)
        destroy_volumes = kwargs.pop("destroy_volumes", False)
        recreate_collection = kwargs.pop("recreate_collection", False)
        log_root = kwargs.pop("log_root", None)
        compose_file = kwargs.pop("compose_file", None)
        wait_for_health = kwargs.pop("wait_for_health", True)
        if kwargs:
            raise AssertionError(f"Unexpected overrides: {kwargs}")
        return RunOverrides(
            preset=preset,
            dataset_path=dataset_path,
            dataset_dir=dataset_dir,
            include_patterns=include_patterns,
            profile=profile,
            semantic_enabled=semantic_enabled,
            evaluation_enabled=evaluation_enabled,
            telemetry=telemetry,
            skip_teardown=skip_teardown,
            destroy_volumes=destroy_volumes,
            recreate_collection=recreate_collection,
            log_root=log_root,
            compose_file=compose_file,
            wait_for_health=wait_for_health,
        )

    return workflow, calls, call_counter, make_overrides, service_settings, tmp_path


def test_service_rollback_invokes_cleanup(workflow_setup, monkeypatch: pytest.MonkeyPatch):
    workflow, _calls, _counter, _make_overrides, service_settings, repo_root = workflow_setup

    log_path = repo_root / "artifacts" / "local_stack" / "kg_build.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "docs/samples/pilot.txt",
                        "relative_path": "docs/samples/pilot.txt",
                        "checksum": "abc123",
                    }
                ],
                "chunks": [
                    {
                        "relative_path": "docs/samples/pilot.txt",
                        "uid": "chunk-1",
                        "checksum": "def456",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    driver_calls: list[tuple[str, tuple[str, str] | None]] = []

    class DummyDriver:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_driver(uri: str, auth):
        driver_calls.append((uri, auth))
        return DummyDriver()

    monkeypatch.setattr(workflow_module.GraphDatabase, "driver", fake_driver)

    rollback_calls: list[tuple[DummyDriver, str | None, list]] = []

    def fake_rollback(driver, database, sources):
        rollback_calls.append((driver, database, sources))

    monkeypatch.setattr(workflow_module, "rollback_ingest", fake_rollback)

    qdrant_events: list[tuple[str, object]] = []

    class DummyQdrant:
        def __init__(self, **kwargs):
            qdrant_events.append(("init", kwargs))

        def delete_collection(self, name: str) -> None:
            qdrant_events.append(("delete", name))

    monkeypatch.setattr(workflow_module, "QdrantClient", DummyQdrant)

    check_calls: list[list[str]] = []

    def fake_check(self, args, env):  # noqa: ANN001 - signature mirrors real method
        check_calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="")

    monkeypatch.setattr(ServiceWorkflow, "_invoke_check_script", fake_check, raising=False)

    workflow.rollback(log_path=log_path, destroy_volumes=True)

    assert driver_calls == [("bolt://localhost:7687", ("neo4j", "pass"))]
    assert rollback_calls and rollback_calls[0][2]
    assert qdrant_events == [("init", {}), ("delete", service_settings.collection)]
    assert check_calls and check_calls[0] == ["--down", "--destroy-volumes"]


def test_stage_failure_sanitizes_summary(workflow_setup, monkeypatch: pytest.MonkeyPatch):
    workflow, _calls, _counter, make_overrides, _service_settings, repo_root = workflow_setup

    secret = "sk-test-secret-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def failing_stage(self, context):  # noqa: ANN001 - matches bound method signature
        raise workflow_module.StageFailure("bootstrap failed", output=f"OPENAI_API_KEY={secret}")

    monkeypatch.setattr(ServiceWorkflow, "_stage_bootstrap", failing_stage, raising=False)

    with pytest.raises(workflow_module.StageFailure) as excinfo:
        workflow.run(make_overrides())

    detail = excinfo.value.detail
    assert secret not in detail

    service_root = repo_root / "artifacts" / "local_stack" / "service"
    summaries = sorted(service_root.glob("*/service_run.json"))
    assert summaries, "service_run summary was not created"
    summary_data = json.loads(summaries[-1].read_text(encoding="utf-8"))
    stage_detail = summary_data["stages"][0]["detail"]
    assert secret not in stage_detail
    assert "***" in stage_detail


def test_service_run_invokes_stages_in_order(workflow_setup):
    workflow, calls, _, make_overrides, _, _ = workflow_setup
    calls.clear()
    summary = workflow.run(make_overrides())
    stage_names = [stage.name for stage in summary.stages]
    assert stage_names == [
        "bootstrap",
        "create_vector_index",
        "ingest",
        "export",
        "evaluation",
        "teardown",
    ]
    assert summary.status == "success"


def test_service_run_uses_typed_settings_cache(workflow_setup):
    workflow, _, call_counter, make_overrides, _, _ = workflow_setup
    workflow.run(make_overrides())
    assert call_counter["count"] == 1
    workflow.run(make_overrides())
    assert call_counter["count"] == 2


def test_service_run_emits_stage_logs(workflow_setup):
    workflow, _, _, make_overrides, _, _ = workflow_setup
    summary = workflow.run(make_overrides())
    artifacts = summary.artifacts
    assert set(artifacts).issuperset({
        "vector_index_log",
        "kg_log",
        "qa_dir",
        "export_log",
        "docs_check",
    })


def test_service_run_applies_preset_env(workflow_setup):
    workflow, calls, _, make_overrides, service_settings, _ = workflow_setup
    calls.clear()
    workflow.run(make_overrides())
    pipeline_env = next(
        call["env"]
        for call in calls
        if "-m" in call["command"] and "fancyrag.cli.kg_build_main" in call["command"]
    )
    assert pipeline_env["FANCYRAG_PRESET"] == service_settings.preset
    assert pipeline_env["DATASET_PATH"] == service_settings.dataset_path
    assert pipeline_env["FANCYRAG_TELEMETRY"] == service_settings.telemetry


def test_service_run_custom_dataset_path(workflow_setup):
    workflow, calls, _, make_overrides, _, repo_root = workflow_setup
    calls.clear()
    custom_dir = repo_root / "content"
    custom_dir.mkdir(parents=True, exist_ok=True)
    custom_file = custom_dir / "custom.txt"
    custom_file.write_text("data", encoding="utf-8")

    summary = workflow.run(make_overrides(dataset_path=custom_file))
    assert summary.status == "success"

    pipeline_call = next(
        call["command"]
        for call in calls
        if "-m" in call["command"] and "fancyrag.cli.kg_build_main" in call["command"]
    )
    source_index = pipeline_call.index("--source")
    assert pipeline_call[source_index + 1] == str(custom_file.resolve())

    pipeline_env = next(
        call["env"]
        for call in calls
        if "-m" in call["command"] and "fancyrag.cli.kg_build_main" in call["command"]
    )
    expected_env_path = str(custom_file.relative_to(repo_root))
    assert pipeline_env["DATASET_PATH"] == expected_env_path


def test_preset_override_refreshes_defaults(workflow_setup):
    workflow, calls, _, make_overrides, _service_settings, repo_root = workflow_setup
    calls.clear()

    summary = workflow.run(make_overrides(preset="full"))

    pipeline_call = next(
        call["command"]
        for call in calls
        if "-m" in call["command"] and "fancyrag.cli.kg_build_main" in call["command"]
    )
    source_dir_index = pipeline_call.index("--source-dir")
    assert pipeline_call[source_dir_index + 1] == str((repo_root / "docs").resolve())

    pipeline_env = next(
        call["env"]
        for call in calls
        if "-m" in call["command"] and "fancyrag.cli.kg_build_main" in call["command"]
    )
    assert pipeline_env["FANCYRAG_PRESET"] == "full"
    assert "DATASET_PATH" not in pipeline_env
    assert pipeline_env["DATASET_DIR"] == "docs"

    assert summary.dataset_path is None
    assert summary.dataset_dir == str((repo_root / "docs").resolve())
