"""Automation entry point for FancyRAG stack lifecycle workflows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from _compat.structlog import get_logger
from cli.sanitizer import sanitize_text, scrub_object
from config.settings import FancyRAGSettings, ServiceSettings
from fancyrag.db.neo4j_queries import rollback_ingest
from fancyrag.qa import QaChunkRecord, QaSourceRecord
from fancyrag.utils.env import get_settings
from fancyrag.utils.paths import ensure_directory, resolve_repo_root

logger = get_logger(__name__)

try:  # pragma: no branch - optional dependency for rollback cleanup
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover - fallback when qdrant client absent
    QdrantClient = None  # type: ignore

from neo4j import GraphDatabase


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    duration_ms: int
    detail: str | None = None


@dataclass
class ServiceRunSummary:
    preset: str
    dataset_path: str | None
    dataset_dir: str | None
    status: str
    stages: list[StageResult]
    artifacts: dict[str, str]
    run_dir: str
    summary_path: str | None = None


@dataclass(frozen=True)
class RunOverrides:
    preset: str | None
    dataset_path: Path | None
    dataset_dir: Path | None
    include_patterns: tuple[str, ...]
    profile: str | None
    semantic_enabled: bool | None
    evaluation_enabled: bool | None
    telemetry: str | None
    skip_teardown: bool
    destroy_volumes: bool
    recreate_collection: bool
    log_root: Path | None
    compose_file: Path | None
    wait_for_health: bool


@dataclass
class RunContext:
    env: dict[str, str]
    settings: FancyRAGSettings
    service: ServiceSettings
    dataset_path: Path | None
    dataset_dir: Path | None
    include_patterns: tuple[str, ...]
    profile: str | None
    semantic_enabled: bool
    evaluation_enabled: bool
    telemetry: str
    run_dir: Path
    vector_index: str
    collection: str
    wait_for_health: bool
    destroy_volumes: bool
    recreate_collection: bool
    stack_started: bool = False


class SkipStage(RuntimeError):
    """Raised to indicate that a stage should be skipped."""


class StageFailure(RuntimeError):
    """Raised when a stage command fails."""

    def __init__(self, message: str, output: str | None = None) -> None:
        sanitized_message = sanitize_text(message)
        super().__init__(sanitized_message)
        sanitized_output = sanitize_text(output) if output is not None else None
        self.detail = sanitized_output if sanitized_output is not None else sanitized_message


class ServiceWorkflow:
    """Coordinate FancyRAG stack automation stages."""

    def __init__(self, repo_root: Path | None = None) -> None:
        resolved_root = resolve_repo_root()
        if repo_root is None:
            repo_root = resolved_root or Path(__file__).resolve().parents[2]
        self.repo_root = repo_root
        self.scripts_dir = self.repo_root / "scripts"
        self.check_script = self.scripts_dir / "check_local_stack.sh"
        self.python = sys.executable
        self.default_compose_file = self.repo_root / "docker-compose.neo4j-qdrant.yml"
        if not self.check_script.exists():
            raise FileNotFoundError(f"Missing automation asset: {self.check_script}")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(self, overrides: RunOverrides) -> ServiceRunSummary:
        settings = get_settings(require={"openai", "neo4j", "qdrant"})
        effective_service = self._merge_service_settings(settings.service, overrides)

        env = os.environ.copy()
        env.update(settings.export_environment())
        env.update(effective_service.export_environment())

        compose_file = overrides.compose_file or self.default_compose_file
        env.setdefault("COMPOSE_FILE", str(compose_file))
        self._ensure_pythonpath(env)

        run_dir = self._prepare_run_directory(overrides.log_root)

        dataset_path = effective_service.resolve_dataset_path(self.repo_root)
        dataset_dir = effective_service.resolve_dataset_dir(self.repo_root)
        include_patterns = overrides.include_patterns or effective_service.include_patterns
        profile = overrides.profile or effective_service.profile
        semantic_enabled = (
            overrides.semantic_enabled if overrides.semantic_enabled is not None else effective_service.semantic_enabled
        )
        evaluation_enabled = (
            overrides.evaluation_enabled if overrides.evaluation_enabled is not None else effective_service.evaluation_enabled
        )
        telemetry = overrides.telemetry or effective_service.telemetry

        context = RunContext(
            env=env,
            settings=settings,
            service=effective_service,
            dataset_path=dataset_path,
            dataset_dir=dataset_dir,
            include_patterns=include_patterns,
            profile=profile,
            semantic_enabled=semantic_enabled,
            evaluation_enabled=evaluation_enabled,
            telemetry=telemetry,
            run_dir=run_dir,
            vector_index=effective_service.vector_index,
            collection=effective_service.collection,
            wait_for_health=overrides.wait_for_health,
            destroy_volumes=overrides.destroy_volumes,
            recreate_collection=overrides.recreate_collection,
        )

        logger.info(
            "service.run.start",
            preset=effective_service.preset,
            dataset_path=str(dataset_path) if dataset_path else None,
            dataset_dir=str(dataset_dir) if dataset_dir else None,
            semantic_enabled=semantic_enabled,
            evaluation_enabled=evaluation_enabled,
            telemetry=telemetry,
        )

        stages: list[StageResult] = []
        artifacts: dict[str, str] = {}

        stage_plan: tuple[tuple[str, Callable[[RunContext], dict[str, str]]], ...] = (
            ("bootstrap", self._stage_bootstrap),
            ("create_vector_index", self._stage_create_vector_index),
            ("ingest", self._stage_ingest),
            ("export", self._stage_export),
            ("evaluation", self._stage_evaluation),
        )

        status = "success"
        for name, handler in stage_plan:
            result, produced = self._run_stage(name, handler, context)
            stages.append(result)
            artifacts.update(produced)
            if result.status == "failed":
                status = "failed"
                break

        try:
            if context.stack_started and not overrides.skip_teardown:
                teardown_result, produced = self._run_stage("teardown", self._stage_teardown, context)
                stages.append(teardown_result)
                artifacts.update(produced)
                if teardown_result.status == "failed":
                    status = "failed"
        finally:
            summary = ServiceRunSummary(
                preset=effective_service.preset,
                dataset_path=str(dataset_path) if dataset_path else None,
                dataset_dir=str(dataset_dir) if dataset_dir else None,
                status=status,
                stages=stages,
                artifacts=artifacts,
                run_dir=str(run_dir),
            )
            summary.summary_path = str(self._write_summary(summary))
            logger.info(
                "service.run.complete",
                status=status,
                summary=summary.summary_path,
                stages=len(stages),
            )
        if status != "success":
            raise StageFailure("Service workflow failed", output=f"See {summary.summary_path}")
        return summary

    def rollback(
        self,
        *,
        log_path: Path | None = None,
        destroy_volumes: bool = False,
        compose_file: Path | None = None,
    ) -> None:
        settings = get_settings(require={"neo4j"})
        env = os.environ.copy()
        env.update(settings.export_environment())
        env.setdefault("COMPOSE_FILE", str((compose_file or self.default_compose_file)))
        self._ensure_pythonpath(env)

        if log_path is None:
            log_path = self._resolve_log_path()
        if log_path is None:
            logger.warning("service.rollback.log_missing", message="kg_build log not found; skipping graph cleanup")
        sources: list[QaSourceRecord] = []
        if log_path is not None:
            sources = self._load_sources_for_rollback(log_path)

        if sources:
            neo4j_settings = settings.neo4j
            with GraphDatabase.driver(neo4j_settings.uri, auth=neo4j_settings.auth()) as driver:
                rollback_ingest(driver, database=neo4j_settings.database, sources=sources)
            logger.info("service.rollback.neo4j", sources=len(sources), log=str(log_path))
        else:
            logger.warning("service.rollback.no_sources", log=str(log_path) if log_path else None)

        if settings.qdrant is not None and QdrantClient is not None:
            try:
                client = QdrantClient(**settings.qdrant.client_kwargs())
                collection = settings.service.collection
                client.delete_collection(collection)
                logger.info("service.rollback.qdrant", collection=collection)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("service.rollback.qdrant_failed", error=str(exc))
        elif settings.qdrant is not None:
            logger.warning("service.rollback.qdrant_client_missing")

        teardown_args = ["--down"]
        if destroy_volumes:
            teardown_args.append("--destroy-volumes")
        try:
            self._invoke_check_script(teardown_args, env)
        except StageFailure as exc:
            logger.warning("service.rollback.teardown_failed", detail=exc.detail)

    # ------------------------------------------------------------------
    # Stage handlers
    # ------------------------------------------------------------------
    def _stage_bootstrap(self, context: RunContext) -> dict[str, str]:
        mount_directories = (
            ".data/neo4j/data",
            ".data/neo4j/logs",
            ".data/neo4j/import",
            ".data/qdrant/storage",
        )
        for relative in mount_directories:
            (self.repo_root / relative).mkdir(parents=True, exist_ok=True)

        self._invoke_check_script(["--up"], context.env)
        context.stack_started = True
        if context.wait_for_health:
            self._invoke_check_script(["--status", "--wait"], context.env)
        return {}

    def _stage_create_vector_index(self, context: RunContext) -> dict[str, str]:
        log_path = context.run_dir / "create_vector_index.json"
        args = [
            str(self.scripts_dir / "create_vector_index.py"),
            "--index-name",
            context.vector_index,
            "--log-path",
            str(log_path),
        ]
        result = self._invoke_python(args, context.env)
        if not log_path.exists():
            default_log = self.repo_root / "artifacts" / "local_stack" / "create_vector_index.json"
            if default_log.exists():
                shutil.copy2(default_log, log_path)
            elif result.stdout:
                log_path.write_text(result.stdout, encoding="utf-8")
        return {"vector_index_log": self._relativize(log_path)}

    def _stage_ingest(self, context: RunContext) -> dict[str, str]:
        if context.dataset_path is None and context.dataset_dir is None:
            raise StageFailure("Dataset path or directory must be provided for ingestion")

        log_path = context.run_dir / "kg_build.json"
        qa_dir = context.run_dir / "qa"

        args = ["-m", "fancyrag.cli.kg_build_main", "--log-path", str(log_path), "--qa-report-dir", str(qa_dir)]

        if context.dataset_path is not None:
            args.extend(["--source", str(context.dataset_path)])
        if context.dataset_dir is not None:
            args.extend(["--source-dir", str(context.dataset_dir)])
        for pattern in context.include_patterns:
            args.extend(["--include-pattern", pattern])
        if context.profile:
            args.extend(["--profile", context.profile])

        args.append("--reset-database")
        if context.semantic_enabled:
            args.append("--enable-semantic")

        ingestion_root = self.repo_root / "artifacts" / "ingestion"
        existing_qa_dirs: set[Path] = set()
        if ingestion_root.exists():
            existing_qa_dirs = {path.resolve() for path in ingestion_root.iterdir() if path.is_dir()}

        result = self._invoke_python_module(args, context.env)
        if not log_path.exists():
            default_log = self.repo_root / "artifacts" / "local_stack" / "kg_build.json"
            if default_log.exists():
                shutil.copy2(default_log, log_path)
            elif result.stdout:
                log_path.write_text(result.stdout, encoding="utf-8")

        qa_dir.mkdir(parents=True, exist_ok=True)
        if ingestion_root.exists():
            for candidate in ingestion_root.iterdir():
                if not candidate.is_dir():
                    continue
                resolved = candidate.resolve()
                if resolved in existing_qa_dirs:
                    continue
                target = qa_dir / candidate.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(candidate, target)

        if result.stdout and not list(qa_dir.iterdir()):
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                payload = {}
            qa_summary = payload.get("qa") if isinstance(payload, dict) else None
            if isinstance(qa_summary, dict):
                qa_dir.mkdir(exist_ok=True, parents=True)
                summary_path = qa_dir / "qa_summary.json"
                summary_path.write_text(json.dumps(qa_summary, indent=2), encoding="utf-8")
        return {
            "kg_log": self._relativize(log_path),
            "qa_dir": self._relativize(qa_dir),
        }

    def _stage_export(self, context: RunContext) -> dict[str, str]:
        args = [
            str(self.scripts_dir / "export_to_qdrant.py"),
            "--collection",
            context.collection,
        ]
        if context.recreate_collection:
            args.append("--recreate-collection")
        self._invoke_python(args, context.env)
        export_log = self.repo_root / "artifacts/local_stack/export_to_qdrant.json"
        return {"export_log": self._relativize(export_log)}

    def _stage_evaluation(self, context: RunContext) -> dict[str, str]:
        if not context.evaluation_enabled:
            raise SkipStage("evaluation disabled")
        log_path = context.run_dir / "check_docs.json"
        args = [
            "-m",
            "scripts.check_docs",
            "--json-output",
            str(log_path),
        ]
        result = self._invoke_python_module(args, context.env)
        if not log_path.exists() and result.stdout:
            log_path.write_text(result.stdout, encoding="utf-8")
        return {"docs_check": self._relativize(log_path)}

    def _stage_teardown(self, context: RunContext) -> dict[str, str]:
        if not context.stack_started:
            raise SkipStage("stack not started")
        args = ["--down"]
        if context.destroy_volumes:
            args.append("--destroy-volumes")
        self._invoke_check_script(args, context.env)
        context.stack_started = False
        return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _merge_service_settings(self, service: ServiceSettings, overrides: RunOverrides) -> ServiceSettings:
        updates: dict[str, object] = {}
        if overrides.preset:
            updates["preset"] = overrides.preset.strip().lower()
        if overrides.dataset_path is not None:
            updates["dataset_path"] = self._format_path_for_service(overrides.dataset_path)
            updates["dataset_dir"] = None
        if overrides.dataset_dir is not None:
            updates["dataset_dir"] = self._format_path_for_service(overrides.dataset_dir)
        if overrides.include_patterns:
            updates["include_patterns"] = overrides.include_patterns
        if overrides.profile:
            updates["profile"] = overrides.profile
        if overrides.semantic_enabled is not None:
            updates["semantic_enabled"] = overrides.semantic_enabled
        if overrides.evaluation_enabled is not None:
            updates["evaluation_enabled"] = overrides.evaluation_enabled
        if overrides.telemetry:
            updates["telemetry"] = overrides.telemetry
        if not updates:
            return service
        return service.model_copy(update=updates)

    def _prepare_run_directory(self, override_root: Path | None) -> Path:
        base = override_root.resolve() if override_root else self.repo_root / "artifacts" / "local_stack" / "service"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = base / timestamp
        suffix = 1
        while run_dir.exists():
            run_dir = base / f"{timestamp}-{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _run_stage(
        self,
        name: str,
        handler: Callable[[RunContext], Mapping[str, str]],
        context: RunContext,
    ) -> tuple[StageResult, dict[str, str]]:
        logger.info("service.stage.start", stage=name)
        start = time.perf_counter()
        try:
            produced = dict(handler(context))
        except SkipStage as skipped:
            duration_ms = int((time.perf_counter() - start) * 1000)
            detail = str(skipped) or None
            logger.info("service.stage.skipped", stage=name, reason=detail)
            return StageResult(name=name, status="skipped", duration_ms=duration_ms, detail=detail), {}
        except StageFailure as failure:
            duration_ms = int((time.perf_counter() - start) * 1000)
            detail = sanitize_text(failure.detail)
            logger.error("service.stage.failed", stage=name, detail=detail)
            return StageResult(name=name, status="failed", duration_ms=duration_ms, detail=detail), {}
        except Exception as exc:  # pragma: no cover - defensive guard
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.exception("service.stage.exception", stage=name)
            detail = str(exc)
            return StageResult(name=name, status="failed", duration_ms=duration_ms, detail=detail), {}
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info("service.stage.complete", stage=name, duration_ms=duration_ms)
        return StageResult(name=name, status="success", duration_ms=duration_ms), produced

    def _invoke_python(self, args: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
        command = [self.python, *args]
        return self._invoke(command, env)

    def _invoke_python_module(self, args: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
        command = [self.python, *args]
        return self._invoke(command, env)

    def _invoke_check_script(self, args: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
        command = [str(self.check_script), *args]
        return self._invoke(command, env)

    def _invoke(self, command: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            cwd=self.repo_root,
            env=dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            output = (result.stdout or "").strip()
            raise StageFailure("Command failed", output=output)
        return result

    def _write_summary(self, summary: ServiceRunSummary) -> Path:
        summary_path = Path(summary.run_dir) / "service_run.json"
        ensure_directory(summary_path)
        payload = scrub_object(asdict(summary))
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return summary_path

    def _ensure_pythonpath(self, env: dict[str, str]) -> None:
        base_entries = ["stubs", "src"]
        existing = env.get("PYTHONPATH")
        if existing:
            base_entries.extend(part for part in existing.split(os.pathsep) if part)
        unique: list[str] = []
        for entry in base_entries:
            if entry and entry not in unique:
                unique.append(entry)
        env["PYTHONPATH"] = os.pathsep.join(unique)

    def _format_path_for_service(self, path: Path) -> str:
        path = path.expanduser()
        if not path.is_absolute():
            return str(path)
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)

    def _relativize(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.repo_root))
        except ValueError:
            return str(path.resolve())

    def _resolve_log_path(self) -> Path | None:
        service_root = self.repo_root / "artifacts" / "local_stack" / "service"
        if not service_root.exists():
            default_log = self.repo_root / "artifacts" / "local_stack" / "kg_build.json"
            return default_log if default_log.exists() else None
        summaries = sorted(service_root.glob("*/service_run.json"))
        if summaries:
            latest = summaries[-1]
            try:
                data = json.loads(latest.read_text(encoding="utf-8"))
                kg_log = data.get("artifacts", {}).get("kg_log")
                if kg_log:
                    candidate = self.repo_root / kg_log
                    if candidate.exists():
                        return candidate
            except json.JSONDecodeError:
                logger.warning("service.rollback.summary_corrupt", path=str(latest))
        default_log = self.repo_root / "artifacts" / "local_stack" / "kg_build.json"
        return default_log if default_log.exists() else None

    def _load_sources_for_rollback(self, log_path: Path) -> list[QaSourceRecord]:
        if not log_path.exists():
            logger.warning("service.rollback.log_not_found", path=str(log_path))
            return []
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("service.rollback.log_invalid", path=str(log_path), error=str(exc))
            return []

        files = payload.get("files") or []
        chunks = payload.get("chunks") or []
        chunk_lookup: dict[str, list[dict[str, object]]] = {}
        for entry in chunks:
            relative = str(entry.get("relative_path", ""))
            chunk_lookup.setdefault(relative, []).append(entry)

        sources: list[QaSourceRecord] = []
        for file_entry in files:
            relative = str(file_entry.get("relative_path", ""))
            chunk_rows = chunk_lookup.get(relative, [])
            qa_chunks = [
                QaChunkRecord(
                    uid=str(row.get("uid")),
                    checksum=str(row.get("checksum", "")),
                    text="",
                )
                for row in chunk_rows
                if row.get("uid") is not None
            ]
            sources.append(
                QaSourceRecord(
                    path=str(file_entry.get("path", "")),
                    relative_path=relative,
                    document_checksum=str(file_entry.get("checksum", "")),
                    git_commit=file_entry.get("git_commit"),
                    chunks=qa_chunks,
                    ingest_run_key=None,
                )
            )
        return sources


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FancyRAG stack automation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute the full service workflow")
    run_parser.add_argument("--preset", help="Override preset selection")
    run_parser.add_argument("--dataset-path", type=Path, help="Override dataset file path")
    run_parser.add_argument("--dataset-dir", type=Path, help="Override dataset directory path")
    run_parser.add_argument(
        "--include-pattern",
        action="append",
        default=None,
        help="Additional glob pattern for directory ingestion",
    )
    run_parser.add_argument("--profile", help="Override chunking profile")
    run_parser.add_argument("--semantic", dest="semantic", action="store_true", help="Enable semantic enrichment")
    run_parser.add_argument("--no-semantic", dest="semantic", action="store_false", help="Disable semantic enrichment")
    run_parser.add_argument("--evaluation", dest="evaluation", action="store_true", help="Enable evaluation stage")
    run_parser.add_argument("--no-evaluation", dest="evaluation", action="store_false", help="Disable evaluation stage")
    run_parser.add_argument("--telemetry", help="Override telemetry preset (e.g. console, otlp)")
    run_parser.add_argument("--skip-teardown", action="store_true", help="Keep stack running after workflow")
    run_parser.add_argument("--destroy-volumes", action="store_true", help="Destroy Docker volumes during teardown")
    run_parser.add_argument("--recreate-collection", action="store_true", help="Recreate Qdrant collection before export")
    run_parser.add_argument("--log-root", type=Path, help="Custom root directory for run artifacts")
    run_parser.add_argument("--compose-file", type=Path, help="Override Docker compose file path")
    run_parser.add_argument("--no-wait", action="store_true", help="Skip waiting for stack health during bootstrap")
    run_parser.set_defaults(semantic=None, evaluation=None)

    rollback_parser = subparsers.add_parser("rollback", help="Rollback ingestion artefacts and stop the stack")
    rollback_parser.add_argument("--log-path", type=Path, help="Path to kg_build log for rollback targeting")
    rollback_parser.add_argument("--destroy-volumes", action="store_true", help="Destroy Docker volumes during teardown")
    rollback_parser.add_argument("--compose-file", type=Path, help="Override Docker compose file path")

    reset_parser = subparsers.add_parser("reset", help="Rollback and destroy Docker volumes")
    reset_parser.add_argument("--log-path", type=Path, help="Path to kg_build log for rollback targeting")
    reset_parser.add_argument("--compose-file", type=Path, help="Override Docker compose file path")

    return parser


def _parse_run_overrides(args: argparse.Namespace) -> RunOverrides:
    include_patterns = tuple(args.include_pattern) if args.include_pattern else tuple()
    return RunOverrides(
        preset=args.preset,
        dataset_path=args.dataset_path,
        dataset_dir=args.dataset_dir,
        include_patterns=include_patterns,
        profile=args.profile,
        semantic_enabled=args.semantic,
        evaluation_enabled=args.evaluation,
        telemetry=args.telemetry,
        skip_teardown=bool(args.skip_teardown),
        destroy_volumes=bool(args.destroy_volumes),
        recreate_collection=bool(args.recreate_collection),
        log_root=args.log_root,
        compose_file=args.compose_file,
        wait_for_health=not bool(args.no_wait),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    workflow = ServiceWorkflow()

    try:
        if args.command == "run":
            overrides = _parse_run_overrides(args)
            workflow.run(overrides)
            return 0
        if args.command == "rollback":
            workflow.rollback(
                log_path=args.log_path,
                destroy_volumes=args.destroy_volumes,
                compose_file=args.compose_file,
            )
            return 0
        if args.command == "reset":
            workflow.rollback(
                log_path=args.log_path,
                destroy_volumes=True,
                compose_file=args.compose_file,
            )
            return 0
    except StageFailure as exc:
        logger.error("service.command_failed", command=args.command, detail=exc.detail)
        return 1
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("service.command_exception", command=args.command)
        return 1
    return 0


__all__ = ["ServiceWorkflow", "main"]
