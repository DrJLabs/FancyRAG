#!/usr/bin/env python
"""Validate architecture documentation references for the minimal-path workflow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from _compat.structlog import get_logger
from cli.sanitizer import scrub_object


logger = get_logger(__name__)

DEFAULT_JSON_PATH = Path("artifacts/docs/check_docs.json")


@dataclass(frozen=True)
class TokenCheck:
    """Represents a documentation lint check that ensures required tokens are present."""

    check_id: str
    tokens: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class LintRule:
    """Represents the set of token checks that must pass for a single documentation file."""

    relative_path: Path
    checks: tuple[TokenCheck, ...]


DEFAULT_RULES: tuple[LintRule, ...] = (
    LintRule(
        relative_path=Path("docs/architecture/overview.md"),
        checks=
        (
            TokenCheck(
                check_id="minimal-path-commands",
                tokens=("scripts/check_local_stack.sh --config", "scripts/check_local_stack.sh --up"),
                description="Minimal-path compose lifecycle commands are documented.",
            ),
            TokenCheck(
                check_id="retriever-reference",
                tokens=("QdrantNeo4jRetriever",),
                description="Overview references the native Qdrant retriever.",
            ),
            TokenCheck(
                check_id="docs-lint-workflow",
                tokens=("scripts/check_docs.py",),
                description="Overview instructs operators to run the documentation lint guard.",
            ),
        ),
    ),
    LintRule(
        relative_path=Path("docs/architecture/source-tree.md"),
        checks=(
            TokenCheck(
                check_id="source-tree-entry",
                tokens=("scripts/check_docs.py",),
                description="Source tree lists the documentation lint script.",
            ),
        ),
    ),
)


@dataclass
class LintIssue:
    """Represents a lint failure for a specific file/check."""

    path: Path
    check_id: str
    missing_tokens: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "check_id": self.check_id,
            "description": self.description,
            "missing_tokens": ", ".join(self.missing_tokens),
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__ or "Documentation lint guard")
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Project root that contains the docs directory (default: repository root)",
    )
    parser.add_argument(
        "--json-output",
        default=DEFAULT_JSON_PATH,
        type=Path,
        help="Path to write sanitized JSON results (default: artifacts/docs/check_docs.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Retained for compatibility; lint always fails on missing tokens.",
    )
    return parser.parse_args(argv)


def _load_text(path: Path) -> str:
    """Load text from a UTF-8 encoded file."""

    return path.read_text(encoding="utf-8")


def _evaluate_rule(rule: LintRule, *, root: Path) -> tuple[list[dict[str, str]], list[LintIssue]]:
    """Evaluate a lint rule and return collected evidence."""

    absolute_path = root / rule.relative_path
    evidence: list[dict[str, str]] = []
    issues: list[LintIssue] = []

    if not absolute_path.exists():
        issues.append(
            LintIssue(
                path=rule.relative_path,
                check_id="missing-file",
                missing_tokens=tuple(),
                description="Documentation file is missing.",
            )
        )
        return evidence, issues

    content = _load_text(absolute_path)

    for check in rule.checks:
        missing = tuple(token for token in check.tokens if token not in content)
        if missing:
            issues.append(
                LintIssue(
                    path=rule.relative_path,
                    check_id=check.check_id,
                    missing_tokens=missing,
                    description=check.description,
                )
            )
        else:
            evidence.append(
                {
                    "path": str(rule.relative_path),
                    "check_id": check.check_id,
                    "status": "passed",
                    "description": check.description,
                }
            )

    return evidence, issues


def _gather_results(rules: Iterable[LintRule], *, root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Run lint rules and return (evidence, issues)."""

    evidence: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []

    for rule in rules:
        rule_evidence, rule_issues = _evaluate_rule(rule, root=root)
        evidence.extend(rule_evidence)
        issues.extend(issue.to_dict() for issue in rule_issues)

    return evidence, issues


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write sanitized JSON payload to disk."""

    sanitized = scrub_object(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitized, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for command-line execution."""

    args = _parse_args(argv)
    root = args.root.resolve()

    evidence, issues = _gather_results(DEFAULT_RULES, root=root)

    status = "pass" if not issues else "fail"
    payload = {
        "operation": "documentation_lint",
        "status": status,
        "root": str(root),
        "issues": issues,
        "evidence": evidence,
    }

    log_method = logger.info if status == "pass" else logger.error
    log_method("documentation_lint_completed", **scrub_object(payload))

    try:
        _write_json(args.json_output, payload)
    except OSError:
        logger.warning("documentation_lint_write_failed", json_output=str(args.json_output))

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
