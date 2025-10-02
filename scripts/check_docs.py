#!/usr/bin/env python
"""Validate architecture documentation references for the minimal-path workflow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

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
        checks=(
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

    def to_dict(self) -> dict[str, object]:
        """
        Serialize the lint issue to a dictionary suitable for JSON output.

        Returns:
            dict[str, object]: A mapping with keys:
                - "path": filesystem path to the document as a string.
                - "check_id": identifier of the failed check.
                - "description": human-readable description of the issue.
                - "missing_tokens": list of tokens that were not found.
        """
        return {
            "path": str(self.path),
            "check_id": self.check_id,
            "description": self.description,
            "missing_tokens": list(self.missing_tokens),
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the documentation lint script.
    
    Parameters:
        argv (Sequence[str] | None): Command-line arguments to parse. If None, the system arguments are used.
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - root (Path): Project root containing the docs directory.
            - json_output (Path): Path to write sanitized JSON results.
    """

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
    return parser.parse_args(argv)


def _load_text(path: Path) -> str:
    """Load text from a UTF-8 encoded file."""

    return path.read_text(encoding="utf-8")


def _evaluate_rule(rule: LintRule, *, root: Path) -> tuple[list[dict[str, str]], list[LintIssue]]:
    """
    Check a LintRule against the repository root and collect passing evidence and any lint issues found.
    
    Parameters:
        rule (LintRule): Rule containing a relative documentation path and its TokenChecks.
        root (Path): Project root used to resolve the rule's relative_path.
    
    Returns:
        tuple[list[dict[str, str]], list[LintIssue]]: 
            - evidence: list of dictionaries for checks that passed; each dictionary contains keys `path`, `check_id`, `status` (set to `"passed"`), and `description`.
            - issues: list of LintIssue objects for failed checks or a missing file. If the documentation file is absent a single LintIssue is returned with `check_id` `"missing-file"` and an empty `missing_tokens`.
    """

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


def _gather_results(
    rules: Iterable[LintRule], *, root: Path
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """
    Run lint rules across the provided rules and collect evidence and issues.
    
    Parameters:
        root (Path): Filesystem root used to resolve each rule's relative_path.
    
    Returns:
        tuple:
            - evidence (list[dict[str, str]]): Passed-check records with keys `path`, `check_id`, `status`, and `description`.
            - issues (list[dict[str, object]]): Failure records produced by `LintIssue.to_dict()` with keys `path`, `check_id`, `description`, and `missing_tokens` (list of tokens).
    """

    evidence: list[dict[str, str]] = []
    issues: list[dict[str, object]] = []

    for rule in rules:
        rule_evidence, rule_issues = _evaluate_rule(rule, root=root)
        evidence.extend(rule_evidence)
        issues.extend(issue.to_dict() for issue in rule_issues)

    return evidence, issues


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """
    Sanitize the given payload and write it as pretty-printed JSON to the specified file path, creating parent directories as needed.
    
    Parameters:
        path (Path): Destination file path where the JSON will be written.
        payload (dict[str, object]): Data to be sanitized and serialized to JSON.
    """

    sanitized = scrub_object(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitized, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the documentation lint checks and emit results.
    
    Runs the configured lint rules against documentation files under the given project root, writes a sanitized JSON summary to the configured output path, and logs the final result.
    
    Parameters:
        argv (Sequence[str] | None): Command-line arguments to parse; defaults to sys.argv when None.
    
    Returns:
        int: Exit code: 0 if all checks passed, 1 if any check failed.
    """

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
