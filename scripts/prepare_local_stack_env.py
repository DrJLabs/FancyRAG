#!/usr/bin/env python
"""Prepare a sanitized environment file for local-stack smoke runs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_EMBEDDING_BASE_URL = "https://api.openai.com/v1"
DEFAULT_NEO4J_PASSWORD = "local-neo4j"
DEFAULT_NEO4J_USERNAME = "neo4j"
EXPORT_ALLOWLIST = {
    "EMBEDDING_API_BASE_URL",
    "EMBEDDING_API_KEY",
    "NEO4J_DATABASE",
    "NEO4J_PASSWORD",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "OPENAI_API_KEY",
    "OPENAI_BACKOFF_SECONDS",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_ENABLE_FALLBACK",
    "OPENAI_MAX_ATTEMPTS",
    "OPENAI_MODEL",
    "QDRANT_API_KEY",
    "QDRANT_URL",
}


def _strip_quotes(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _parse_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, _, remainder = stripped.partition("=")
    key = key.strip()
    value = _strip_quotes(remainder.split("#", 1)[0].strip())
    return key, value


def _sanitize_placeholder(
    line: str,
    *,
    neo4j_username: str,
    neo4j_password: str,
) -> str:
    parsed = _parse_assignment(line)
    if not parsed:
        return line
    key, value = parsed
    if not value.startswith("YOUR_"):
        return line
    if key == "NEO4J_PASSWORD":
        return f"{key}={DEFAULT_NEO4J_PASSWORD}"
    if key == "NEO4J_AUTH":
        return f"{key}={neo4j_username}/{neo4j_password}"
    return f"{key}="


def _write_env(
    *,
    input_path: Path,
    output_path: Path,
    openai_secret: str | None,
    require_openai: bool,
    github_env_path: Path | None,
) -> None:
    rendered_lines: list[str] = []
    export_lines: list[str] = []
    found_openai_key = False
    embedding_base_seen = False
    neo4j_username = DEFAULT_NEO4J_USERNAME
    neo4j_password = DEFAULT_NEO4J_PASSWORD

    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("OPENAI_API_KEY="):
            if openai_secret:
                rendered = f'OPENAI_API_KEY="{openai_secret}"'
                found_openai_key = True
            else:
                rendered = "OPENAI_API_KEY="
        elif line.startswith("OPENAI_MAX_ATTEMPTS="):
            rendered = "OPENAI_MAX_ATTEMPTS=3  # default 3 attempts total"
        elif line.startswith("OPENAI_BACKOFF_SECONDS="):
            rendered = "OPENAI_BACKOFF_SECONDS=0.5  # default 0.5s base delay"
        elif line.startswith("OPENAI_ENABLE_FALLBACK="):
            rendered = "OPENAI_ENABLE_FALLBACK=true  # set to true to enable fallback model selection"
        else:
            rendered = line

        rendered = _sanitize_placeholder(
            rendered,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
        )

        parsed = _parse_assignment(rendered)
        if parsed:
            key, value = parsed
            if key == "NEO4J_USERNAME" and value:
                neo4j_username = value
            if key == "NEO4J_PASSWORD":
                if not value:
                    rendered = f"{key}={DEFAULT_NEO4J_PASSWORD}"
                    key, value = _parse_assignment(rendered) or (
                        key,
                        DEFAULT_NEO4J_PASSWORD,
                    )
                neo4j_password = value
            if key == "NEO4J_AUTH" and not value:
                rendered = f"{key}={neo4j_username}/{neo4j_password}"

        stripped_line = rendered.strip()
        if stripped_line.startswith("EMBEDDING_API_BASE_URL="):
            embedding_base_seen = True
            _, _, remainder = stripped_line.partition("=")
            value = _strip_quotes(remainder.split("#", 1)[0].strip())
            if not value:
                rendered = (
                    f'EMBEDDING_API_BASE_URL="{DEFAULT_EMBEDDING_BASE_URL}"'
                    "  # CI fallback for OpenAI embeddings"
                )

        rendered_lines.append(rendered)

        parsed = _parse_assignment(rendered)
        if parsed:
            key, value = parsed
            if key in EXPORT_ALLOWLIST:
                export_lines.append(f"{key}={value}")

    if not found_openai_key:
        if openai_secret:
            rendered_lines.append(f'OPENAI_API_KEY="{openai_secret}"')
            if "OPENAI_API_KEY" in EXPORT_ALLOWLIST:
                export_lines.append(f"OPENAI_API_KEY={openai_secret}")
        elif require_openai:
            raise RuntimeError(
                "OPENAI_API_KEY is missing; set the environment variable before running."
            )

    if not embedding_base_seen:
        rendered_lines.append(
            f'EMBEDDING_API_BASE_URL="{DEFAULT_EMBEDDING_BASE_URL}"'
            "  # CI fallback for OpenAI embeddings"
        )
        if "EMBEDDING_API_BASE_URL" in EXPORT_ALLOWLIST:
            export_lines.append(f"EMBEDDING_API_BASE_URL={DEFAULT_EMBEDDING_BASE_URL}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rendered_lines) + "\n", encoding="utf-8")

    if openai_secret:
        if not any(
            line.startswith("OPENAI_API_KEY=") and openai_secret in line
            for line in rendered_lines
        ):
            raise RuntimeError("Failed to inject OPENAI_API_KEY into the output env file.")

    if github_env_path:
        github_env_path.parent.mkdir(parents=True, exist_ok=True)
        github_env_path.write_text(
            "\n".join(export_lines) + "\n",
            encoding="utf-8",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a sanitized .env for local-stack smoke runs.",
    )
    parser.add_argument(
        "--input",
        default=".env.example",
        help="Source env template (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=".env",
        help="Output env file (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-missing-openai",
        action="store_true",
        help="Do not fail when OPENAI_API_KEY is missing.",
    )
    parser.add_argument(
        "--write-github-env",
        action="store_true",
        help="Append allowlisted exports to the GITHUB_ENV file when available.",
    )
    parser.add_argument(
        "--github-env",
        default=None,
        help="Override the path used when --write-github-env is set.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1

    openai_secret = os.getenv("OPENAI_API_KEY")
    require_openai = not args.allow_missing_openai

    github_env_path: Path | None = None
    if args.write_github_env:
        if args.github_env:
            github_env_path = Path(args.github_env)
        else:
            github_env = os.getenv("GITHUB_ENV")
            if github_env:
                github_env_path = Path(github_env)

    try:
        _write_env(
            input_path=input_path,
            output_path=output_path,
            openai_secret=openai_secret if openai_secret else None,
            require_openai=require_openai,
            github_env_path=github_env_path,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
