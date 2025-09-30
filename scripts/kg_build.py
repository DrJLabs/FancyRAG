#!/usr/bin/env python
"""Stub KG build script (Story 2.5 will replace with full pipeline)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import datetime as _dt

from fancyrag.utils import ensure_env


def main() -> None:
    """
    Run a stub knowledge-graph build pipeline that validates environment configuration and emits a JSON log.
    
    Parses an optional `--source` argument (default "docs/samples/pilot.txt"), verifies that the environment variables OPENAI_API_KEY, NEO4J_URI, and QDRANT_URL are present (exits if any are missing), and constructs a log record with a UTC ISO-8601 timestamp, operation "kg_build", the provided source, status "skipped", and a stub message. The log is written to artifacts/local_stack/kg_build.json (creating the directory if needed) and also printed to stdout as JSON.
    """
    parser = argparse.ArgumentParser(description="Run SimpleKGPipeline (stub)")
    parser.add_argument("--source", default="docs/samples/pilot.txt")
    args = parser.parse_args()

    ensure_env("OPENAI_API_KEY")
    ensure_env("NEO4J_URI")
    ensure_env("QDRANT_URL")

    log = {
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "operation": "kg_build",
        "source": args.source,
        "status": "skipped",
        "message": "Stub implementation - full pipeline delivered in Story 2.5",
    }

    Path("artifacts/local_stack").mkdir(parents=True, exist_ok=True)
    Path("artifacts/local_stack/kg_build.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
