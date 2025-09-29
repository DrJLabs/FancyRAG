#!/usr/bin/env python
"""Stub vector index creation script for local stack smoke tests.

This placeholder ensures Story 2.4 automation can execute end-to-end while
Story 2.5 finishes the true ingestion pipeline. The script validates required
settings and emits structured logs so smoke tests can assert behaviour.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import datetime as _dt


def ensure_env(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise SystemExit(f"Missing required environment variable: {var}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Neo4j vector index (stub)")
    parser.add_argument("--dimensions", default="1536")
    parser.add_argument("--name", default="chunks_vec")
    args = parser.parse_args()

    ensure_env("NEO4J_URI")
    ensure_env("NEO4J_USERNAME")
    ensure_env("NEO4J_PASSWORD")

    log = {
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "operation": "create_vector_index",
        "index_name": args.name,
        "dimensions": int(args.dimensions),
        "status": "skipped",
        "message": "Stub implementation – full pipeline delivered in Story 2.5",
    }

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/local_stack").mkdir(exist_ok=True)
    output = Path("artifacts/local_stack/create_vector_index.json")
    output.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log))


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:  # pragma: no cover - allow friendly error
        raise
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
