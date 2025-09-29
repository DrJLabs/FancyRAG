#!/usr/bin/env python
"""Stub KG build script (Story 2.5 will replace with full pipeline)."""

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
        "message": "Stub implementation – full pipeline delivered in Story 2.5",
    }

    Path("artifacts/local_stack").mkdir(parents=True, exist_ok=True)
    Path("artifacts/local_stack/kg_build.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
