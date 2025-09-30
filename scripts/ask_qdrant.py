#!/usr/bin/env python
"""Stub retrieval script for Story 2.4 smoke automation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import datetime as _dt


def ensure_env(var: str) -> str:
    """
    Retrieve the value of an environment variable or exit the process if it is not set.
    
    Parameters:
        var (str): Name of the environment variable to read.
    
    Returns:
        value (str): The value of the requested environment variable.
    
    Raises:
        SystemExit: If the environment variable is not set.
    """
    value = os.getenv(var)
    if not value:
        raise SystemExit(f"Missing required environment variable: {var}")
    return value


def main() -> None:
    """
    Run the stub retrieval workflow for asking Qdrant + Neo4j and produce a structured JSON log.
    
    Parses command-line arguments (--question, --top-k), verifies required environment variables (OPENAI_API_KEY, QDRANT_URL, NEO4J_URI), constructs a log entry describing the (stubbed) retrieval, writes the log to artifacts/local_stack/ask_qdrant.json, and prints the same JSON log to stdout.
    """
    parser = argparse.ArgumentParser(description="Ask Qdrant + Neo4j (stub)")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", default="3")
    args = parser.parse_args()

    ensure_env("OPENAI_API_KEY")
    ensure_env("QDRANT_URL")
    ensure_env("NEO4J_URI")

    log = {
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "operation": "ask_qdrant",
        "question": args.question,
        "top_k": int(args.top_k),
        "status": "skipped",
        "message": "Stub implementation – full retrieval delivered in Story 2.5",
        "answer": "This is a placeholder response for local smoke testing.",
    }

    Path("artifacts/local_stack").mkdir(parents=True, exist_ok=True)
    Path("artifacts/local_stack/ask_qdrant.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
