"""Utility to wait until the Neo4j Bolt endpoint is reachable."""

from __future__ import annotations

import os
import socket
import sys
import time


def _parse_target(uri: str) -> tuple[str, int]:
    host_port = uri.split("://", 1)[-1]
    if ":" not in host_port:
        raise ValueError(f"Bolt URI must include host and port: {uri}")
    host, port_raw = host_port.rsplit(":", 1)
    return host, int(port_raw)


def main() -> int:
    uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
    host, port = _parse_target(uri)

    timeout_seconds = float(os.environ.get("NEO4J_WAIT_TIMEOUT", "120"))
    interval = float(os.environ.get("NEO4J_WAIT_INTERVAL", "2"))
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                return 0
        except OSError:
            time.sleep(interval)

    print(
        f"error: Neo4j Bolt endpoint {uri} unreachable after {timeout_seconds}s",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover - simple CLI utility
    sys.exit(main())
