#!/usr/bin/env python
"""CLI wrapper delegating to fancyrag.cli.kg_build_main."""

from __future__ import annotations

from fancyrag.cli.kg_build_main import main


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
