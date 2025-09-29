#!/usr/bin/env python3
"""Verify the configured OpenAI chat-model allowlist against the live catalog."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Iterable

from config.settings import ALLOWED_CHAT_MODELS

CATALOG_URL = "https://api.openai.com/v1/models"


def _fetch_models(api_key: str) -> set[str]:
    request = urllib.request.Request(
        CATALOG_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "graphrag-allowlist-audit",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return {item.get("id", "") for item in payload.get("data", []) if isinstance(item, dict)}


def _format_list(models: Iterable[str]) -> str:
    return ", ".join(sorted(models)) or "<none>"


def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required to audit the OpenAI model catalog.", file=sys.stderr)
        return 1

    try:
        available_models = _fetch_models(api_key)
    except urllib.error.HTTPError as exc:  # pragma: no cover - exercised in CI
        print(f"Failed to fetch model catalog (HTTP {exc.code}): {exc.reason}", file=sys.stderr)
        return 2
    except urllib.error.URLError as exc:  # pragma: no cover - exercised in CI
        print(f"Failed to reach OpenAI API: {exc.reason}", file=sys.stderr)
        return 2

    missing = sorted(model for model in ALLOWED_CHAT_MODELS if model not in available_models)
    new_variants = sorted(
        model for model in available_models if model.startswith("gpt-4.1") and model not in ALLOWED_CHAT_MODELS
    )

    if missing:
        print(
            "Allowlist validation failed. Missing models: " + _format_list(missing),
            file=sys.stderr,
        )
        return 3

    if new_variants:
        print(
            "Allowlist validation succeeded but new GPT-4.1 variants were detected: " + _format_list(new_variants),
            file=sys.stderr,
        )
        return 4

    print("Allowlist verified: " + _format_list(ALLOWED_CHAT_MODELS))
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())
