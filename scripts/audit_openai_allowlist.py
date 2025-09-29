#!/usr/bin/env python3
"""Verify the configured OpenAI chat-model allowlist against the live catalog."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable

from config.settings import ALLOWED_CHAT_MODELS

CATALOG_URL = "https://api.openai.com/v1/models"

_DATE_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2})(?:-[a-z0-9]+)?$", re.IGNORECASE)


def _family_of(model: str) -> str:
    """Collapse a model identifier to its allowlist family name."""

    base = _DATE_SUFFIX.sub("", model)
    if "-" not in base:
        return base

    head, _tail = base.rsplit("-", 1)
    return head


def _fetch_models(api_key: str) -> set[str]:
    """
    Fetch the OpenAI model catalog and extract model identifiers.
    
    Parameters:
        api_key (str): OpenAI API key used for Authorization header.
    
    Returns:
        set[str]: A set of model ID strings extracted from the catalog's `data` array. The function
        follows pagination using the `after` cursor, ignores non-dictionary entries, and discards
        any catalog entries that lack an `id` value.
    """
    models: set[str] = set()
    after: str | None = None
    while True:
        query: dict[str, str] = {"limit": "100"}
        if after:
            query["after"] = after
        request = urllib.request.Request(
            f"{CATALOG_URL}?{urllib.parse.urlencode(query)}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "graphrag-allowlist-audit",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)

        data = [item for item in payload.get("data", []) if isinstance(item, dict)]
        models.update(item["id"] for item in data if item.get("id"))

        if not payload.get("has_more") or not data:
            break

        after = data[-1].get("id")
        if not after:
            break

    return models


def _format_list(models: Iterable[str]) -> str:
    """
    Format an iterable of model identifiers into a sorted, comma-separated string.
    
    Parameters:
        models: An iterable of model identifier strings.
    
    Returns:
        A comma-separated string of the models sorted lexicographically; returns "<none>" if `models` is empty.
    """
    return ", ".join(sorted(models)) or "<none>"


def main() -> int:
    """
    Validate the configured OpenAI chat-model allowlist against the live model catalog.
    
    Fetches the model catalog using OPENAI_API_KEY, compares it to ALLOWED_CHAT_MODELS, and prints outcome messages to stdout or stderr. The function does not raise; it returns an integer exit code describing the result.
    
    Returns:
        int: Exit code indicating the validation result:
            0: Allowlist verified with no issues.
            1: OPENAI_API_KEY environment variable is missing.
            2: Failed to contact or fetch the model catalog (HTTP/URL error).
            3: One or more allowlisted models are missing from the catalog.
            4: New variants of allowlisted model families were detected in the catalog.
    """
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
    families = {_family_of(model) for model in ALLOWED_CHAT_MODELS}
    new_variants = sorted(
        model
        for model in available_models
        if model not in ALLOWED_CHAT_MODELS
        and any(model == family or model.startswith(f"{family}-") for family in families)
    )

    if missing:
        print(
            "Allowlist validation failed. Missing models: " + _format_list(missing),
            file=sys.stderr,
        )
        return 3

    if new_variants:
        print(
            "Allowlist validation succeeded but new model variants were detected: " + _format_list(new_variants),
            file=sys.stderr,
        )
        return 4

    print("Allowlist verified: " + _format_list(ALLOWED_CHAT_MODELS))
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())
