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
from typing import Iterable, cast

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC_PATH = os.path.join(_PROJECT_ROOT, "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from config.settings import ALLOWED_CHAT_MODELS  # noqa: E402

CATALOG_URL = "https://api.openai.com/v1/models"

_DATE_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2})(?:-[a-z0-9]+)?$", re.IGNORECASE)
_VARIANT_SUFFIX_PATTERN = re.compile(r"(?i)^(mini(?:-[a-z0-9]+)?|flash|lite|light|pro)$")
_NUMERIC_SUFFIX_PATTERN = re.compile(r"^(\d+)(k)?$", re.IGNORECASE)
_LONG_NAME_THRESHOLD = 40


def _family_of(model: str) -> str:
    """Collapse a model identifier to its allowlist family name."""

    base = _DATE_SUFFIX.sub("", model)
    segments = base.split("-")

    while segments:
        suffix = segments[-1].lower()
        if suffix in {"preview", "latest"}:
            segments.pop()
            continue
        if suffix.startswith("v") and suffix[1:].isdigit() and len(segments) > 2:
            segments.pop()
            continue
        if _NUMERIC_SUFFIX_PATTERN.match(suffix) and len(segments) > 2:
            segments.pop()
            continue
        break

    base = "-".join(segments)
    if not base or "-" not in base:
        return base

    if base in {"gpt-4o"}:
        return base

    head, tail = base.rsplit("-", 1)
    if _VARIANT_SUFFIX_PATTERN.match(tail):
        return head

    return base


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
                "User-Agent": "fancyrag-allowlist-audit",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)

        data = [item for item in payload.get("data", []) if isinstance(item, dict)]
        for item in data:
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                models.add(model_id)

        has_more = bool(payload.get("has_more"))
        if not has_more:
            break

        next_after = cast(str | None, payload.get("last_id"))
        if not next_after and data:
            next_after = cast(str | None, data[-1].get("id"))

        if not next_after or next_after == after:
            break

        after = next_after

    return models


def _format_list(models: Iterable[str]) -> str:
    """
    Format an iterable of model identifiers into a sorted, comma-separated string.
    
    Parameters:
        models: An iterable of model identifier strings.
    
    Returns:
        A comma-separated string of the models sorted lexicographically; returns "<none>" if `models` is empty.
    """
    sorted_models = sorted(models, key=lambda model: (len(model) > _LONG_NAME_THRESHOLD, model))
    return ", ".join(sorted_models) or "<none>"


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
    except json.JSONDecodeError as exc:
        print(f"Malformed JSON from OpenAI API: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Unexpected error contacting OpenAI API: {exc}", file=sys.stderr)
        return 2

    missing = sorted(model for model in ALLOWED_CHAT_MODELS if model not in available_models)
    families = {_family_of(model) for model in ALLOWED_CHAT_MODELS}
    family_roots: set[str] = set()
    family_prefixes: set[str] = set()
    for family in families:
        if not family or "-" not in family:
            continue
        head, _, tail = family.rpartition("-")
        if head and tail.isdigit():
            family_roots.add(head)
        segments = family.split("-")
        if len(segments) >= 2:
            family_prefixes.add("-".join(segments[:2]))
    new_variants = sorted(
        model
        for model in available_models
        if model not in ALLOWED_CHAT_MODELS
        and (
            any(
                model == family
                or model.startswith(f"{family}-")
                or model.startswith(f"{family}.")
                for family in families
            )
            or (
                "-" in model
                and model.rpartition("-")[0] in family_roots
                and model.rpartition("-")[2].isdigit()
            )
            or any(
                prefix and model.startswith(f"{prefix}-")
                for prefix in family_prefixes
            )
        )
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
