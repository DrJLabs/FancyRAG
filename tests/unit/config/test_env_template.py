from collections import Counter
import re
from pathlib import Path


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise AssertionError("Unable to locate repository root for env template tests.")


REPO_ROOT = _find_repo_root()
ENV_TEMPLATE = REPO_ROOT / ".env.example"
DOC_OVERVIEW = REPO_ROOT / "docs" / "architecture" / "overview.md"

REQUIRED_KEYS = {
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "QDRANT_URL",
    "QDRANT_API_KEY",
}

SAFE_PLACEHOLDER_PREFIX = "YOUR_"
ALLOWED_NON_PLACEHOLDER_VALUES = {
    "gpt-4o-mini",
    "text-embedding-3-small",
}


def _sanitize_value(raw_value: str) -> str:
    return raw_value.split("#", 1)[0].strip().strip('"\'')


def _load_template_entries():
    assert ENV_TEMPLATE.exists(), ".env.example is missing; run bootstrap story tasks."
    entries = []
    for line in ENV_TEMPLATE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        entries.append((key, value))
    return entries


def _load_template_dict():
    return {key: value for key, value in _load_template_entries()}


def test_env_template_contains_required_keys():
    entries = _load_template_entries()
    keys = {key: value for key, value in entries}
    duplicates = {key for key, count in Counter(key for key, _ in entries).items() if count > 1}
    assert REQUIRED_KEYS.issubset(keys.keys()), (
        "Missing required keys in .env.example",
        REQUIRED_KEYS - set(keys.keys()),
    )
    assert not (duplicates & REQUIRED_KEYS), f"Duplicate required keys found in .env.example: {sorted(duplicates & REQUIRED_KEYS)}"
    assert all(keys[key] for key in REQUIRED_KEYS), "All required keys must have placeholder values"


def test_env_template_placeholders_are_safe():
    keys = _load_template_dict()
    for key, value in keys.items():
        sanitized = _sanitize_value(value)
        if sanitized in ALLOWED_NON_PLACEHOLDER_VALUES:
            continue
        assert SAFE_PLACEHOLDER_PREFIX in sanitized, f"{key} should use safe placeholder, found {value}"


def test_env_template_models_document_optional_upgrade():
    contents = ENV_TEMPLATE.read_text()
    model_line = next((line for line in contents.splitlines() if line.startswith("OPENAI_MODEL")), "")
    assert "gpt-4o-mini" in model_line, "Default model should be gpt-4o-mini"
    assert re.search(r"gpt-4\.1-mini", contents), "Template should mention optional gpt-4.1-mini override"


def test_env_template_endpoint_guidance():
    keys = _load_template_dict()
    neo4j_uri = _sanitize_value(keys["NEO4J_URI"])
    qdrant_url = _sanitize_value(keys["QDRANT_URL"])
    assert neo4j_uri.startswith((
        "bolt://",
        "bolt+s://",
        "neo4j://",
        "neo4j+s://",
    )), "NEO4J_URI should reference Bolt or Neo4j scheme"
    assert qdrant_url.startswith("https://"), "QDRANT_URL should reference HTTPS endpoint"


def test_documentation_references_env_template_workflow():
    contents = DOC_OVERVIEW.read_text().lower()
    assert (
        ("copy" in contents or "cp " in contents)
        and ".env.example" in contents
        and ".env" in contents
    ), "Overview must instruct copying env template"
    assert "gpt-4o-mini" in contents and "gpt-4.1-mini" in contents, "Docs should document model defaults"
    assert ("never commit" in contents) or ("do not commit" in contents), "Docs should warn against committing secrets"
