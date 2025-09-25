from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
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


def _load_template_lines():
    if not ENV_TEMPLATE.exists():
        raise AssertionError(".env.example is missing; run bootstrap story tasks.")
    return [line.strip() for line in ENV_TEMPLATE.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def test_env_template_contains_required_keys():
    keys = {}
    for line in _load_template_lines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        keys[key] = value
    assert REQUIRED_KEYS.issubset(keys.keys()), (
        "Missing required keys in .env.example",
        REQUIRED_KEYS - set(keys.keys()),
    )
    assert all(keys[key] for key in REQUIRED_KEYS), "All required keys must have placeholder values"


def test_env_template_placeholders_are_safe():
    keys = {line.split("=", 1)[0]: line.split("=", 1)[1] for line in _load_template_lines() if "=" in line}
    for key, value in keys.items():
        sanitized = value.split("#", 1)[0].strip()
        if sanitized in ALLOWED_NON_PLACEHOLDER_VALUES:
            continue
        assert SAFE_PLACEHOLDER_PREFIX in sanitized, f"{key} should use safe placeholder, found {value}"


def test_env_template_models_document_optional_upgrade():
    lines = ENV_TEMPLATE.read_text().splitlines()
    model_line = next((line for line in lines if line.startswith("OPENAI_MODEL")), "")
    assert "gpt-4o-mini" in model_line, "Default model should be gpt-4o-mini"
    assert "gpt-4.1-mini" in model_line, "Comment should mention optional gpt-4.1-mini override"


def test_env_template_endpoint_guidance():
    keys = {line.split("=", 1)[0]: line.split("=", 1)[1] for line in _load_template_lines() if "=" in line}
    assert keys["NEO4J_URI"].startswith("bolt://"), "NEO4J_URI should reference Bolt scheme"
    assert keys["QDRANT_URL"].startswith("https://"), "QDRANT_URL should reference HTTPS endpoint"


def test_documentation_references_env_template_workflow():
    contents = DOC_OVERVIEW.read_text().lower()
    assert "copy `.env.example` to `.env`" in contents, "Overview must instruct copying env template"
    assert "gpt-4o-mini" in contents and "gpt-4.1-mini" in contents, "Docs should document model defaults"
    assert "never commit" in contents, "Docs should warn against committing secrets"
