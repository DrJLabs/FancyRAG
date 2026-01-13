from pathlib import Path
import os

import pytest

from fancyrag.config import ConfigurationError, load_config


ANY_INTERFACE = "0.0.0.0"  # noqa: S104 - intentional for test expectations


@pytest.fixture
def query_file(tmp_path: Path) -> Path:
    path = tmp_path / "hybrid.cypher"
    path.write_text("RETURN node, score", encoding="utf-8")
    return path


def _set_required_env(monkeypatch: pytest.MonkeyPatch, query_path: Path) -> None:
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    monkeypatch.setenv("INDEX_NAME", "text_embeddings")
    monkeypatch.setenv("FULLTEXT_INDEX_NAME", "chunk_text_fulltext")
    monkeypatch.setenv("EMBEDDING_API_BASE_URL", "http://localhost:20010/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "dummy")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("MCP_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid,https://www.googleapis.com/auth/userinfo.email")
    monkeypatch.setenv("HYBRID_RETRIEVAL_QUERY_PATH", str(query_path))


def test_load_config_success(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    config = load_config()

    assert config.neo4j.uri == "bolt://localhost:7687"
    assert config.indexes.vector == "text_embeddings"
    assert config.indexes.fulltext == "chunk_text_fulltext"
    assert config.embedding.base_url == "http://localhost:20010/v1"
    assert config.oauth.required_scopes == ["openid", "https://www.googleapis.com/auth/userinfo.email"]
    assert config.query.template.strip() == "RETURN node, score"


def test_missing_variable_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("NEO4J_PASSWORD")

    with pytest.raises(ConfigurationError):
        load_config()


def test_missing_neo4j_uri_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("NEO4J_URI")
    
    with pytest.raises(ConfigurationError, match="NEO4J_URI"):
        load_config()


def test_missing_neo4j_username_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("NEO4J_USERNAME")
    
    with pytest.raises(ConfigurationError, match="NEO4J_USERNAME"):
        load_config()


def test_missing_neo4j_database_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("NEO4J_DATABASE")
    
    with pytest.raises(ConfigurationError, match="NEO4J_DATABASE"):
        load_config()


def test_missing_index_name_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("INDEX_NAME")
    
    with pytest.raises(ConfigurationError, match="INDEX_NAME"):
        load_config()


def test_missing_fulltext_index_name_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("FULLTEXT_INDEX_NAME")
    
    with pytest.raises(ConfigurationError, match="FULLTEXT_INDEX_NAME"):
        load_config()


def test_missing_embedding_api_base_url_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("EMBEDDING_API_BASE_URL")
    
    with pytest.raises(ConfigurationError, match="EMBEDDING_API_BASE_URL"):
        load_config()


def test_missing_embedding_api_key_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("EMBEDDING_API_KEY")
    
    with pytest.raises(ConfigurationError, match="EMBEDDING_API_KEY"):
        load_config()


def test_missing_oauth_client_id_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID")
    
    with pytest.raises(ConfigurationError, match="GOOGLE_OAUTH_CLIENT_ID"):
        load_config()


def test_missing_oauth_client_secret_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_SECRET")

    with pytest.raises(ConfigurationError, match="GOOGLE_OAUTH_CLIENT_SECRET"):
        load_config()


def test_auth_disabled_skips_oauth_requirements(
    monkeypatch: pytest.MonkeyPatch, query_file: Path
) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    monkeypatch.setenv("INDEX_NAME", "text_embeddings")
    monkeypatch.setenv("FULLTEXT_INDEX_NAME", "chunk_text_fulltext")
    monkeypatch.setenv("EMBEDDING_API_BASE_URL", "http://localhost:20010/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "dummy")
    monkeypatch.setenv("MCP_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("HYBRID_RETRIEVAL_QUERY_PATH", str(query_file))
    monkeypatch.setenv("MCP_AUTH_REQUIRED", "false")

    config = load_config()

    assert config.server.auth_required is False
    assert config.oauth is None


def test_auth_required_invalid_value_raises(
    monkeypatch: pytest.MonkeyPatch, query_file: Path
) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("MCP_AUTH_REQUIRED", "maybe")

    with pytest.raises(ConfigurationError, match="MCP_AUTH_REQUIRED"):
        load_config()


def test_missing_mcp_base_url_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("MCP_BASE_URL")
    
    with pytest.raises(ConfigurationError, match="MCP_BASE_URL"):
        load_config()


def test_missing_query_path_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.delenv("HYBRID_RETRIEVAL_QUERY_PATH")
    
    with pytest.raises(ConfigurationError, match="HYBRID_RETRIEVAL_QUERY_PATH"):
        load_config()


def test_empty_string_required_variable_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("NEO4J_URI", "")
    
    with pytest.raises(ConfigurationError, match="NEO4J_URI"):
        load_config()


def test_whitespace_only_required_variable_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("NEO4J_USERNAME", "   ")
    
    with pytest.raises(ConfigurationError, match="NEO4J_USERNAME"):
        load_config()


def test_required_variable_strips_whitespace(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("NEO4J_USERNAME", "  neo4j  ")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid,https://www.googleapis.com/auth/userinfo.email")
    
    config = load_config()
    assert config.neo4j.username == "neo4j"


def test_invalid_timeout_seconds_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "not-a-number")
    
    with pytest.raises(ConfigurationError, match="Invalid numeric value"):
        load_config()


def test_invalid_max_retries_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("EMBEDDING_MAX_RETRIES", "invalid")
    
    with pytest.raises(ConfigurationError, match="Invalid numeric value"):
        load_config()


def test_custom_timeout_seconds(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "15.5")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.embedding.timeout_seconds == 15.5


def test_custom_max_retries(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("EMBEDDING_MAX_RETRIES", "5")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.embedding.max_retries == 5


def test_default_timeout_and_retries(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.embedding.timeout_seconds == 10.0
    assert config.embedding.max_retries == 3


def test_custom_embedding_model(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-model-v2")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.embedding.model == "custom-model-v2"


def test_default_embedding_model(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.embedding.model == "text-embedding-3-large"


def test_oauth_scopes_single_scope(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.oauth.required_scopes == ["openid"]


def test_oauth_scopes_multiple_scopes(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid, email, profile")
    
    config = load_config()
    assert config.oauth.required_scopes == ["openid", "email", "profile"]


def test_oauth_scopes_strips_whitespace(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "  openid  ,  email  ")
    
    config = load_config()
    assert config.oauth.required_scopes == ["openid", "email"]


def test_oauth_scopes_empty_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "")
    
    with pytest.raises(ConfigurationError, match="at least one scope"):
        load_config()


def test_oauth_scopes_only_commas_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", ", , ,")
    
    with pytest.raises(ConfigurationError, match="at least one scope"):
        load_config()


def test_invalid_mcp_server_port_raises(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("MCP_SERVER_PORT", "not-a-port")
    
    with pytest.raises(ConfigurationError, match="valid integer"):
        load_config()


def test_custom_mcp_server_port(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("MCP_SERVER_PORT", "9000")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.server.port == 9000


def test_default_mcp_server_port(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.server.port == 8080


def test_custom_mcp_server_host(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("MCP_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.server.host == "127.0.0.1"


def test_default_mcp_server_host(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.server.host == ANY_INTERFACE


def test_custom_mcp_server_path(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("MCP_SERVER_PATH", "/api/mcp")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.server.path == "/api/mcp"


def test_default_mcp_server_path(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.server.path == "/mcp"


def test_query_path_nonexistent_file_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_env(monkeypatch, tmp_path / "nonexistent.cypher")
    
    with pytest.raises(ConfigurationError, match="does not reference a file"):
        load_config()


def test_query_path_is_directory_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_required_env(monkeypatch, tmp_path)
    
    with pytest.raises(ConfigurationError, match="does not reference a file"):
        load_config()


def test_query_template_content_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    query_file = tmp_path / "complex.cypher"
    query_content = """MATCH (n:Document)
WHERE n.embedding IS NOT NULL
RETURN n, score
ORDER BY score DESC
LIMIT 10"""
    query_file.write_text(query_content, encoding="utf-8")
    
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.query.template == query_content


def test_load_config_with_explicit_env_dict(query_file: Path) -> None:
    env_dict = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password",
        "NEO4J_DATABASE": "neo4j",
        "INDEX_NAME": "text_embeddings",
        "FULLTEXT_INDEX_NAME": "chunk_text_fulltext",
        "EMBEDDING_API_BASE_URL": "http://localhost:20010/v1",
        "EMBEDDING_API_KEY": "dummy",
        "GOOGLE_OAUTH_CLIENT_ID": "client",
        "GOOGLE_OAUTH_CLIENT_SECRET": "secret",
        "GOOGLE_OAUTH_REQUIRED_SCOPES": "openid,email",
        "MCP_BASE_URL": "http://localhost:8080",
        "HYBRID_RETRIEVAL_QUERY_PATH": str(query_file),
    }
    
    config = load_config(env=env_dict)
    
    assert config.neo4j.uri == "bolt://localhost:7687"
    assert config.oauth.required_scopes == ["openid", "email"]


def test_port_boundary_valid_minimum(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("MCP_SERVER_PORT", "1")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.server.port == 1


def test_port_boundary_valid_maximum(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("MCP_SERVER_PORT", "65535")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.server.port == 65535


def test_negative_timeout_raises_validation_error(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "-1.0")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    with pytest.raises(ConfigurationError):
        load_config()


def test_zero_retries_raises_validation_error(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("EMBEDDING_MAX_RETRIES", "0")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    with pytest.raises(ConfigurationError):
        load_config()


def test_negative_retries_raises_validation_error(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("EMBEDDING_MAX_RETRIES", "-1")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    with pytest.raises(ConfigurationError):
        load_config()


def test_float_retries_truncates_to_int(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("EMBEDDING_MAX_RETRIES", "3.7")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.embedding.max_retries == 3


def test_optional_empty_string_returns_none(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "openid")
    
    config = load_config()
    assert config.embedding.model == "text-embedding-3-large"


def test_all_settings_populated_correctly(monkeypatch: pytest.MonkeyPatch, query_file: Path) -> None:
    _set_required_env(monkeypatch, query_file)
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "20.0")
    monkeypatch.setenv("EMBEDDING_MAX_RETRIES", "5")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("MCP_SERVER_HOST", "192.168.1.1")
    monkeypatch.setenv("MCP_SERVER_PORT", "9999")
    monkeypatch.setenv("MCP_SERVER_PATH", "/custom/path")
    monkeypatch.setenv("GOOGLE_OAUTH_REQUIRED_SCOPES", "scope1,scope2,scope3")
    
    config = load_config()
    
    assert config.neo4j.uri == "bolt://localhost:7687"
    assert config.neo4j.username == "neo4j"
    assert config.neo4j.password == os.environ["NEO4J_PASSWORD"]
    assert config.neo4j.database == "neo4j"
    assert config.indexes.vector == "text_embeddings"
    assert config.indexes.fulltext == "chunk_text_fulltext"
    assert config.embedding.base_url == "http://localhost:20010/v1"
    assert config.embedding.api_key == "dummy"
    assert config.embedding.model == "custom-model"
    assert config.embedding.timeout_seconds == 20.0
    assert config.embedding.max_retries == 5
    assert config.oauth.client_id == "client"
    assert config.oauth.client_secret == os.environ["GOOGLE_OAUTH_CLIENT_SECRET"]
    assert config.oauth.required_scopes == ["scope1", "scope2", "scope3"]
    assert config.server.host == "192.168.1.1"
    assert config.server.port == 9999
    assert config.server.path == "/custom/path"
    assert config.server.base_url == "http://localhost:8080"
    assert config.query.template == "RETURN node, score"
    assert config.query.path == query_file
