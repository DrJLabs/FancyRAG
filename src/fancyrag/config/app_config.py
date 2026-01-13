"""Application configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, ValidationError

from config.utils import parse_boolean


class ConfigurationError(RuntimeError):
    """Raised when application configuration is invalid."""


class Neo4jSettings(BaseModel):
    uri: str
    username: str
    password: str
    database: str


class IndexSettings(BaseModel):
    vector: str = Field(alias="index_name")
    fulltext: str = Field(alias="fulltext_index_name")


class EmbeddingSettings(BaseModel):
    base_url: str
    api_key: str
    model: str = Field(default="text-embedding-3-large")
    timeout_seconds: PositiveFloat = Field(default=10.0)
    max_retries: PositiveInt = Field(default=3)


class OAuthSettings(BaseModel):
    client_id: str
    client_secret: str
    required_scopes: list[str]


class ServerSettings(BaseModel):
    host: str = Field(default="0.0.0.0")  # noqa: S104
    port: int = Field(default=8080, ge=1, le=65535)
    path: str = Field(default="/mcp")
    base_url: str
    auth_required: bool = Field(default=True)


class QuerySettings(BaseModel):
    path: Path
    template: str


class AppConfig(BaseModel):
    neo4j: Neo4jSettings
    indexes: IndexSettings
    embedding: EmbeddingSettings
    oauth: OAuthSettings | None
    server: ServerSettings
    query: QuerySettings


def _require(env: Mapping[str, str], key: str) -> str:
    value = env.get(key)
    if value is None or not value.strip():
        raise ConfigurationError(f"Missing required environment variable: {key}")  # noqa
    return value.strip()


def _optional(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    return value.strip() or None


def _parse_scopes(raw: str) -> list[str]:
    scopes = [scope.strip() for scope in raw.split(",") if scope.strip()]
    if not scopes:
        raise ConfigurationError(
            "GOOGLE_OAUTH_REQUIRED_SCOPES must include at least one scope"
        )  # noqa
    return scopes


def _parse_bool(env: Mapping[str, str], key: str, *, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        return parse_boolean(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be a boolean") from exc


def load_config(env: Mapping[str, str] | None = None) -> AppConfig:
    """Load application configuration from environment variables."""

    if env is None:
        import os

        env_map: Mapping[str, str] = os.environ
    else:
        env_map = env

    neo4j_settings = Neo4jSettings(
        uri=_require(env_map, "NEO4J_URI"),
        username=_require(env_map, "NEO4J_USERNAME"),
        password=_require(env_map, "NEO4J_PASSWORD"),
        database=_require(env_map, "NEO4J_DATABASE"),
    )

    index_settings = IndexSettings(
        index_name=_require(env_map, "INDEX_NAME"),
        fulltext_index_name=_require(env_map, "FULLTEXT_INDEX_NAME"),
    )

    embedding_kwargs: dict[str, Any] = {}
    if (model_raw := _optional(env_map, "EMBEDDING_MODEL")) is not None:
        embedding_kwargs["model"] = model_raw
    if (timeout_raw := _optional(env_map, "EMBEDDING_TIMEOUT_SECONDS")) is not None:
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise ConfigurationError(
                "Invalid numeric value for EMBEDDING_TIMEOUT_SECONDS"
            ) from exc
        if timeout_seconds <= 0:
            raise ConfigurationError("EMBEDDING_TIMEOUT_SECONDS must be greater than 0")
        embedding_kwargs["timeout_seconds"] = timeout_seconds
    if (retries_raw := _optional(env_map, "EMBEDDING_MAX_RETRIES")) is not None:
        try:
            max_retries = int(float(retries_raw))
        except ValueError as exc:
            raise ConfigurationError(
                "Invalid numeric value for EMBEDDING_MAX_RETRIES"
            ) from exc
        if max_retries <= 0:
            raise ConfigurationError("EMBEDDING_MAX_RETRIES must be greater than 0")
        embedding_kwargs["max_retries"] = max_retries

    embedding_settings = EmbeddingSettings(
        base_url=_require(env_map, "EMBEDDING_API_BASE_URL"),
        api_key=_require(env_map, "EMBEDDING_API_KEY"),
        **embedding_kwargs,
    )

    auth_required = _parse_bool(env_map, "MCP_AUTH_REQUIRED", default=True)
    oauth_settings: OAuthSettings | None = None
    if auth_required:
        scopes_raw = env_map.get("GOOGLE_OAUTH_REQUIRED_SCOPES")
        if scopes_raw is None:
            scopes_raw = "openid,https://www.googleapis.com/auth/userinfo.email"
        oauth_settings = OAuthSettings(
            client_id=_require(env_map, "GOOGLE_OAUTH_CLIENT_ID"),
            client_secret=_require(env_map, "GOOGLE_OAUTH_CLIENT_SECRET"),
            required_scopes=_parse_scopes(scopes_raw),
        )

    server_kwargs: dict[str, Any] = {}
    if (host_raw := _optional(env_map, "MCP_SERVER_HOST")) is not None:
        server_kwargs["host"] = host_raw
    if (port_raw := _optional(env_map, "MCP_SERVER_PORT")) is not None:
        try:
            server_kwargs["port"] = int(port_raw)
        except ValueError as exc:
            raise ConfigurationError("MCP_SERVER_PORT must be a valid integer") from exc
    if (path_raw := _optional(env_map, "MCP_SERVER_PATH")) is not None:
        server_kwargs["path"] = path_raw

    server_settings = ServerSettings(
        base_url=_require(env_map, "MCP_BASE_URL"),
        auth_required=auth_required,
        **server_kwargs,
    )

    query_path = (
        Path(_require(env_map, "HYBRID_RETRIEVAL_QUERY_PATH")).expanduser().resolve()
    )
    if not query_path.is_file():
        raise ConfigurationError(
            f"HYBRID_RETRIEVAL_QUERY_PATH does not reference a file: {query_path}"
        )  # noqa

    try:
        query_text = query_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - unlikely, but we report gracefully
        raise ConfigurationError(
            f"Failed to read hybrid query template: {exc}"
        ) from exc  # noqa

    query_settings = QuerySettings(path=query_path, template=query_text)

    try:
        return AppConfig(
            neo4j=neo4j_settings,
            indexes=index_settings,
            embedding=embedding_settings,
            oauth=oauth_settings,
            server=server_settings,
            query=query_settings,
        )
    except ValidationError as exc:  # pragma: no cover - defensive guard
        friendly_overrides = {
            (
                "embedding",
                "timeout_seconds",
            ): "Invalid numeric value for EMBEDDING_TIMEOUT_SECONDS",
            (
                "embedding",
                "max_retries",
            ): "Invalid numeric value for EMBEDDING_MAX_RETRIES",
            ("server", "port"): "MCP_SERVER_PORT must be a valid integer",
        }

        formatted_errors: list[str] = []
        for error in exc.errors():
            location = tuple(error.get("loc", ()))
            message = error.get("msg", "validation error")
            friendly = friendly_overrides.get(location[:2])
            if friendly:
                formatted_errors.append(f"{friendly}: {message}")
                continue
            path = ".".join(str(part) for part in location)
            formatted_errors.append(f"{path or '<root>'}: {message}")

        combined = "\n".join(formatted_errors) or str(exc)
        raise ConfigurationError(f"Invalid configuration:\n{combined}") from exc


__all__ = [
    "AppConfig",
    "ConfigurationError",
    "EmbeddingSettings",
    "IndexSettings",
    "Neo4jSettings",
    "OAuthSettings",
    "QuerySettings",
    "ServerSettings",
    "load_config",
]
