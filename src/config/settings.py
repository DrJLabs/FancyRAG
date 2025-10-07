"""Typed configuration surfaces for FancyRAG tooling."""

from __future__ import annotations

import getpass
import os
from typing import Any, ClassVar, Mapping, Optional
from urllib.parse import urlparse

from _compat.structlog import get_logger
from cli.sanitizer import mask_base_url
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

logger = get_logger(__name__)

DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
FALLBACK_CHAT_MODELS: frozenset[str] = frozenset({"gpt-4o-mini"})
ALLOWED_CHAT_MODELS: frozenset[str] = frozenset({DEFAULT_CHAT_MODEL, *FALLBACK_CHAT_MODELS})

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_MAX_RETRY_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 0.5
DEFAULT_FALLBACK_ENABLED = True

_ENV_OPENAI_MODEL = "OPENAI_MODEL"
_ENV_OPENAI_EMBEDDING_MODEL = "OPENAI_EMBEDDING_MODEL"
_ENV_OPENAI_EMBEDDING_DIMENSIONS = "OPENAI_EMBEDDING_DIMENSIONS"
_ENV_ACTOR_HINT = "GRAPH_RAG_ACTOR"
_ENV_OPENAI_MAX_ATTEMPTS = "OPENAI_MAX_ATTEMPTS"
_ENV_OPENAI_BACKOFF_SECONDS = "OPENAI_BACKOFF_SECONDS"
_ENV_OPENAI_ENABLE_FALLBACK = "OPENAI_ENABLE_FALLBACK"
_ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
_ENV_OPENAI_ALLOW_INSECURE_BASE_URL = "OPENAI_ALLOW_INSECURE_BASE_URL"

_VALID_BASE_SCHEMES = frozenset({"http", "https"})
_TRUE_BOOL_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_BOOL_VALUES = frozenset({"0", "false", "no", "off"})
_ALLOWED_NEO4J_SCHEMES = frozenset({"bolt", "bolt+s", "neo4j", "neo4j+s"})
_ALLOWED_QDRANT_SCHEMES = frozenset({"http", "https"})


def _parse_boolean_flag(*, raw_value: str, env_key: str, actor_name: str, error_event: str) -> bool:
    normalized = raw_value.lower()
    if normalized in _TRUE_BOOL_VALUES:
        return True
    if normalized in _FALSE_BOOL_VALUES:
        return False
    allowed_values = ", ".join(sorted(_TRUE_BOOL_VALUES | _FALSE_BOOL_VALUES))
    logger.error(
        error_event,
        actor=actor_name,
        supplied=raw_value,
        allowed_values=allowed_values,
    )
    raise ValueError(
        f"Invalid {env_key} value '{raw_value}'. Use one of: {allowed_values}."
    )


class OpenAISettings(BaseModel):
    """Resolved OpenAI settings with guardrails and audit metadata."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr | None = None
    chat_model: str = Field(default=DEFAULT_CHAT_MODEL)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    embedding_dimensions: int = Field(default=DEFAULT_EMBEDDING_DIMENSIONS, gt=0)
    embedding_dimensions_override: int | None = Field(default=None, gt=0)
    actor: str
    max_attempts: int = Field(default=DEFAULT_MAX_RETRY_ATTEMPTS, gt=0)
    backoff_seconds: float = Field(default=DEFAULT_BACKOFF_SECONDS, gt=0)
    enable_fallback: bool = Field(default=DEFAULT_FALLBACK_ENABLED)
    api_base_url: str | None = None
    allow_insecure_base_url: bool = False

    @property
    def is_chat_override(self) -> bool:
        """Return True when a non-default chat model is configured."""

        return self.chat_model != DEFAULT_CHAT_MODEL

    @property
    def allowed_chat_models(self) -> frozenset[str]:
        """Expose the valid chat models for guardrails and telemetry."""

        return ALLOWED_CHAT_MODELS

    def expected_embedding_dimensions(self) -> int:
        """Return the embedding dimensionality accounting for overrides."""

        return self.embedding_dimensions_override or self.embedding_dimensions

    def for_actor(self, actor: str) -> OpenAISettings:
        """Return a copy of the settings with the provided actor metadata."""

        return self.model_copy(update={"actor": actor})

    @classmethod
    def load(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        actor: Optional[str] = None,
    ) -> OpenAISettings:
        """Load OpenAI configuration from environment values with validation."""

        source = env or os.environ
        actor_name = actor or source.get(_ENV_ACTOR_HINT) or getpass.getuser() or "unknown"

        requested_chat_model = (source.get(_ENV_OPENAI_MODEL) or "").strip() or DEFAULT_CHAT_MODEL

        if requested_chat_model not in ALLOWED_CHAT_MODELS:
            logger.error(
                "openai.chat.invalid_model",
                actor=actor_name,
                supplied_model=requested_chat_model,
                allowed_models=sorted(ALLOWED_CHAT_MODELS),
            )
            raise ValueError(
                f"Unsupported OpenAI chat model '{requested_chat_model}'. Set {_ENV_OPENAI_MODEL} to one of {sorted(ALLOWED_CHAT_MODELS)}."
            )

        if requested_chat_model != DEFAULT_CHAT_MODEL:
            logger.info(
                "openai.chat.override",
                actor=actor_name,
                model=requested_chat_model,
                default_model=DEFAULT_CHAT_MODEL,
            )

        embedding_model = (source.get(_ENV_OPENAI_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL).strip()
        override_raw = (source.get(_ENV_OPENAI_EMBEDDING_DIMENSIONS) or "").strip()
        override_dimensions: int | None = None
        if override_raw:
            try:
                override_dimensions = int(override_raw)
            except ValueError as exc:  # pragma: no cover - ValueError path tested
                logger.error(
                    "openai.embedding.invalid_override",
                    actor=actor_name,
                    override_value=override_raw,
                )
                raise ValueError(
                    f"Invalid {_ENV_OPENAI_EMBEDDING_DIMENSIONS} value '{override_raw}'. Provide a positive integer."
                ) from exc
            if override_dimensions <= 0:
                logger.error(
                    "openai.embedding.non_positive_override",
                    actor=actor_name,
                    override_value=override_raw,
                )
                raise ValueError(
                    f"{_ENV_OPENAI_EMBEDDING_DIMENSIONS} must be a positive integer when set; received {override_dimensions}."
                )
            logger.info(
                "openai.embedding.override",
                actor=actor_name,
                override_dimensions=override_dimensions,
                default_dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            )

        max_attempts_raw = (source.get(_ENV_OPENAI_MAX_ATTEMPTS) or "").strip()
        max_attempts = DEFAULT_MAX_RETRY_ATTEMPTS
        if max_attempts_raw:
            try:
                max_attempts = int(max_attempts_raw)
            except ValueError as exc:  # pragma: no cover - invalid atoi path
                logger.error(
                    "openai.settings.invalid_max_attempts",
                    actor=actor_name,
                    supplied=max_attempts_raw,
                )
                raise ValueError(
                    f"Invalid {_ENV_OPENAI_MAX_ATTEMPTS} value '{max_attempts_raw}'. Provide a positive integer."
                ) from exc
            if max_attempts <= 0:
                logger.error(
                    "openai.settings.non_positive_max_attempts",
                    actor=actor_name,
                    supplied=max_attempts_raw,
                )
                raise ValueError(
                    f"{_ENV_OPENAI_MAX_ATTEMPTS} must be a positive integer when set; received {max_attempts}."
                )
            logger.info(
                "openai.settings.max_attempts_override",
                actor=actor_name,
                max_attempts=max_attempts,
                default_attempts=DEFAULT_MAX_RETRY_ATTEMPTS,
            )

        backoff_raw = (source.get(_ENV_OPENAI_BACKOFF_SECONDS) or "").strip()
        backoff_seconds = DEFAULT_BACKOFF_SECONDS
        if backoff_raw:
            try:
                backoff_seconds = float(backoff_raw)
            except ValueError as exc:  # pragma: no cover - invalid atof path
                logger.error(
                    "openai.settings.invalid_backoff",
                    actor=actor_name,
                    supplied=backoff_raw,
                )
                raise ValueError(
                    f"Invalid {_ENV_OPENAI_BACKOFF_SECONDS} value '{backoff_raw}'. Provide a positive number."
                ) from exc
            if backoff_seconds <= 0:
                logger.error(
                    "openai.settings.non_positive_backoff",
                    actor=actor_name,
                    supplied=backoff_raw,
                )
                raise ValueError(
                    f"{_ENV_OPENAI_BACKOFF_SECONDS} must be greater than zero when set; received {backoff_seconds}."
                )
            logger.info(
                "openai.settings.backoff_override",
                actor=actor_name,
                backoff_seconds=backoff_seconds,
                default_seconds=DEFAULT_BACKOFF_SECONDS,
            )

        fallback_raw = (source.get(_ENV_OPENAI_ENABLE_FALLBACK) or "").strip()
        enable_fallback = DEFAULT_FALLBACK_ENABLED
        if fallback_raw:
            enable_fallback = _parse_boolean_flag(
                raw_value=fallback_raw,
                env_key=_ENV_OPENAI_ENABLE_FALLBACK,
                actor_name=actor_name,
                error_event="openai.settings.invalid_fallback",
            )
            logger.info(
                "openai.settings.fallback_toggle",
                actor=actor_name,
                enable_fallback=enable_fallback,
            )

        if not enable_fallback and requested_chat_model != DEFAULT_CHAT_MODEL:
            logger.error(
                "openai.settings.fallback_disabled",
                actor=actor_name,
                supplied_model=requested_chat_model,
            )
            raise ValueError(
                f"{_ENV_OPENAI_ENABLE_FALLBACK} is disabled; {requested_chat_model} cannot be used as it is reserved for fallback scenarios."
            )

        allow_insecure_raw = (source.get(_ENV_OPENAI_ALLOW_INSECURE_BASE_URL) or "").strip()
        allow_insecure = False
        if allow_insecure_raw:
            allow_insecure = _parse_boolean_flag(
                raw_value=allow_insecure_raw,
                env_key=_ENV_OPENAI_ALLOW_INSECURE_BASE_URL,
                actor_name=actor_name,
                error_event="openai.settings.invalid_insecure_flag",
            )
            if allow_insecure:
                logger.warning(
                    "openai.settings.insecure_flag_enabled",
                    actor=actor_name,
                )
            else:
                logger.info(
                    "openai.settings.insecure_flag_disabled",
                    actor=actor_name,
                )

        base_url_raw = (source.get(_ENV_OPENAI_BASE_URL) or "").strip()
        api_base_url: Optional[str] = None
        if base_url_raw:
            parsed = urlparse(base_url_raw)
            if parsed.scheme not in _VALID_BASE_SCHEMES or not parsed.netloc:
                logger.error(
                    "openai.settings.invalid_base_url",
                    actor=actor_name,
                    supplied=mask_base_url(base_url_raw),
                )
                raise ValueError(
                    f"Invalid {_ENV_OPENAI_BASE_URL} value '{base_url_raw}'. Provide an absolute http(s) URL."
                )
            if parsed.scheme != "https":
                if not allow_insecure:
                    logger.error(
                        "openai.settings.insecure_base_url",
                        actor=actor_name,
                        supplied=mask_base_url(base_url_raw),
                    )
                    raise ValueError(
                        f"{_ENV_OPENAI_BASE_URL} must use https. Set {_ENV_OPENAI_ALLOW_INSECURE_BASE_URL}=true only for explicit testing scenarios."
                    )
                logger.warning(
                    "openai.settings.insecure_base_url_override",
                    actor=actor_name,
                    base_url=mask_base_url(base_url_raw),
                )
            api_base_url = base_url_raw
            logger.info(
                "openai.settings.base_url_override",
                actor=actor_name,
                base_url=mask_base_url(api_base_url),
            )

        return cls(
            api_key=None,
            chat_model=requested_chat_model,
            embedding_model=embedding_model,
            embedding_dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            embedding_dimensions_override=override_dimensions,
            actor=actor_name,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            enable_fallback=enable_fallback,
            api_base_url=api_base_url,
            allow_insecure_base_url=allow_insecure,
        )


class Neo4jSettings(BaseModel):
    """Typed Neo4j connection settings."""

    model_config = ConfigDict(frozen=True)

    uri: str
    username: str
    password: SecretStr
    database: str | None = None
    bolt_advertised_address: str | None = None
    http_advertised_address: str | None = None

    @field_validator("uri")
    @classmethod
    def _validate_uri(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in _ALLOWED_NEO4J_SCHEMES or not parsed.netloc:
            raise ValueError(
                "NEO4J_URI must use bolt:// or neo4j:// scheme with host and port."
            )
        return value

    def auth(self) -> tuple[str, str]:
        """Return the username/password tuple for Neo4j driver usage."""

        return self.username, self.password.get_secret_value()


class QdrantSettings(BaseModel):
    """Typed Qdrant connection settings."""

    model_config = ConfigDict(frozen=True)

    url: str
    api_key: SecretStr | None = None
    neo4j_id_property: str = "chunk_id"
    external_id_property: str = "chunk_id"

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in _ALLOWED_QDRANT_SCHEMES or not parsed.netloc:
            raise ValueError("QDRANT_URL must be an http(s) URL with host information.")
        return value

    def client_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments suitable for QdrantClient construction."""

        kwargs: dict[str, Any] = {"url": self.url}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key.get_secret_value()
        return kwargs


class FancyRAGSettings(BaseModel):
    """Aggregate typed settings for FancyRAG tooling."""

    model_config = ConfigDict(frozen=True)

    openai: OpenAISettings
    neo4j: Neo4jSettings
    qdrant: QdrantSettings

    _CACHE: ClassVar[FancyRAGSettings | None] = None

    @classmethod
    def load(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        refresh: bool = False,
    ) -> FancyRAGSettings:
        """Load settings from environment variables and cache the result."""

        if env is None and not refresh and cls._CACHE is not None:
            return cls._CACHE

        if env is None:
            from fancyrag.utils.env import load_project_dotenv

            load_project_dotenv()
            source: Mapping[str, str] = os.environ
        else:
            source = env

        def _require(key: str) -> str:
            value = (source.get(key) or "").strip()
            if not value:
                raise ValueError(f"Missing required environment variable: {key}")
            return value

        def _optional(key: str) -> str | None:
            value = (source.get(key) or "").strip()
            return value or None

        openai_settings = OpenAISettings.load(source, actor="fancyrag.service")
        api_key_raw = _require("OPENAI_API_KEY")
        openai_settings = openai_settings.model_copy(
            update={"api_key": SecretStr(api_key_raw)}
        )

        try:
            neo4j_settings = Neo4jSettings(
                uri=_require("NEO4J_URI"),
                username=_require("NEO4J_USERNAME"),
                password=SecretStr(_require("NEO4J_PASSWORD")),
                database=_optional("NEO4J_DATABASE"),
                bolt_advertised_address=_optional("NEO4J_BOLT_ADVERTISED_ADDRESS"),
                http_advertised_address=_optional("NEO4J_HTTP_ADVERTISED_ADDRESS"),
            )
        except ValidationError as exc:
            raise ValueError("Invalid Neo4j configuration") from exc

        qdrant_api_key = _optional("QDRANT_API_KEY")
        try:
            qdrant_settings = QdrantSettings(
                url=_require("QDRANT_URL"),
                api_key=SecretStr(qdrant_api_key) if qdrant_api_key else None,
                neo4j_id_property=_optional("QDRANT_NEO4J_ID_PROPERTY_NEO4J") or "chunk_id",
                external_id_property=_optional("QDRANT_NEO4J_ID_PROPERTY_EXTERNAL") or "chunk_id",
            )
        except ValidationError as exc:
            raise ValueError("Invalid Qdrant configuration") from exc

        settings = cls(openai=openai_settings, neo4j=neo4j_settings, qdrant=qdrant_settings)
        if env is None:
            cls._CACHE = settings
        return settings

    @classmethod
    def clear_cache(cls) -> None:
        """Reset the cached settings instance."""

        cls._CACHE = None

    def export_environment(self) -> dict[str, str]:
        """Return an environment-style mapping of the resolved settings."""

        env: dict[str, str] = {
            "OPENAI_API_KEY": self.openai.api_key.get_secret_value() if self.openai.api_key else "",
            "OPENAI_MODEL": self.openai.chat_model,
            "OPENAI_EMBEDDING_MODEL": self.openai.embedding_model,
            "OPENAI_MAX_ATTEMPTS": str(self.openai.max_attempts),
            "OPENAI_BACKOFF_SECONDS": str(self.openai.backoff_seconds),
            "OPENAI_ENABLE_FALLBACK": "true" if self.openai.enable_fallback else "false",
        }
        if self.openai.embedding_dimensions_override is not None:
            env["OPENAI_EMBEDDING_DIMENSIONS"] = str(self.openai.embedding_dimensions_override)
        if self.openai.api_base_url:
            env["OPENAI_BASE_URL"] = self.openai.api_base_url
            env["OPENAI_ALLOW_INSECURE_BASE_URL"] = "true" if self.openai.allow_insecure_base_url else "false"

        env.update(
            {
                "NEO4J_URI": self.neo4j.uri,
                "NEO4J_USERNAME": self.neo4j.username,
                "NEO4J_PASSWORD": self.neo4j.password.get_secret_value(),
            }
        )
        if self.neo4j.database:
            env["NEO4J_DATABASE"] = self.neo4j.database
        if self.neo4j.bolt_advertised_address:
            env["NEO4J_BOLT_ADVERTISED_ADDRESS"] = self.neo4j.bolt_advertised_address
        if self.neo4j.http_advertised_address:
            env["NEO4J_HTTP_ADVERTISED_ADDRESS"] = self.neo4j.http_advertised_address

        env.update({"QDRANT_URL": self.qdrant.url})
        if self.qdrant.api_key:
            env["QDRANT_API_KEY"] = self.qdrant.api_key.get_secret_value()
        if self.qdrant.neo4j_id_property:
            env["QDRANT_NEO4J_ID_PROPERTY_NEO4J"] = self.qdrant.neo4j_id_property
        if self.qdrant.external_id_property:
            env["QDRANT_NEO4J_ID_PROPERTY_EXTERNAL"] = self.qdrant.external_id_property
        return env


__all__ = [
    "OpenAISettings",
    "Neo4jSettings",
    "QdrantSettings",
    "FancyRAGSettings",
    "DEFAULT_CHAT_MODEL",
    "FALLBACK_CHAT_MODELS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_MAX_RETRY_ATTEMPTS",
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_FALLBACK_ENABLED",
]
