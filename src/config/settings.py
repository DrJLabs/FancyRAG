"""OpenAI configuration helpers for GraphRAG CLI components."""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import urlparse

from _compat.structlog import get_logger
from cli.sanitizer import mask_base_url

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


def _parse_boolean_flag(*, raw_value: str, env_key: str, actor_name: str, error_event: str) -> bool:
    normalized = raw_value.lower()
    if normalized in _TRUE_BOOL_VALUES:
        return True
    if normalized in _FALSE_BOOL_VALUES:
        return False
    logger.error(error_event, actor=actor_name, supplied=raw_value)
    raise ValueError(f"Invalid {env_key} value '{raw_value}'. Use true/false.")


@dataclass(frozen=True)
class OpenAISettings:
    """Resolved OpenAI settings with guardrails and audit metadata."""

    chat_model: str
    embedding_model: str
    embedding_dimensions: int
    embedding_dimensions_override: Optional[int]
    actor: str
    max_attempts: int
    backoff_seconds: float
    enable_fallback: bool
    api_base_url: Optional[str] = None
    allow_insecure_base_url: bool = False

    @property
    def is_chat_override(self) -> bool:
        """
        Indicates whether a non-default chat model is configured.
        
        Returns:
            True if `chat_model` is different from DEFAULT_CHAT_MODEL, False otherwise.
        """
        return self.chat_model != DEFAULT_CHAT_MODEL

    @property
    def allowed_chat_models(self) -> frozenset[str]:
        """
        Get the set of chat model identifiers allowed by the application.
        
        Returns:
            frozenset[str]: Allowed chat model identifiers.
        """
        return ALLOWED_CHAT_MODELS

    @classmethod
    def load(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        actor: Optional[str] = None,
    ) -> "OpenAISettings":
        """
        Load OpenAI configuration from the given environment mapping or the process environment, applying validation and overrides.
        
        Parameters:
            env (Optional[Mapping[str, str]]): Optional mapping of environment variables to read; if omitted, os.environ is used.
            actor (Optional[str]): Optional actor name to record; if omitted, the value is taken from the environment variable GRAPH_RAG_ACTOR, then USER, then "unknown".
        
        Description:
            Reads OPENAI_MODEL, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIMENSIONS,
            OPENAI_MAX_ATTEMPTS, OPENAI_BACKOFF_SECONDS, OPENAI_ENABLE_FALLBACK,
            OPENAI_ALLOW_INSECURE_BASE_URL, and OPENAI_BASE_URL (plus GRAPH_RAG_ACTOR
            for actor hint). Validates
            chat-model allowlists, embedding override dimensions, retry/backoff
            ranges, fallback toggles, and base URL overrides. Records informational
            logs for overrides and error logs for invalid values.
        
        Returns:
            OpenAISettings: An instance populated with the resolved chat model, embedding model, default embedding dimensions, any embedding-dimensions override, and the resolved actor name.
        
        Raises:
            ValueError: If OPENAI_MODEL is not in the allowed models.
            ValueError: If OPENAI_EMBEDDING_DIMENSIONS is not a valid positive integer when provided.
        """

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
        override_dimensions = None
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

    def expected_embedding_dimensions(self) -> int:
        """
        Return the effective embedding vector dimensionality for these settings.
        
        Returns:
            The embedding dimension to use: the `embedding_dimensions_override` value if set, otherwise `embedding_dimensions`.
        """
        return self.embedding_dimensions_override or self.embedding_dimensions


__all__ = [
    "OpenAISettings",
    "DEFAULT_CHAT_MODEL",
    "FALLBACK_CHAT_MODELS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_MAX_RETRY_ATTEMPTS",
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_FALLBACK_ENABLED",
]
