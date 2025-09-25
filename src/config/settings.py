"""OpenAI configuration helpers for GraphRAG CLI components."""

from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from typing import Mapping, Optional

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
FALLBACK_CHAT_MODELS: frozenset[str] = frozenset({"gpt-4o-mini"})
ALLOWED_CHAT_MODELS: frozenset[str] = frozenset({DEFAULT_CHAT_MODEL, *FALLBACK_CHAT_MODELS})

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536

_ENV_OPENAI_MODEL = "OPENAI_MODEL"
_ENV_OPENAI_EMBEDDING_MODEL = "OPENAI_EMBEDDING_MODEL"
_ENV_OPENAI_EMBEDDING_DIMENSIONS = "OPENAI_EMBEDDING_DIMENSIONS"
_ENV_ACTOR_HINT = "GRAPH_RAG_ACTOR"


@dataclass(frozen=True)
class OpenAISettings:
    """Resolved OpenAI settings with guardrails and audit metadata."""

    chat_model: str
    embedding_model: str
    embedding_dimensions: int
    embedding_dimensions_override: Optional[int]
    actor: str

    @property
    def is_chat_override(self) -> bool:
        return self.chat_model != DEFAULT_CHAT_MODEL

    @property
    def allowed_chat_models(self) -> frozenset[str]:
        return ALLOWED_CHAT_MODELS

    @classmethod
    def load(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        actor: Optional[str] = None,
    ) -> "OpenAISettings":
        """Load settings from environment with strict validation."""

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

        return cls(
            chat_model=requested_chat_model,
            embedding_model=embedding_model,
            embedding_dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            embedding_dimensions_override=override_dimensions,
            actor=actor_name,
        )

    def expected_embedding_dimensions(self) -> int:
        return self.embedding_dimensions_override or self.embedding_dimensions


__all__ = [
    "OpenAISettings",
    "DEFAULT_CHAT_MODEL",
    "FALLBACK_CHAT_MODELS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMENSIONS",
]
