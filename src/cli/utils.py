"""CLI utility helpers for GraphRAG."""

from __future__ import annotations

from typing import Sequence

import structlog

from config.settings import OpenAISettings

logger = structlog.get_logger(__name__)


def ensure_embedding_dimensions(
    embedding: Sequence[float],
    *,
    settings: OpenAISettings,
    override_dimensions: int | None = None,
) -> Sequence[float]:
    """Validate embedding vector dimensions with override awareness.

    Args:
        embedding: Vector returned from OpenAI embeddings API.
        settings: Resolved OpenAI settings for the current actor.
        override_dimensions: Optional explicit override (takes precedence over
            settings.embedding_dimensions_override).

    Returns:
        The original embedding if validation passes.

    Raises:
        ValueError: If the embedding size does not match the expected dimension.
    """

    expected = override_dimensions or settings.embedding_dimensions_override or settings.embedding_dimensions
    actual = len(embedding)

    if actual == expected:
        if override_dimensions or settings.embedding_dimensions_override:
            logger.info(
                "openai.embedding.override_applied",
                actor=settings.actor,
                model=settings.embedding_model,
                expected_dimensions=expected,
            )
        return embedding

    remediation_hint = (
        "Set OPENAI_EMBEDDING_DIMENSIONS to the provider's reported vector length"
        if not (override_dimensions or settings.embedding_dimensions_override)
        else "Update the override to match the embedding service response"
    )

    event = (
        "openai.embedding.override_mismatch"
        if override_dimensions or settings.embedding_dimensions_override
        else "openai.embedding.dimension_mismatch"
    )
    logger.error(
        event,
        actor=settings.actor,
        model=settings.embedding_model,
        expected_dimensions=expected,
        actual_dimensions=actual,
        remediation=remediation_hint,
    )
    raise ValueError(
        f"Embedding length {actual} does not match expected {expected} for {settings.embedding_model}. {remediation_hint}."
    )


__all__ = ["ensure_embedding_dimensions"]
