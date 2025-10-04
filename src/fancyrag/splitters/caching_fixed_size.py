"""Caching-enabled fixed size splitter used by FancyRAG pipelines."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

try:  # pragma: no cover - exercised only in minimal CI environments
    from pydantic import ValidationError
except ModuleNotFoundError:  # pragma: no cover
    class ValidationError(ValueError):  # type: ignore[no-redef]
        """Fallback validation error when pydantic is unavailable."""


try:  # pragma: no branch - import-time dependency check
    from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
        FixedSizeSplitter,
    )
    from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
except ModuleNotFoundError:  # pragma: no cover - dependencies unavailable in minimal environments
    @dataclass
    class TextChunk:  # type: ignore[no-redef]
        """Simple text chunk representation used for caching in tests."""

        text: str
        index: int
        metadata: Any | None = None
        uid: str = field(default_factory=lambda: str(uuid4()))

    @dataclass
    class TextChunks:  # type: ignore[no-redef]
        """Container matching the interface expected from neo4j_graphrag splitters."""

        chunks: list[TextChunk]

    class FixedSizeSplitter:  # type: ignore[no-redef]
        """Fallback splitter that yields one chunk per input string."""

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def run(self, text: str | Sequence[str], *_args, **_kwargs) -> TextChunks:
            if isinstance(text, str):
                items = [text]
            else:
                items = list(text)
            chunks = [
                TextChunk(text=item, index=index, metadata=None) for index, item in enumerate(items)
            ]
            return TextChunks(chunks=chunks)


@dataclass(frozen=True)
class CachingSplitterConfig:
    """Typed configuration for the caching splitter factory."""

    chunk_size: int
    chunk_overlap: int = 0

    def create_splitter(self) -> CachingFixedSizeSplitter:
        """Instantiate a splitter using this configuration."""

        return build_caching_splitter(self)


class CachingFixedSizeSplitter(FixedSizeSplitter):
    """Fixed-size splitter that caches results while yielding fresh chunk UIDs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._blueprints: dict[str | tuple[str, ...], list[dict[str, Any]]] = {}
        self._last_outputs: dict[str | tuple[str, ...], TextChunks] = {}
        self._scope_stack: list[str | None] = []

    @contextmanager
    def scoped(self, scope: str | Path | None):
        """Scope cache lookups to a specific source identifier."""

        scope_id = str(scope) if scope is not None else None
        self._scope_stack.append(scope_id)
        try:
            yield self
        finally:
            self._scope_stack.pop()

    def _current_scope(self) -> str | None:
        if not self._scope_stack:
            return None
        return self._scope_stack[-1]

    def _cache_key(self, text: str | Sequence[str]) -> str | tuple[str, ...]:
        if isinstance(text, str):
            base_key: str | tuple[str, ...] = text
        else:
            base_key = tuple(text)
        scope = self._current_scope()
        if scope is None:
            return base_key
        if isinstance(base_key, tuple):
            return (scope, *base_key)
        return (scope, base_key)

    async def run(
        self, text: str | Sequence[str], config: Any | None = None
    ) -> TextChunks:  # type: ignore[override]
        if config is not None:
            try:
                return await super().run(text, config)
            except (TypeError, ValidationError):  # pragma: no cover - fallback path
                return await super().run(text)

        key = self._cache_key(text)
        blueprint = self._blueprints.get(key)
        if blueprint is None:
            result = await super().run(text)
            self._blueprints[key] = [
                {
                    "text": chunk.text,
                    "index": chunk.index,
                    "metadata": copy.deepcopy(getattr(chunk, "metadata", None)),
                }
                for chunk in result.chunks
            ]
            self._last_outputs[key] = result
            return result

        chunks: list[TextChunk] = []
        for template in blueprint:
            chunks.append(
                TextChunk(
                    text=template["text"],
                    index=template["index"],
                    metadata=copy.deepcopy(template["metadata"]),
                    uid=str(uuid4()),
                )
            )
        text_chunks = TextChunks(chunks=chunks)
        self._last_outputs[key] = text_chunks
        return text_chunks

    def get_cached(self, text: str | Sequence[str]) -> TextChunks | None:
        """Return the cached chunk result for ``text`` if available."""

        return self._last_outputs.get(self._cache_key(text))


def build_caching_splitter(
    config: CachingSplitterConfig | None = None,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> CachingFixedSizeSplitter:
    """Construct a caching splitter using a configuration dataclass or explicit overrides."""

    if config is not None and (chunk_size is not None or chunk_overlap is not None):
        raise ValueError("Provide either a config object or explicit overrides, not both.")

    if config is None:
        if chunk_size is None:
            raise ValueError("chunk_size is required when config is not supplied.")
        resolved_overlap = 0 if chunk_overlap is None else chunk_overlap
    else:
        chunk_size = config.chunk_size
        resolved_overlap = config.chunk_overlap

    return CachingFixedSizeSplitter(chunk_size=chunk_size, chunk_overlap=resolved_overlap)


__all__ = [
    "CachingFixedSizeSplitter",
    "CachingSplitterConfig",
    "build_caching_splitter",
    "TextChunk",
    "TextChunks",
]
