"""Caching-enabled fixed size splitter used by FancyRAG pipelines."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

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
            """
            No-op initializer that accepts and ignores all positional and keyword arguments.
            
            This preserves a compatible constructor signature for environments where a full
            FixedSizeSplitter implementation is not available; the initializer performs no setup.
            """
            pass

        async def run(self, text: str | Sequence[str], *_args, **_kwargs) -> TextChunks:
            """
            Split the input into one TextChunk per input item.
            
            If `text` is a string it is treated as a single item; if it is a sequence, each element becomes a chunk. Each produced TextChunk contains the original text, an `index` corresponding to the element's position (starting at 0), and `metadata` set to `None`.
            
            Parameters:
                text (str | Sequence[str]): Input text or sequence of text items to split.
            
            Returns:
                TextChunks: A container with one TextChunk per input item; each chunk's `metadata` is `None` and `index` reflects the item's position.
            """
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
        """
        Create a CachingFixedSizeSplitter configured from this CachingSplitterConfig.
        
        Returns:
            CachingFixedSizeSplitter: splitter instance using this config's chunk_size and chunk_overlap.
        """

        return build_caching_splitter(self)


class CachingFixedSizeSplitter(FixedSizeSplitter):
    """Fixed-size splitter that caches results while yielding fresh chunk UIDs."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the caching splitter and prepare its internal caches and scope stack.
        
        Sets up three instance attributes used to store cache blueprints, the most recent outputs per cache key, and the active scope stack:
        - _blueprints: maps a cache key (string or tuple of strings) to a list of blueprint dicts describing chunks (text, index, metadata).
        - _last_outputs: maps a cache key to the last produced TextChunks for quick retrieval.
        - _scope_stack: a list representing nested scope identifiers (string or None) used to isolate cache entries.
        """
        super().__init__(*args, **kwargs)
        self._blueprints: dict[str | tuple[str, ...], list[dict[str, Any]]] = {}
        self._last_outputs: dict[str | tuple[str, ...], TextChunks] = {}
        self._scope_stack: list[str | None] = []

    @contextmanager
    def scoped(self, scope: str | Path | None):
        """
        Temporarily set a scope identifier that isolates cache lookups for the duration of a context.
        
        Parameters:
            scope (str | Path | None): Identifier used to namespace cache keys; if `None`, clears any active scope for the context.
        
        Returns:
            context manager: Yields the splitter instance while the provided scope is active; the previous scope is restored on exit.
        """

        scope_id = str(scope) if scope is not None else None
        self._scope_stack.append(scope_id)
        try:
            yield self
        finally:
            self._scope_stack.pop()

    def _current_scope(self) -> str | None:
        """
        Get the active scope used for cache isolation.
        
        Returns:
            str | None: The current scope string from the top of the scope stack, or `None` if no scope is active.
        """
        if not self._scope_stack:
            return None
        return self._scope_stack[-1]

    def _cache_key(self, text: str | Sequence[str]) -> str | tuple[str, ...]:
        """
        Builds a cache key for the provided text and prefixes it with the active scope when one is set.
        
        Parameters:
            text (str | Sequence[str]): The input text or sequence of text segments to key. If a single string is provided, the base key is that string; if a sequence is provided, the base key is a tuple of the sequence's items.
        
        Returns:
            str | tuple[str, ...]: The cache key. Returns the base string or tuple for the input; if a scope is active, returns a tuple whose first element is the scope followed by the base key (for a single-string base the scope and string are returned as a two-element tuple).
        """
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
        """
        Run the splitter on the given text, using a cached blueprint when available to rehydrate fresh chunk UIDs.
        
        When `config` is provided, caching is bypassed but the same normalization logic is applied before delegating to the upstream splitter. When `config` is not provided, a cache key is computed for `text`; on a cache miss the splitter (or sequence aggregator) is executed, a blueprint (text, index, deep-copied metadata) is stored alongside the most recent output, and the fresh result is returned. On a cache hit the stored blueprint is rehydrated into new TextChunks with regenerated UIDs, and the cached output reference is updated.
        
        Parameters:
            text (str | Sequence[str]): The input text or sequence of texts to split.
            config (Any | None): Optional configuration presence flag; when provided, caching is bypassed but the upstream splitter still receives the normalized inputs.
        
        Returns:
            TextChunks: The resulting chunks for `text`. On cache hits, chunks preserve text/index/metadata but have newly generated `uid` values.
        """
        is_sequence_input = not isinstance(text, str)
        normalized_sequence: tuple[str, ...] | None = None
        if is_sequence_input:
            normalized_sequence = tuple(str(part) for part in text)  # type: ignore[arg-type]

        cache_key_input: str | tuple[str, ...]
        if normalized_sequence is not None:
            cache_key_input = normalized_sequence
        else:
            cache_key_input = text  # type: ignore[assignment]

        parent_run = super().run

        async def _invoke_super(item: str) -> TextChunks:
            return await parent_run(item)

        async def _execute_without_cache(
            input_value: str | tuple[str, ...]
        ) -> TextChunks:
            if isinstance(input_value, tuple):
                combined_chunks: list[TextChunk] = []
                index_offset = 0
                for part in input_value:
                    sub_result = await _invoke_super(part)
                    for chunk in sub_result.chunks:
                        chunk_cls = chunk.__class__
                        combined_chunks.append(
                            chunk_cls(
                                text=chunk.text,
                                index=index_offset + getattr(chunk, "index", 0),
                                metadata=getattr(chunk, "metadata", None),
                                uid=getattr(chunk, "uid", None),
                            )
                        )
                    index_offset = len(combined_chunks)
                return TextChunks(chunks=combined_chunks)
            return await _invoke_super(input_value)

        if config is not None:
            return await _execute_without_cache(cache_key_input)

        key = self._cache_key(cache_key_input)
        blueprint = self._blueprints.get(key)
        if blueprint is None:
            result = await _execute_without_cache(cache_key_input)
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
        """
        Retrieve the cached TextChunks for the given input within the current scope.
        
        Parameters:
            text (str | Sequence[str]): The input string or sequence of strings used as the cache lookup key.
        
        Returns:
            TextChunks | None: `TextChunks` if a cached result exists for the given input and current scope, `None` otherwise.
        """

        return self._last_outputs.get(self._cache_key(text))


def build_caching_splitter(
    config: CachingSplitterConfig | None = None,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> CachingFixedSizeSplitter:
    """
    Create a CachingFixedSizeSplitter from either a CachingSplitterConfig or explicit chunk parameters.
    
    Parameters:
        config (CachingSplitterConfig | None): Configuration dataclass to derive chunk_size and chunk_overlap from. Mutually exclusive with explicit overrides.
        chunk_size (int | None): Chunk size override used when `config` is not provided. Required if `config` is None.
        chunk_overlap (int | None): Chunk overlap override used when `config` is not provided. Defaults to 0 when omitted.
    
    Returns:
        CachingFixedSizeSplitter: A splitter configured with the resolved `chunk_size` and `chunk_overlap`.
    """

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
    "TextChunk",
    "TextChunks",
    "build_caching_splitter",
]
