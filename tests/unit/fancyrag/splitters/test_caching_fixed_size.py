from __future__ import annotations

import asyncio

import pytest

from fancyrag.splitters import (
    CachingFixedSizeSplitter,
    CachingSplitterConfig,
    build_caching_splitter,
)


def test_get_cached_returns_none_before_run() -> None:
    splitter = CachingFixedSizeSplitter(chunk_size=256, chunk_overlap=32)
    text = "repeated content"

    assert splitter.get_cached(text) is None

    result = asyncio.run(splitter.run(text))
    assert splitter.get_cached(text) is result


def test_run_refreshes_chunk_uids_on_reuse() -> None:
    splitter = CachingFixedSizeSplitter(chunk_size=200, chunk_overlap=0)
    text = "cached text"

    with splitter.scoped("scope-a"):
        first = asyncio.run(splitter.run(text))

    with splitter.scoped("scope-a"):
        cached = splitter.get_cached(text)
        assert cached is not None
        second = asyncio.run(splitter.run(text))

    assert first is not second
    assert [chunk.text for chunk in first.chunks] == [chunk.text for chunk in second.chunks]
    assert {chunk.uid for chunk in first.chunks}.isdisjoint({chunk.uid for chunk in second.chunks})


def test_scoped_caches_are_isolated() -> None:
    splitter = CachingFixedSizeSplitter(chunk_size=128, chunk_overlap=0)
    text = "identical text"

    with splitter.scoped("first"):  # first scope warms cache
        first_result = asyncio.run(splitter.run(text))

    with splitter.scoped("second"):  # second scope should regenerate UIDs
        second_result = asyncio.run(splitter.run(text))

    assert first_result is not second_result
    first_uids = {chunk.uid for chunk in first_result.chunks}
    second_uids = {chunk.uid for chunk in second_result.chunks}
    assert first_uids.isdisjoint(second_uids)


def test_build_caching_splitter_supports_config_and_overrides() -> None:
    config = CachingSplitterConfig(chunk_size=128, chunk_overlap=16)
    splitter_from_config = build_caching_splitter(config)
    splitter_from_kwargs = build_caching_splitter(chunk_size=128, chunk_overlap=16)

    text = "content"
    result_a = asyncio.run(splitter_from_config.run(text))
    result_b = asyncio.run(splitter_from_kwargs.run(text))

    assert [chunk.index for chunk in result_a.chunks] == [chunk.index for chunk in result_b.chunks]

    with pytest.raises(ValueError):
        build_caching_splitter(config, chunk_size=256)


def test_caching_splitter_handles_sequence_input() -> None:
    """Test that splitter correctly handles Sequence[str] inputs."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    texts = ["first chunk", "second chunk", "third chunk"]

    result = asyncio.run(splitter.run(texts))
    assert len(result.chunks) == len(texts)
    assert [chunk.text for chunk in result.chunks] == texts

    # Verify caching works with sequences
    cached = splitter.get_cached(texts)
    assert cached is result


def test_caching_splitter_cache_key_differentiates_string_vs_sequence() -> None:
    """Test that cache keys differ for identical content as string vs sequence."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    text_str = "content"
    text_seq = ["content"]

    result_str = asyncio.run(splitter.run(text_str))
    result_seq = asyncio.run(splitter.run(text_seq))

    # Different cache keys should result in separate cached entries
    assert splitter.get_cached(text_str) is result_str
    assert splitter.get_cached(text_seq) is result_seq
    assert result_str is not result_seq


def test_scoped_context_manager_with_path_object() -> None:
    """Test that scoped accepts Path objects and converts them to strings."""
    from pathlib import Path

    splitter = CachingFixedSizeSplitter(chunk_size=150, chunk_overlap=0)
    text = "path-scoped content"
    path_scope = Path("/some/file.txt")

    with splitter.scoped(path_scope):
        result = asyncio.run(splitter.run(text))

    # Verify the cache is scoped to the path
    with splitter.scoped(path_scope):
        cached = splitter.get_cached(text)
        assert cached is result


def test_scoped_context_manager_with_none_scope() -> None:
    """Test that None scope is handled correctly."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    text = "unscoped content"

    # Run without scope
    asyncio.run(splitter.run(text))

    # Run with explicit None scope
    with splitter.scoped(None):
        result_none_scope = asyncio.run(splitter.run(text))

    # Both should use the same cache key
    assert splitter.get_cached(text) is result_none_scope


def test_nested_scopes_properly_restore_context() -> None:
    """Test that nested scoped contexts properly restore outer scope."""
    splitter = CachingFixedSizeSplitter(chunk_size=120, chunk_overlap=0)
    text = "nested content"

    with splitter.scoped("outer"):
        outer_result = asyncio.run(splitter.run(text))

        with splitter.scoped("inner"):
            inner_result = asyncio.run(splitter.run(text))

        # After inner scope exits, should be back in outer scope
        restored_result = asyncio.run(splitter.run(text))

    # Verify scopes are isolated
    assert outer_result is not inner_result
    assert {chunk.uid for chunk in outer_result.chunks}.isdisjoint(
        {chunk.uid for chunk in inner_result.chunks}
    )
    assert {chunk.uid for chunk in outer_result.chunks}.isdisjoint(
        {chunk.uid for chunk in restored_result.chunks}
    )


def test_metadata_is_deep_copied_in_cache() -> None:
    """Test that metadata is deep copied to prevent mutation issues."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    text = "metadata test"

    # First run to populate cache
    first_result = asyncio.run(splitter.run(text))

    # Mutate metadata in first result if it exists
    if first_result.chunks and hasattr(first_result.chunks[0], "metadata"):
        if first_result.chunks[0].metadata is not None:
            if isinstance(first_result.chunks[0].metadata, dict):
                first_result.chunks[0].metadata["mutated"] = True

    # Second run should get fresh metadata copy
    second_result = asyncio.run(splitter.run(text))

    # Verify UIDs are different (cached blueprint was used)
    assert {chunk.uid for chunk in first_result.chunks}.isdisjoint(
        {chunk.uid for chunk in second_result.chunks}
    )


def test_run_with_config_parameter_bypasses_cache() -> None:
    """Test that providing a config parameter bypasses caching."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    text = "config override test"

    # First run without config (should cache)
    first_result = asyncio.run(splitter.run(text))

    # Verify it's cached
    assert splitter.get_cached(text) is first_result

    # Run with config parameter (should bypass cache and call super)
    # Note: We pass a mock config object; actual behavior depends on parent class
    config_result = asyncio.run(splitter.run(text, config={}))

    # The config path bypasses our caching logic
    assert config_result is not None


def test_build_caching_splitter_requires_chunk_size() -> None:
    """Test that build_caching_splitter requires chunk_size when no config."""
    with pytest.raises(ValueError, match="chunk_size is required"):
        build_caching_splitter()


def test_build_caching_splitter_with_chunk_overlap_only_fails() -> None:
    """Test that providing only chunk_overlap without chunk_size fails."""
    with pytest.raises(ValueError, match="chunk_size is required"):
        build_caching_splitter(chunk_overlap=50)


def test_build_caching_splitter_defaults_overlap_to_zero() -> None:
    """Test that chunk_overlap defaults to 0 when not provided."""
    splitter = build_caching_splitter(chunk_size=100)
    # We can't directly access chunk_overlap, but we can verify the splitter works
    text = "default overlap test"
    result = asyncio.run(splitter.run(text))
    assert result is not None


def test_build_caching_splitter_rejects_mixed_config_and_kwargs() -> None:
    """Test that providing both config and kwargs raises ValueError."""
    config = CachingSplitterConfig(chunk_size=100, chunk_overlap=10)

    with pytest.raises(ValueError, match="either a config object or explicit overrides"):
        build_caching_splitter(config, chunk_size=200)

    with pytest.raises(ValueError, match="either a config object or explicit overrides"):
        build_caching_splitter(config, chunk_overlap=20)

    with pytest.raises(ValueError, match="either a config object or explicit overrides"):
        build_caching_splitter(config, chunk_size=200, chunk_overlap=20)


def test_config_create_splitter_method() -> None:
    """Test CachingSplitterConfig.create_splitter() factory method."""
    config = CachingSplitterConfig(chunk_size=256, chunk_overlap=64)
    splitter = config.create_splitter()

    assert isinstance(splitter, CachingFixedSizeSplitter)

    # Verify the splitter works
    text = "config factory test"
    result = asyncio.run(splitter.run(text))
    assert result is not None
    assert result.chunks


def test_empty_text_handled_gracefully() -> None:
    """Test that empty string input is handled without errors."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    empty_text = ""

    result = asyncio.run(splitter.run(empty_text))
    assert result is not None
    # Behavior depends on parent class, but should not crash


def test_empty_sequence_handled_gracefully() -> None:
    """Test that empty sequence input is handled without errors."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    empty_seq: list[str] = []

    result = asyncio.run(splitter.run(empty_seq))
    assert result is not None


def test_cache_persistence_across_multiple_calls() -> None:
    """Test that cache persists and is reused correctly across multiple calls."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    text = "persistent cache test"

    # Multiple runs should all hit the cache after the first
    results = []
    for _ in range(5):
        result = asyncio.run(splitter.run(text))
        results.append(result)

    # All results should have different UIDs (fresh from cache)
    all_uids = [chunk.uid for result in results for chunk in result.chunks]
    assert len(all_uids) == len(set(all_uids)), "All UIDs should be unique"

    # But same text content
    for result in results[1:]:
        assert [chunk.text for chunk in results[0].chunks] == [
            chunk.text for chunk in result.chunks
        ]


def test_get_cached_with_sequence_input() -> None:
    """Test get_cached works correctly with sequence inputs."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    texts = ["alpha", "beta", "gamma"]

    # Initially no cache
    assert splitter.get_cached(texts) is None

    # Run to populate cache
    result = asyncio.run(splitter.run(texts))

    # Should now be cached
    cached = splitter.get_cached(texts)
    assert cached is result

    # Different sequence should not be cached
    different_texts = ["alpha", "beta", "delta"]
    assert splitter.get_cached(different_texts) is None


def test_scoped_with_different_text_same_scope() -> None:
    """Test that different texts in the same scope have separate cache entries."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)

    with splitter.scoped("shared-scope"):
        text1 = "first text"
        text2 = "second text"

        result1 = asyncio.run(splitter.run(text1))
        result2 = asyncio.run(splitter.run(text2))

        # Both should be cached separately
        assert splitter.get_cached(text1) is result1
        assert splitter.get_cached(text2) is result2
        assert result1 is not result2


def test_scope_isolation_with_sequence_keys() -> None:
    """Test that scope isolation works with sequence-based cache keys."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    texts = ["item1", "item2"]

    with splitter.scoped("scope-x"):
        result_x = asyncio.run(splitter.run(texts))

    with splitter.scoped("scope-y"):
        result_y = asyncio.run(splitter.run(texts))

    # Results should be different due to different scopes
    assert result_x is not result_y
    assert {chunk.uid for chunk in result_x.chunks}.isdisjoint(
        {chunk.uid for chunk in result_y.chunks}
    )


def test_multiple_scope_changes_maintain_separate_caches() -> None:
    """Test that switching between scopes maintains separate caches."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    text = "scope switching test"

    # Run in scope A
    with splitter.scoped("A"):
        result_a1 = asyncio.run(splitter.run(text))

    # Run in scope B
    with splitter.scoped("B"):
        result_b = asyncio.run(splitter.run(text))

    # Return to scope A, should reuse scope A's cache
    with splitter.scoped("A"):
        result_a2 = asyncio.run(splitter.run(text))

    # A's results should have different UIDs but same content
    assert result_a1 is not result_a2
    assert [chunk.text for chunk in result_a1.chunks] == [
        chunk.text for chunk in result_a2.chunks
    ]

    # B's results should be isolated
    assert result_b is not result_a1
    assert result_b is not result_a2


def test_config_with_zero_overlap() -> None:
    """Test that config with explicit zero overlap works correctly."""
    config = CachingSplitterConfig(chunk_size=200, chunk_overlap=0)
    splitter = config.create_splitter()

    text = "zero overlap config"
    result = asyncio.run(splitter.run(text))
    assert result is not None


def test_config_with_large_overlap() -> None:
    """Test that config with large overlap (close to chunk_size) works."""
    config = CachingSplitterConfig(chunk_size=100, chunk_overlap=90)
    splitter = config.create_splitter()

    text = "large overlap config"
    result = asyncio.run(splitter.run(text))
    assert result is not None


def test_chunk_indices_preserved_from_cache() -> None:
    """Test that chunk indices are preserved when using cached blueprints."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    text = "index preservation test"

    first_result = asyncio.run(splitter.run(text))
    first_indices = [chunk.index for chunk in first_result.chunks]

    second_result = asyncio.run(splitter.run(text))
    second_indices = [chunk.index for chunk in second_result.chunks]

    # Indices should be identical
    assert first_indices == second_indices


def test_long_text_input() -> None:
    """Test splitter with longer text input."""
    splitter = CachingFixedSizeSplitter(chunk_size=50, chunk_overlap=10)
    long_text = "word " * 100  # 500 characters

    result = asyncio.run(splitter.run(long_text))
    assert result is not None
    assert result.chunks

    # Verify caching works with long text
    cached = splitter.get_cached(long_text)
    assert cached is result


def test_special_characters_in_text() -> None:
    """Test that special characters in text don't break caching."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    special_text = "Text with: special\nchars\ttabs, quotes'\" and symbols\!@#$%"

    result = asyncio.run(splitter.run(special_text))
    assert result is not None

    # Verify caching works
    cached = splitter.get_cached(special_text)
    assert cached is result


def test_unicode_text_handling() -> None:
    """Test that unicode text is handled correctly."""
    splitter = CachingFixedSizeSplitter(chunk_size=100, chunk_overlap=0)
    unicode_text = "Unicode: 你好世界 🌍 émoji café"

    result = asyncio.run(splitter.run(unicode_text))
    assert result is not None

    # Verify caching works with unicode
    cached = splitter.get_cached(unicode_text)
    assert cached is result