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
    """
    Ensure re-running the splitter in the same named scope returns a distinct Result object with identical chunk texts but newly generated chunk UIDs.
    
    This test:
    - Warms the scope cache by running the splitter once.
    - Retrieves the cached entry and runs the splitter again within the same scope.
    - Asserts the two Result objects are different, the sequence of chunk texts is identical, and the sets of chunk UIDs are disjoint.
    """
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