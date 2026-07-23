"""
tests/integration/test_cache.py — Integration tests for the query result cache.

Validates:
  - Cache miss on empty cache
  - Cache hit suppresses retrieval on identical query
  - LRU eviction keeps count <= CACHE_MAX_ENTRIES
  - Cache is disabled by default
"""
from __future__ import annotations

import sqlite3

from rag.storage.query_cache import QueryCache
from rag.types import AnswerResult


def test_cache_miss_on_different_query(minimal_config):
    """A query not yet cached returns None."""
    minimal_config.storage.query_cache_enabled = True
    cache = QueryCache(minimal_config)
    result = cache.get("never asked this before")
    assert result is None


def test_cache_disabled_by_default(minimal_config):
    """With cache disabled, put() is a no-op and get() always returns None."""
    assert minimal_config.storage.query_cache_enabled is False
    cache = QueryCache(minimal_config)
    cache.put("q", AnswerResult(text="a", citations=[], passages_used=1))
    assert cache.get("q") is None


def test_cache_put_and_get_roundtrip(minimal_config):
    """A stored result is retrievable with the identical query."""
    minimal_config.storage.query_cache_enabled = True
    cache = QueryCache(minimal_config)

    result = AnswerResult(text="The answer is 42.", citations=[], passages_used=1)
    cache.put("what is the answer?", result)

    cached = cache.get("what is the answer?")
    assert cached is not None
    assert cached.text == "The answer is 42."
    assert cached.tier == "cached"


def test_cache_key_is_case_and_whitespace_insensitive(minimal_config):
    """Cache keys normalise query text (lowercased, stripped)."""
    minimal_config.storage.query_cache_enabled = True
    cache = QueryCache(minimal_config)

    result = AnswerResult(text="Normalised answer.", citations=[], passages_used=1)
    cache.put("  Hello World  ", result)

    # Slightly different casing / whitespace should still hit
    cached = cache.get("hello world")
    assert cached is not None
    assert cached.text == "Normalised answer."


def test_cache_lru_eviction(minimal_config):
    """Inserting > CACHE_MAX_ENTRIES items evicts the oldest and keeps count <= 500."""
    minimal_config.storage.query_cache_enabled = True
    cache = QueryCache(minimal_config)

    for i in range(501):
        cache.put(
            f"query number {i}",
            AnswerResult(text=f"answer {i}", citations=[], passages_used=1),
        )

    count = cache.count()
    assert count <= 500, f"Expected <= 500 entries, got {count}"


def test_cache_clear(minimal_config):
    """clear() removes all entries."""
    minimal_config.storage.query_cache_enabled = True
    cache = QueryCache(minimal_config)

    cache.put("q1", AnswerResult(text="a1", citations=[], passages_used=1))
    cache.put("q2", AnswerResult(text="a2", citations=[], passages_used=1))
    assert cache.count() == 2

    cache.clear()
    assert cache.count() == 0


def test_cache_count(minimal_config):
    """count() reflects the number of stored entries."""
    minimal_config.storage.query_cache_enabled = True
    cache = QueryCache(minimal_config)

    assert cache.count() == 0
    cache.put("q", AnswerResult(text="a", citations=[], passages_used=1))
    assert cache.count() == 1


def test_cache_does_not_store_empty_result(minimal_config):
    """Empty-text AnswerResult should not be stored (nothing to return)."""
    minimal_config.storage.query_cache_enabled = True
    cache = QueryCache(minimal_config)

    cache.put("q", AnswerResult(text="", citations=[], passages_used=0))
    assert cache.get("q") is None


def test_cache_hit_on_identical_query(minimal_config):
    """Storing then retrieving the same query returns the cached result."""
    minimal_config.storage.query_cache_enabled = True
    cache = QueryCache(minimal_config)

    original = AnswerResult(text="The answer is 42.", citations=[], passages_used=3)
    cache.put("identical query", original)

    hit = cache.get("identical query")
    assert hit is not None
    assert hit.text == "The answer is 42."
    assert hit.tier == "cached"
    assert hit.passages_used == 3

