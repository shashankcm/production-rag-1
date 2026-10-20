"""ResponseCache: lookup, key normalisation, expiry and statistics."""

from types import SimpleNamespace

import pytest

from app import cache as cache_module
from app.cache import ResponseCache


@pytest.fixture
def clock(monkeypatch):
    """A controllable replacement for time, so expiry can be tested without sleeping."""
    now = [1000.0]
    monkeypatch.setattr(cache_module, "time", SimpleNamespace(time=lambda: now[0]))
    return now


def test_a_stored_response_is_returned():
    cache = ResponseCache()
    cache.set("What is a pod?", "The smallest deployable unit.")
    assert cache.get("What is a pod?") == "The smallest deployable unit."


def test_an_unknown_query_is_a_miss():
    assert ResponseCache().get("never asked") is None


def test_keys_ignore_case_and_surrounding_whitespace():
    cache = ResponseCache()
    cache.set("  What Is A Pod?  ", "answer")
    assert cache.get("what is a pod?") == "answer"


def test_different_queries_do_not_collide():
    cache = ResponseCache()
    cache.set("query one", "answer one")
    cache.set("query two", "answer two")
    assert cache.get("query one") == "answer one"
    assert cache.get("query two") == "answer two"


def test_an_entry_is_served_until_the_ttl_and_then_expires(clock):
    cache = ResponseCache(ttl_seconds=60)
    cache.set("q", "a")

    clock[0] += 59
    assert cache.get("q") == "a"

    clock[0] += 2  # 61 seconds after it was stored
    assert cache.get("q") is None


def test_an_expired_entry_is_removed(clock):
    cache = ResponseCache(ttl_seconds=10)
    cache.set("q", "a")
    clock[0] += 11
    cache.get("q")
    assert cache.stats["cache_size"] == 0


def test_setting_again_refreshes_the_entry(clock):
    cache = ResponseCache(ttl_seconds=60)
    cache.set("q", "old")
    clock[0] += 50
    cache.set("q", "new")
    clock[0] += 50  # 100 s after the first set, 50 s after the second
    assert cache.get("q") == "new"


def test_statistics_count_hits_and_misses():
    cache = ResponseCache()
    cache.get("a")          # miss
    cache.set("a", "x")
    cache.get("a")          # hit
    cache.get("a")          # hit
    cache.get("b")          # miss

    stats = cache.stats
    assert (stats["hits"], stats["misses"], stats["cache_size"]) == (2, 2, 1)
    assert stats["hit_rate"] == "50.0%"


def test_the_hit_rate_is_zero_with_no_lookups():
    assert ResponseCache().stats["hit_rate"] == "0.0%"
