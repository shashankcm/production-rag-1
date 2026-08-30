""" """

import hashlib
import time
from typing import Optional

from langsmith import traceable


class ResponseCache:
    """
    In-memory response cache with TTL (time-to-live) support.

    Note: In production, consider using a persistent cache like Redis.
    - Persistence across server restarts
    - Shared across all server instances
    - Built-in TTL expiration
    """

    def __init__(self, ttl_seconds: int = 300):
        self._ttl = ttl_seconds
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self.__misses = 0

    def _make_key(self, query: str) -> str:
        """Generate a cache key for the normalized query."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[dict]:
        """
        Get cached response if available and hasn't expired
        Returns None on cache miss or expired entry.
        """

        key = self._make_key(query)

        if key not in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self._ttl:
                self._hits += 1
                return entry["response"]
            else:
                del self._cache[key]

        self._misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        """Cache a response"""
        key = self._make_key(query)
        self._cache[key] = {
            "timestamp": time.time(),
            "response": response,
            "query": query,
        }

    @property
    def stats(self) -> dict:
        """Cache statistics"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "cache_size": len(self._cache),
            "hit_rate": f"{hit_rate:.1%}",
        }
