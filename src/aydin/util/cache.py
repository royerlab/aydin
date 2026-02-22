"""Lightweight caching utilities using only the standard library.

Provides a TTL-aware LRU cache decorator that handles unhashable arguments
(e.g., numpy arrays) by keying on ``id()``.
"""

import time
from collections import OrderedDict
from functools import wraps


def ttl_cache(maxsize=128, ttl=60):
    """LRU cache decorator with time-to-live expiry.

    Unlike ``functools.lru_cache``, this decorator supports unhashable
    arguments (numpy arrays, lists, etc.) by keying on ``id()`` for
    objects whose ``__hash__`` is ``None``.

    Parameters
    ----------
    maxsize : int, optional
        Maximum number of cached results, by default 128.
        Least-recently-used entries are evicted first.
    ttl : float, optional
        Time-to-live in seconds, by default 60.
        Cached results older than this are discarded.

    Returns
    -------
    Callable
        Decorated function with ``.cache_clear()`` method.
    """

    def decorator(func):
        _cache = OrderedDict()  # key -> (result, timestamp)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key: use id() for unhashable args, value otherwise
            key = tuple(
                id(a) if hasattr(a, '__hash__') and a.__hash__ is None else a
                for a in args
            )
            if kwargs:
                key += tuple(sorted(kwargs.items()))

            now = time.monotonic()

            if key in _cache:
                result, ts = _cache[key]
                if now - ts < ttl:
                    _cache.move_to_end(key)
                    return result
                del _cache[key]

            result = func(*args, **kwargs)
            _cache[key] = (result, now)

            # Evict oldest entries if over capacity
            while len(_cache) > maxsize:
                _cache.popitem(last=False)

            return result

        wrapper.cache_clear = lambda: _cache.clear()
        return wrapper

    return decorator
