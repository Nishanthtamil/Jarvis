"""
Filesystem markdown cache.
Stores cleaned markdown keyed by SHA-256 of the URL.
TTL: 24 hours by default (configurable via CACHE_TTL_HOURS env var).
"""

from __future__ import annotations

import hashlib
import os
import time

CACHE_DIR = os.getenv("CACHE_DIR", "./data/cache")
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "24"))
CACHE_TTL_SECS = CACHE_TTL_HOURS * 3600


def _url_to_path(url: str) -> str:
    key = hashlib.sha256(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.md")


def get_cached_markdown(url: str) -> str | None:
    """Returns cached markdown if it exists and is within TTL, else None."""
    path = _url_to_path(url)
    if not os.path.exists(path):
        return None
    age = time.time() - os.path.getmtime(path)
    if age > CACHE_TTL_SECS:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def set_cached_markdown(url: str, markdown: str) -> None:
    """Write markdown to cache. Creates CACHE_DIR if needed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _url_to_path(url)
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)


def invalidate(url: str) -> None:
    path = _url_to_path(url)
    if os.path.exists(path):
        os.remove(path)
