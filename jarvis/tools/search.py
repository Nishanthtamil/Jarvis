"""
SearXNG search client.
Returns a ranked list of {url, title, snippet, score} dicts.
Scoring is pure string-matching — no model call.
"""

from __future__ import annotations

import os
import re

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8888")
SEARXNG_MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", "10"))

# Domains to skip — login walls, paywalls, and low-value aggregators
SKIP_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "tiktok.com", "reddit.com",  # reddit is JS-heavy; add back with Playwright if needed
    "quora.com", "pinterest.com", "linkedin.com",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def _score(query_tokens: set[str], text: str) -> float:
    doc_tokens = _tokenize(text)
    if not doc_tokens:
        return 0.0
    overlap = query_tokens & doc_tokens
    return len(overlap) / len(query_tokens) if query_tokens else 0.0


def _domain(url: str) -> str:
    try:
        return url.split("/")[2].lstrip("www.")
    except IndexError:
        return ""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
)
async def search(query: str, max_results: int = SEARXNG_MAX_RESULTS) -> list[dict]:
    """
    Run a search against SearXNG.
    Returns up to max_results results sorted by relevance score, skipping bad domains.
    """
    params = {
        "q": query,
        "format": "json",
        "language": "en",
        "time_range": "",
        "safesearch": "0",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(f"{SEARXNG_BASE_URL}/search", params=params)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    query_tokens = _tokenize(query)

    scored = []
    for r in results:
        url = r.get("url", "")
        if not url or _domain(url) in SKIP_DOMAINS:
            continue
        title = r.get("title", "")
        snippet = r.get("content", "")
        score = _score(query_tokens, f"{title} {snippet}")
        scored.append({
            "url": url,
            "title": title,
            "snippet": snippet,
            "domain": _domain(url),
            "score": round(score, 3),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:max_results]
