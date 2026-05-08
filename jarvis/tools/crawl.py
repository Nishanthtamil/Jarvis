"""
Crawl4AI extraction wrapper.
Fetches URLs and returns cleaned, token-limited markdown.
Falls back gracefully if extraction fails.
"""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import NoExtractionStrategy

import httpx
import html2text
import logging

logger = logging.getLogger(__name__)

from llm.client import truncate_to_tokens, count_tokens
from store.cache import get_cached_markdown, set_cached_markdown

MAX_TOKENS_PER_SOURCE = int(os.getenv("MAX_TOKENS_PER_SOURCE", "1500"))
MAX_SOURCES_PER_ROUND = int(os.getenv("MAX_SOURCES_PER_ROUND", "4"))
CRAWL_TIMEOUT = int(os.getenv("CRAWL_TIMEOUT", "30"))

# File extensions we skip entirely — no useful text content
SKIP_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".mp4", ".mp3", ".zip", ".exe"}


def _should_skip(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in SKIP_EXTENSIONS)


async def _fallback_fetch(url: str) -> dict:
    """Simple httpx + html2text fallback with browser-like headers."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    logger.info(f"Attempting fallback fetch for {url}")
    try:
        async with httpx.AsyncClient(
            timeout=CRAWL_TIMEOUT, 
            follow_redirects=True, 
            headers=headers,
            http2=True
        ) as client:
            resp = await client.get(url)
            logger.info(f"Fallback fetch for {url} returned status {resp.status_code}")
            resp.raise_for_status()
            
            h = html2text.HTML2Text()
            h.ignore_links = False # Change to False to preserve links
            h.ignore_images = True
            h.ignore_emphasis = True
            h.baseurl = url # Ensure relative links are handled
            markdown = h.handle(resp.text)
            
            truncated = truncate_to_tokens(markdown, MAX_TOKENS_PER_SOURCE)
            token_count = count_tokens(truncated)
            set_cached_markdown(url, truncated)
            
            return {
                "url": url,
                "title": "", # harder to get accurately without parsing
                "markdown": truncated,
                "token_count": token_count,
                "ok": True,
                "error": None,
                "fallback": True
            }
    except Exception as e:
        logger.error(f"Fallback fetch failed for {url}: {e}")
        return {"url": url, "title": "", "markdown": "", "token_count": 0, "ok": False, "error": str(e)}


async def fetch_url(url: str, use_browser: bool = False) -> dict:
    """
    Fetch a single URL. Returns:
    {url, title, markdown, token_count, ok, error}
    Checks filesystem cache first. Writes to cache on success.
    """
    if _should_skip(url):
        return {"url": url, "title": "", "markdown": "", "token_count": 0, "ok": False, "error": "skipped"}

    cached = get_cached_markdown(url)
    if cached:
        return {
            "url": url,
            "title": "",
            "markdown": cached,
            "token_count": count_tokens(cached),
            "ok": True,
            "error": None,
            "from_cache": True,
        }

    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=NoExtractionStrategy(),
        word_count_threshold=50,           # skip pages with less than 50 words
        excluded_tags=["nav", "footer", "header", "aside", "script", "style", "form"],
        remove_overlay_elements=True,
        process_iframes=False,
        simulate_user=use_browser,         # only for browser mode
        magic=use_browser,                 # enable Crawl4AI magic mode for dynamic sites
        page_timeout=CRAWL_TIMEOUT * 1000,
    )

    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url, config=config)

        if not result.success:
            logger.warning(f"Crawl4AI failed for {url}: {result.error_message}. Falling back.")
            return await _fallback_fetch(url)

        raw_md = result.markdown.fit_markdown or result.markdown.raw_markdown or ""
        truncated = truncate_to_tokens(raw_md, MAX_TOKENS_PER_SOURCE)
        token_count = count_tokens(truncated)

        set_cached_markdown(url, truncated)

        return {
            "url": url,
            "title": result.metadata.get("title", "") if result.metadata else "",
            "markdown": truncated,
            "token_count": token_count,
            "ok": True,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Crawl4AI exception for {url}: {e}. Falling back.")
        return await _fallback_fetch(url)


async def fetch_many(urls: list[str], use_browser: bool = False) -> list[dict]:
    """
    Fetch multiple URLs concurrently, capped at MAX_SOURCES_PER_ROUND.
    Returns all results (both success and failure).
    """
    urls = urls[:MAX_SOURCES_PER_ROUND]
    tasks = [fetch_url(url, use_browser=use_browser) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    cleaned = []
    for r in results:
        if isinstance(r, Exception):
            cleaned.append({"url": "unknown", "ok": False, "error": str(r)})
            continue
        cleaned.append(r)
    return cleaned
