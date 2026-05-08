"""
Playwright browser fallback for JavaScript-heavy sites.
Only used when Crawl4AI's httpx-based fetch fails.
Hard concurrency limit: 2 simultaneous sessions.
"""

from __future__ import annotations

import asyncio
import os

from playwright.async_api import async_playwright

from llm.client import truncate_to_tokens, count_tokens
from store.cache import get_cached_markdown, set_cached_markdown

MAX_TOKENS_PER_SOURCE = int(os.getenv("MAX_TOKENS_PER_SOURCE", "1500"))
BROWSER_TIMEOUT = int(os.getenv("BROWSER_TIMEOUT", "45")) * 1000  # ms
_semaphore = asyncio.Semaphore(2)  # max 2 concurrent browser sessions


async def fetch_with_browser(url: str) -> dict:
    """
    Opens a real Chromium browser, waits for page to load, extracts visible text.
    Always check cache first.
    """
    cached = get_cached_markdown(url)
    if cached:
        return {"url": url, "markdown": cached, "token_count": count_tokens(cached), "ok": True}

    async with _semaphore:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
                )
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                )
                page = await context.new_page()

                # Block images, fonts, and media to save bandwidth and RAM
                await page.route(
                    "**/*",
                    lambda route: route.abort()
                    if route.request.resource_type in ("image", "media", "font", "stylesheet")
                    else route.continue_(),
                )

                await page.goto(url, wait_until="domcontentloaded", timeout=BROWSER_TIMEOUT)
                await page.wait_for_timeout(2000)  # let JS settle

                # Extract visible text via innerText (strips hidden elements)
                text = await page.evaluate("""() => {
                    const unwanted = document.querySelectorAll('nav, footer, header, aside, script, style, form, .ad, .cookie-banner');
                    unwanted.forEach(el => el.remove());
                    return document.body.innerText || '';
                }""")

                await browser.close()

                truncated = truncate_to_tokens(text, MAX_TOKENS_PER_SOURCE)
                set_cached_markdown(url, truncated)

                return {
                    "url": url,
                    "markdown": truncated,
                    "token_count": count_tokens(truncated),
                    "ok": True,
                    "error": None,
                }

        except Exception as e:
            return {"url": url, "markdown": "", "token_count": 0, "ok": False, "error": str(e)}
