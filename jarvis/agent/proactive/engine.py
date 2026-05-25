"""
Proactive Engine — background jobs for tracking topics and briefings.
"""

from __future__ import annotations

import asyncio
import logging
import json
import time
import hashlib
import difflib
import feedparser
import arxiv
from datetime import datetime
from agent.task.scheduler import get_scheduler
from store.db import get_active_trackers, update_tracker_last_checked, log_alert, update_tracker_config
from tools.crawl import fetch_url
from llm.client import get_client, LLMMessage, ModelTier

logger = logging.getLogger(__name__)

# Global callback for WebSocket notifications, set by app.py
_alert_callback = None

def set_alert_callback(callback):
    global _alert_callback
    _alert_callback = callback

async def check_rss_tracker(tracker_id: str, config: dict, last_checked: float) -> list[str]:
    try:
        feed = feedparser.parse(config["url"])
        new_entries = [e for e in feed.entries
                       if e.get("published_parsed") and
                          time.mktime(e.published_parsed) > last_checked]
        if not new_entries: return []
        
        summaries = []
        client = get_client()
        for entry in new_entries[:3]:
            text = entry.get("summary", entry.get("title", ""))
            summary = await client.complete([
                LLMMessage(role="system", content="Summarize in one sentence."),
                LLMMessage(role="user", content=text)
            ], tier=ModelTier.FAST, max_tokens=80)
            summaries.append(f"{entry.title}: {summary}")
        return summaries
    except Exception as e:
        logger.error(f"RSS tracker error: {e}")
        return []

async def check_arxiv_tracker(tracker_id: str, config: dict, last_checked: float) -> list[str]:
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=config["topic"], max_results=5,
                              sort_by=arxiv.SortCriterion.SubmittedDate)
        results = list(client.results(search))
        new_papers = [r for r in results
                      if r.published.timestamp() > last_checked]
        if not new_papers: return []
        
        summaries = []
        llm = get_client()
        for paper in new_papers:
            summary = await llm.complete([
                LLMMessage(role="system", content="Summarize this paper title and abstract in one sentence."),
                LLMMessage(role="user", content=f"Title: {paper.title}\n\nAbstract: {paper.summary}")
            ], tier=ModelTier.FAST, max_tokens=100)
            summaries.append(f"{paper.title}: {summary}")
        return summaries
    except Exception as e:
        logger.error(f"Arxiv tracker error: {e}")
        return []

async def check_page_diff_tracker(tracker_id: str, config: dict) -> str | None:
    try:
        current = await fetch_url(config["url"])
        if not current["ok"]: return None
        
        content_hash = hashlib.sha256(current["markdown"].encode()).hexdigest()
        stored_hash = config.get("last_hash", "")
        
        if content_hash == stored_hash: return None
        
        diff = difflib.unified_diff(
            config.get("last_content", "").splitlines(),
            current["markdown"].splitlines(), lineterm="")
        diff_text = "\n".join(list(diff)[:50])
        
        config["last_hash"] = content_hash
        config["last_content"] = current["markdown"]
        await update_tracker_config(tracker_id, config)
        return diff_text
    except Exception as e:
        logger.error(f"Page diff tracker error: {e}")
        return None

async def check_trackers():
    """
    Periodic job to check all active trackers.
    """
    logger.info("Proactive Engine: Checking active trackers...")
    trackers = await get_active_trackers()
    for t in trackers:
        tracker_id = t["id"]
        tracker_type = t["type"]
        config = json.loads(t["config_json"])
        last_checked = t["last_checked"] or 0
        
        alerts = []

        if tracker_type == "rss":
            summaries = await check_rss_tracker(tracker_id, config, last_checked)
            if summaries:
                alerts.append(f"New entries in {config['url']}:\n" + "\n".join(summaries))
        
        elif tracker_type == "arxiv":
            summaries = await check_arxiv_tracker(tracker_id, config, last_checked)
            if summaries:
                alerts.append(f"New Arxiv papers for '{config['topic']}':\n" + "\n".join(summaries))

        elif tracker_type == "page_diff":
            diff = await check_page_diff_tracker(tracker_id, config)
            if diff:
                alerts.append(f"Page changed: {config['url']}\n{diff}")

        elif tracker_type == "daily_briefing":
            # Only alert if more than 20 hours since last alert (approx daily)
            last_alert = t["last_alert"]
            if last_alert is None or (time.time() - last_alert > 72000):
                content = "Good morning! Your briefing is ready. All tasks are up to date."
                alerts.append(content)
        
        for content in alerts:
            alert_id = await log_alert(tracker_id, content)
            if _alert_callback:
                await _alert_callback({"type": "alert", "id": alert_id, "content": content})

        await update_tracker_last_checked(tracker_id)

def init_proactive_engine():
    """
    Schedule the periodic tracker check.
    """
    scheduler = get_scheduler()
    # Check every hour (or for testing, every minute)
    scheduler.add_job(check_trackers, 'interval', minutes=60, id='check_trackers')
    logger.info("Proactive engine initialized.")
