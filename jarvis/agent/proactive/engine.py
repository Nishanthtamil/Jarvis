"""
Proactive Engine — background jobs for tracking topics and briefings.
"""

from __future__ import annotations

import asyncio
import logging
import json
import time
from datetime import datetime
from agent.task.scheduler import get_scheduler
from store.db import get_active_trackers, update_tracker_last_checked, log_alert

logger = logging.getLogger(__name__)

# Global callback for WebSocket notifications, set by app.py
_alert_callback = None

def set_alert_callback(callback):
    global _alert_callback
    _alert_callback = callback

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
        
        # Logic for different tracker types
        if tracker_type == "daily_briefing":
            # Only alert if more than 20 hours since last alert (approx daily)
            last_alert = t["last_alert"]
            if last_alert is None or (time.time() - last_alert > 72000):
                content = "Good morning! Your briefing is ready. All tasks are up to date."
                alert_id = await log_alert(tracker_id, content)
                if _alert_callback:
                    await _alert_callback({"type": "alert", "id": alert_id, "content": content})
        
        elif tracker_type == "topic_watch":
            # Mock check for a topic
            topic = config.get("topic", "AI")
            # In a real app, this would perform a search and compare with last_checked
            pass

        await update_tracker_last_checked(tracker_id)

def init_proactive_engine():
    """
    Schedule the periodic tracker check.
    """
    scheduler = get_scheduler()
    # Check every hour (or for testing, every minute)
    scheduler.add_job(check_trackers, 'interval', minutes=60, id='check_trackers')
    logger.info("Proactive engine initialized.")
