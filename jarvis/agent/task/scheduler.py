"""
Task Scheduler — manages background tasks and reminders.
"""

from __future__ import annotations

import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from datetime import datetime

logger = logging.getLogger(__name__)

_scheduler = AsyncIOScheduler()

def start_scheduler():
    if not _scheduler.running:
        _scheduler.start()
        logger.info("Task scheduler started.")

def add_reminder(text: str, due_at: datetime, callback=None):
    """
    Schedule a reminder.
    """
    _scheduler.add_job(
        callback or (lambda: logger.info(f"REMINDER: {text}")),
        trigger=DateTrigger(run_date=due_at),
        args=[text] if callback else [],
        id=f"reminder_{datetime.now().timestamp()}"
    )
    logger.info(f"Reminder scheduled: '{text}' at {due_at}")

def get_scheduler():
    return _scheduler
