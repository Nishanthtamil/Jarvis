"""
SQLite metadata store.
Tables: sessions, searches, sources, summaries, messages, user_facts
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

DB_PATH = os.getenv("DB_PATH", "./data/jarvis.db")


async def init_db(db_path: str = DB_PATH):
    """Create tables if they don't exist. Call once at startup."""
    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            goal TEXT NOT NULL,
            created_at REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            answer TEXT
        );

        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            query TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            domain TEXT,
            token_count INTEGER,
            fetch_ok INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            source_id INTEGER,
            content TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source_session TEXT,
            created_at REAL NOT NULL,
            last_updated REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            due_at REAL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trackers (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL, -- 'rss', 'arxiv', 'price', etc.
            config_json TEXT NOT NULL,
            frequency TEXT NOT NULL, -- 'hourly', 'daily', etc.
            last_checked REAL,
            last_alert REAL,
            active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            tracker_id TEXT,
            content TEXT NOT NULL,
            read INTEGER DEFAULT 0,
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url);
        CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        """)
        await db.commit()


@asynccontextmanager
async def get_db(db_path: str = DB_PATH):
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def create_session(session_id: str, goal: str, db_path: str = DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO sessions (id, goal, created_at, status) VALUES (?, ?, ?, 'running')",
            (session_id, goal, time.time()),
        )
        await db.commit()


async def update_session_answer(session_id: str, answer: str, db_path: str = DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE sessions SET answer=?, status='done' WHERE id=?",
            (answer, session_id),
        )
        await db.commit()


async def log_search(
    session_id: str, query: str, results: list[dict], db_path: str = DB_PATH
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO searches (session_id, query, results_json, created_at) VALUES (?, ?, ?, ?)",
            (session_id, query, json.dumps(results), time.time()),
        )
        await db.commit()


async def log_source(
    session_id: str,
    url: str,
    title: str,
    domain: str,
    token_count: int,
    fetch_ok: bool,
    db_path: str = DB_PATH,
) -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO sources (session_id, url, title, domain, token_count, fetch_ok, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, url, title, domain, token_count, int(fetch_ok), time.time()),
        )
        await db.commit()
        return cursor.lastrowid


async def log_summary(
    session_id: str, source_id: int, content: str, db_path: str = DB_PATH
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO summaries (session_id, source_id, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, source_id, content, time.time()),
        )
        await db.commit()


async def log_message(session_id: str, role: str, content: str, db_path: str = DB_PATH) -> None:
    import uuid
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), session_id, role, content, time.time()),
        )
        await db.commit()


async def get_messages(session_id: str, db_path: str = DB_PATH) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def add_user_fact(category: str, content: str, source_session: str = None, db_path: str = DB_PATH) -> None:
    now = time.time()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO user_facts (category, content, source_session, created_at, last_updated) VALUES (?, ?, ?, ?, ?)",
            (category, content, source_session, now, now),
        )
        await db.commit()


async def get_user_facts(category: str = None, db_path: str = DB_PATH) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        if category:
            cursor = await db.execute("SELECT * FROM user_facts WHERE category=? ORDER BY last_updated DESC", (category,))
        else:
            cursor = await db.execute("SELECT * FROM user_facts ORDER BY last_updated DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_session_history(limit: int = 20, db_path: str = DB_PATH) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, goal, created_at, status FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_session_data(session_id: str, db_path: str = DB_PATH) -> dict | None:
    """Retrieve all data for a session to generate a report."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        
        # Get session info
        cursor = await db.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
        session = await cursor.fetchone()
        if not session:
            return None
            
        # Get sources
        cursor = await db.execute("SELECT * FROM sources WHERE session_id=? AND fetch_ok=1", (session_id,))
        sources = await cursor.fetchall()
        
        return {
            "session": dict(session),
            "sources": [dict(s) for s in sources]
        }


async def add_tracker(tracker_id: str, type: str, config: dict, frequency: str, db_path: str = DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO trackers (id, type, config_json, frequency, active) VALUES (?, ?, ?, ?, 1)",
            (tracker_id, type, json.dumps(config), frequency),
        )
        await db.commit()


async def get_active_trackers(db_path: str = DB_PATH) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM trackers WHERE active=1")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def update_tracker_last_checked(tracker_id: str, db_path: str = DB_PATH) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE trackers SET last_checked=? WHERE id=?",
            (time.time(), tracker_id),
        )
        await db.commit()


async def log_alert(tracker_id: str, content: str, db_path: str = DB_PATH) -> str:
    import uuid
    alert_id = str(uuid.uuid4())
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO alerts (id, tracker_id, content, created_at) VALUES (?, ?, ?, ?)",
            (alert_id, tracker_id, content, time.time()),
        )
        await db.execute(
            "UPDATE trackers SET last_alert=? WHERE id=?",
            (time.time(), tracker_id),
        )
        await db.commit()
    return alert_id


async def get_unread_alerts(db_path: str = DB_PATH) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM alerts WHERE read=0 ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
