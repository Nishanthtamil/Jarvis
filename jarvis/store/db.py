"""
SQLite metadata store.
Tables: sessions, searches, sources, summaries
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

        CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url);
        CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
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
