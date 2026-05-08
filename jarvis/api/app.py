"""
FastAPI application.
Routes:
  GET  /                        → serves frontend/index.html
  POST /research                → SSE stream of research events
  GET  /history                 → recent session list
  GET  /health                  → health check
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.orchestrator import run_research
from store.db import init_db, get_session_history, get_session_data

app = FastAPI(title="Jarvis Research Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    os.makedirs("./data", exist_ok=True)
    await init_db()


class ResearchRequest(BaseModel):
    goal: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/history")
async def history():
    sessions = await get_session_history(limit=20)
    return {"sessions": sessions}


@app.get("/export/{session_id}")
async def export_session(session_id: str):
    """
    Export a research session as a formatted Markdown report.
    """
    data = await get_session_data(session_id)
    if not data:
        return {"error": "Session not found"}

    session = data["session"]
    sources = data["sources"]

    report = [
        f"# JARVIS Research Report",
        f"**Date:** {os.popen('date').read().strip()}",
        f"**Goal:** {session['goal']}",
        f"\n## Final Synthesis\n",
        session["answer"] or "No answer synthesized yet.",
        f"\n## References\n"
    ]

    for i, s in enumerate(sources):
        title = s["title"] or "Untitled"
        report.append(f"[{i+1}] **{title}**")
        report.append(f"   URL: {s['url']}")
        report.append(f"   Domain: {s['domain']}\n")

    report_content = "\n".join(report)
    
    return Response(
        content=report_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=jarvis_report_{session_id[:8]}.md"
        }
    )


@app.post("/research")
async def research(req: ResearchRequest):
    """
    SSE endpoint. Each event is a JSON line prefixed with 'data: '.
    The frontend reads these incrementally.
    """
    goal = req.goal.strip()
    if not goal:
        return {"error": "goal is required"}

    async def event_stream():
        async for event in run_research(goal):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    with open(frontend_path, "r", encoding="utf-8") as f:
        return f.read()
