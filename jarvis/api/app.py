"""
FastAPI application.
Routes:
  GET  /                        → serves frontend/index.html
  POST /research                → SSE stream of research events
  POST /chat                    → Threaded chat endpoint
  GET  /history                 → recent session list
  GET  /health                  → health check
  POST /voice/transcribe        → Speech-to-Text
  GET  /voice/tts               → Text-to-Speech
  WS   /ws                      → Proactive alerts
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel

from agent.orchestrator import run_research
from agent.router import route_request, Intent
from agent.code.agent import run_code_agent
from agent.task.agent import run_task_agent
from agent.task.scheduler import start_scheduler
from agent.proactive.engine import init_proactive_engine, set_alert_callback
from voice.stt import STT
from voice.tts import TTS
from store.db import init_db, get_session_history, get_session_data, log_message, get_messages

app = FastAPI(title="Jarvis Research Agent", version="1.0.0")

# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

async def broadcast_alert(message: dict):
    await manager.broadcast(message)

# Lazy load voice components
_stt = None
_tts = None

def get_stt():
    global _stt
    if _stt is None: _stt = STT()
    return _stt

def get_tts():
    global _tts
    if _tts is None: _tts = TTS()
    return _tts

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
    start_scheduler()
    init_proactive_engine()
    set_alert_callback(broadcast_alert)


class ResearchRequest(BaseModel):
    goal: str


class ChatRequest(BaseModel):
    message: str
    session_id: str = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/history")
async def history():
    sessions = await get_session_history(limit=20)
    return {"sessions": sessions}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Just keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


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
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
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


@app.post("/voice/transcribe")
async def transcribe_audio(request: Request):
    """
    Endpoint for uploading audio for transcription.
    """
    body = await request.body()
    stt = get_stt()
    text = stt.transcribe(body)
    return {"text": text}


@app.get("/voice/tts")
async def text_to_speech(text: str):
    """
    Endpoint for generating speech from text.
    """
    tts = get_tts()
    audio_bytes = tts.speak(text)
    if audio_bytes:
        return Response(content=audio_bytes, media_type="audio/wav")
    return {"error": "TTS failed"}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Threaded chat endpoint. Dispatches to specialist agents based on intent.
    """
    message = req.message.strip()
    session_id = req.session_id

    if not message:
        return {"error": "message is required"}

    # Log user message if we have a session
    if session_id:
        await log_message(session_id, "user", message)

    async def event_stream():
        # 1. Route the request
        history_msgs = await get_messages(session_id) if session_id else []
        route = await route_request(message, history_msgs)
        
        yield f"data: {json.dumps({'type': 'status', 'message': f'Intent: {route.intent} ({route.rationale})'})}\n\n"

        # 2. Dispatch to the correct agent
        if route.intent == Intent.RESEARCH:
            agent_gen = run_research(message, session_id=session_id)
        elif route.intent == Intent.CODE:
            agent_gen = run_code_agent(message)
        elif route.intent == Intent.TASK:
            agent_gen = run_task_agent(message)
        else: # Intent.CHAT
            async def chat_fallback():
                from llm.client import get_client, LLMMessage
                client = get_client()
                messages = [LLMMessage(role="user", content=message)]
                async for chunk in client.stream(messages):
                    yield {"type": "answer_chunk", "text": chunk}
                yield {"type": "done", "iterations": 1}
            agent_gen = chat_fallback()

        async for event in agent_gen:
            if event["type"] == "done":
                # Log assistant answer
                final_sid = event.get("session_id") or session_id
                if final_sid:
                    await log_message(final_sid, "assistant", "Agent task complete.")
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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
