"""
Intent Router — classifies user queries and dispatches them to specialist agents.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel

from llm.client import get_client, LLMMessage, ModelTier

class Intent(str, Enum):
    RESEARCH = "research"  # "What is...", "Compare...", deep knowledge retrieval
    CODE = "code"          # "Write a script...", "Debug...", "How do I code..."
    TASK = "task"          # "Remind me...", "Schedule...", "Add to my list"
    CHAT = "chat"          # Conversational, greetings, follow-ups without research

class RoutingResult(BaseModel):
    intent: Intent
    rationale: str

ROUTER_SYSTEM = """You are the intent router for Jarvis, a personal AI assistant.
Your job is to classify the user's message into one of four categories:

1. 'research': For factual questions, explanations, or comparisons that require searching the web.
2. 'code': For requests to write, explain, debug, or run code/scripts.
3. 'task': For setting reminders, scheduling, or managing a to-do list.
4. 'chat': For simple greetings, casual conversation, or follow-up questions that don't need new research.

Respond ONLY with a JSON object: {"intent": "...", "rationale": "..."}"""

async def route_request(message: str, history: list[dict] = None) -> RoutingResult:
    """Classify the user request into an Intent."""
    client = get_client()
    
    user_content = f"User Message: {message}"
    if history:
        # Include a bit of history context for follow-up classification
        recent = history[-3:]
        hist_str = "\n".join([f"{m['role']}: {m['content'][:100]}" for m in recent])
        user_content = f"Recent History:\n{hist_str}\n\nCurrent Message: {message}"

    messages = [
        LLMMessage(role="system", content=ROUTER_SYSTEM),
        LLMMessage(role="user", content=user_content),
    ]
    
    try:
        data = await client.complete_json(messages, tier=ModelTier.FAST, max_tokens=150)
        return RoutingResult.model_validate(data)
    except Exception:
        # Fallback to research if classification fails
        return RoutingResult(intent=Intent.RESEARCH, rationale="Classification failed, defaulting to research.")
