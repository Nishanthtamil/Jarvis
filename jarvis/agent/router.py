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
    VISION = "vision"      # Describe or analyze an attached image

class RoutingResult(BaseModel):
    intent: Intent
    rationale: str

ROUTER_SYSTEM = """You are the intent router for Jarvis, a personal AI assistant.
Your job is to classify the user's message into one of five categories:

1. 'research': For factual questions, explanations, or comparisons that require searching the web.
2. 'code': For requests to write, explain, debug, or run code/scripts.
3. 'task': For setting reminders, scheduling, or managing a to-do list.
4. 'chat': For simple greetings, casual conversation, or follow-up questions that don't need new research.
5. 'vision': When an image is attached and the user asks about it (describe, analyze, read text from, find objects in).

Respond ONLY with a JSON object: {"intent": "...", "rationale": "..."}"""

async def route_request(message: str, history: list[dict] = None, has_image: bool = False) -> RoutingResult:
    """Classify the user request into an Intent."""
    client = get_client()
    
    user_content = f"User Message: {message}"
    if has_image:
        user_content += "\n(Note: The user has attached an image with this request)"

    if history:
        # Include a bit of history context for follow-up classification
        recent = history[-3:]
        hist_str = "\n".join([f"{m['role']}: {m['content'][:100]}" for m in recent])
        user_content = f"Recent History:\n{hist_str}\n\nCurrent Message: {message}"
        if has_image:
            user_content += "\n(Note: The user has attached an image with this request)"

    messages = [
        LLMMessage(role="system", content=ROUTER_SYSTEM),
        LLMMessage(role="user", content=user_content),
    ]
    
    try:
        data = await client.complete_json(messages, tier=ModelTier.FAST, max_tokens=150)
        res = RoutingResult.model_validate(data)
        
        # If has_image is True and intent is ambiguous, prefer Intent.VISION
        if has_image and res.intent in [Intent.CHAT, Intent.RESEARCH]:
             # Only override if it's not clearly CODE or TASK
             res.intent = Intent.VISION
             
        return res
    except Exception:
        # Fallback
        if has_image:
            return RoutingResult(intent=Intent.VISION, rationale="Image attached, defaulting to vision.")
        return RoutingResult(intent=Intent.RESEARCH, rationale="Classification failed, defaulting to research.")
