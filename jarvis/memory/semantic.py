"""
Semantic Memory — extracts persistent user facts from research sessions.
"""

from __future__ import annotations

import json
from llm.client import get_client, LLMMessage, ModelTier
from store.db import add_user_fact

SEMANTIC_EXTRACTOR_SYSTEM = """You are a user profile specialist.
Given a research goal and the final synthesized answer, extract 1-3 persistent facts about the user's interests, background, or preferences.
Focus on "The user is interested in...", "The user frequently researches...", "The user's background includes...".

Return JSON: {"facts": [{"category": "interest|background|preference", "content": "..."}]}
If nothing new or significant is found, return {"facts": []}."""

async def evaluate_and_store_user_facts(session_id: str, goal: str, answer: str) -> None:
    """
    Evaluate a completed session and extract any persistent user facts.
    """
    client = get_client()
    messages = [
        LLMMessage(role="system", content=SEMANTIC_EXTRACTOR_SYSTEM),
        LLMMessage(role="user", content=f"Research Goal: {goal}\n\nFinal Answer: {answer[:2000]}"),
    ]
    
    try:
        data = await client.complete_json(messages, tier=ModelTier.FAST, max_tokens=300)
        facts = data.get("facts", [])
        for fact in facts:
            await add_user_fact(
                category=fact["category"],
                content=fact["content"],
                source_session=session_id
            )
    except Exception:
        pass
