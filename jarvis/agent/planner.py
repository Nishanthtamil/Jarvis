"""
Planner — breaks the user's research goal into 2-4 subtasks.
Each subtask has a search query and a preferred tool (search | pdf | browser).
Uses json_mode for reliable structured output.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from llm.client import get_client, LLMMessage

PLANNER_SYSTEM = """You are a research planning assistant.
Given a research goal, produce a JSON plan with 2-4 search subtasks.
Each subtask targets a distinct angle of the goal.
Avoid duplicate or overly similar queries.

Respond ONLY with a JSON object in this exact format:
{
  "subtasks": [
    {"query": "...", "tool": "search", "rationale": "..."},
    ...
  ],
  "expected_answer_type": "factual | comparative | explanatory | investigative"
}

tool must be one of: search, pdf, browser
Keep queries short (5-10 words), specific, and search-engine-friendly."""


class Subtask(BaseModel):
    query: str
    tool: str  # "search" | "pdf" | "browser"
    rationale: str


class Plan(BaseModel):
    subtasks: list[Subtask]
    expected_answer_type: str


async def plan(goal: str) -> Plan:
    """Generate a research plan for the given goal."""
    client = get_client()
    messages = [
        LLMMessage(role="system", content=PLANNER_SYSTEM),
        LLMMessage(role="user", content=f"Research goal: {goal}"),
    ]

    for attempt in range(3):
        try:
            data = await client.complete_json(messages, max_tokens=600)
            return Plan.model_validate(data)
        except (ValidationError, KeyError, json.JSONDecodeError) as e:
            if attempt == 2:
                # Fallback: single generic search query
                return Plan(
                    subtasks=[Subtask(query=goal[:100], tool="search", rationale="fallback")],
                    expected_answer_type="factual",
                )
