"""
Planner — breaks the user's research goal into 2-4 subtasks.
Each subtask has a search query and a preferred tool (search | pdf | browser).
Uses json_mode for reliable structured output.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError, field_validator

from llm.client import get_client, LLMMessage, ModelTier
from tools.registry import list_tools, TOOL_REGISTRY

def get_planner_system() -> str:
    tools_block = "\n".join(f"- {t['name']}: {t['description']}" for t in list_tools())
    return f"""You are a research planning assistant.
Given a research goal, produce a JSON plan with 2-4 search subtasks.
Each subtask targets a distinct angle of the goal.
Avoid duplicate or overly similar queries.

Respond ONLY with a JSON object in this exact format:
{{
  "subtasks": [
    {{"query": "...", "tool": "search_web", "rationale": "..."}},
    ...
  ],
  "expected_answer_type": "factual | comparative | explanatory | investigative"
}}

Available tools:
{tools_block}

Keep queries short (5-10 words), specific, and search-engine-friendly."""


class Subtask(BaseModel):
    query: str
    tool: str
    rationale: str

    @field_validator("tool")
    @classmethod
    def validate_tool(cls, v: str) -> str:
        if v not in TOOL_REGISTRY:
            raise ValueError(f"Tool '{v}' must be one of {list(TOOL_REGISTRY.keys())}")
        return v


class Plan(BaseModel):
    subtasks: list[Subtask]
    expected_answer_type: str


async def plan(goal: str) -> Plan:
    """Generate a research plan for the given goal."""
    client = get_client()
    messages = [
        LLMMessage(role="system", content=get_planner_system()),
        LLMMessage(role="user", content=f"Research goal: {goal}"),
    ]

    for attempt in range(3):
        try:
            data = await client.complete_json(messages, tier=ModelTier.SMART, max_tokens=600)
            return Plan.model_validate(data)
        except (ValidationError, KeyError, json.JSONDecodeError) as e:
            if attempt == 2:
                # Fallback: single generic search query
                return Plan(
                    subtasks=[Subtask(query=goal[:100], tool="search", rationale="fallback")],
                    expected_answer_type="factual",
                )
