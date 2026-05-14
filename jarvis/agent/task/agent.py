"""
Task Agent — manages user tasks and reminders using natural language.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import AsyncGenerator

from llm.client import get_client, LLMMessage, ModelTier
from agent.task.scheduler import add_reminder

TASK_SYSTEM_PROMPT = """You are a task management assistant. 
Your goal is to parse user requests for reminders or tasks.
Current time: {now}

Respond ONLY with a JSON object:
{
  "action": "create_reminder | list_tasks | complete_task",
  "description": "...",
  "due_at": "ISO-8601 timestamp or null",
  "rationale": "..."
}
"""

async def run_task_agent(goal: str) -> AsyncGenerator[dict, None]:
    """
    Parses and executes task-related requests.
    """
    client = get_client()
    now = datetime.now().isoformat()
    
    yield {"type": "status", "message": "Parsing task request..."}
    
    messages = [
        LLMMessage(role="system", content=TASK_SYSTEM_PROMPT.format(now=now)),
        LLMMessage(role="user", content=goal),
    ]
    
    try:
        data = await client.complete_json(messages, tier=ModelTier.FAST)
        action = data.get("action")
        description = data.get("description")
        due_at_str = data.get("due_at")
        
        if action == "create_reminder":
            if due_at_str:
                due_at = datetime.fromisoformat(due_at_str.replace("Z", "+00:00"))
                add_reminder(description, due_at)
                yield {"type": "status", "message": f"Reminder created for {due_at_str}"}
                yield {"type": "answer_chunk", "text": f"I've set a reminder: '{description}' for {due_at_str}."}
            else:
                yield {"type": "error", "message": "Could not determine a due date for the reminder."}
        elif action == "list_tasks":
            yield {"type": "answer_chunk", "text": "Task listing is not fully implemented yet, but I've noted your request."}
        else:
            yield {"type": "answer_chunk", "text": f"I understood you want to {action}, but that feature is still being built."}
            
        yield {"type": "done", "iterations": 1}
        
    except Exception as e:
        yield {"type": "error", "message": f"Failed to process task: {e}"}
