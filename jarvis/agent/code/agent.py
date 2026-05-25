"""
Code Agent — generates and executes code to solve user tasks.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator

import os
from llm.client import get_client, LLMMessage, ModelTier
from agent.code.sandbox import execute_code

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "llm", "prompts", "code_agent.md")
CODE_AGENT_PROMPT = open(PROMPT_PATH).read()

async def run_code_agent(goal: str, context: str = "") -> AsyncGenerator[dict, None]:
    """
    Generates, executes, and potentially fixes code to achieve a goal.
    Yields status events similar to the research orchestrator.
    """
    client = get_client()
    
    yield {"type": "status", "message": "Generating code solution..."}
    
    messages = [
        LLMMessage(role="system", content=CODE_AGENT_PROMPT),
        LLMMessage(role="user", content=f"Goal: {goal}\n\nContext: {context}"),
    ]
    
    # Use the SMART model for better code generation
    response = await client.complete(messages, tier=ModelTier.SMART, max_tokens=1500)
    
    # Extract code block
    code_match = re.search(r"```(python|bash)\n(.*?)\n```", response, re.DOTALL)
    if not code_match:
        yield {"type": "error", "message": "Failed to generate a valid code block."}
        return
    
    lang = code_match.group(1)
    code = code_match.group(2)
    
    yield {"type": "status", "message": f"Executing {lang} code in sandbox..."}
    
    # Execute in sandbox
    result = execute_code(code, language=lang)
    
    if result["success"]:
        yield {"type": "status", "message": "Code executed successfully."}
        yield {"type": "answer_chunk", "text": f"Output:\n```\n{result['output']}\n```"}
        yield {"type": "done", "iterations": 1}
    else:
        # Attempt one fix
        yield {"type": "status", "message": f"Execution failed. Attempting to fix...\nError: {result['error']}"}
        
        fix_messages = messages + [
            LLMMessage(role="assistant", content=response),
            LLMMessage(role="user", content=f"The execution failed with this error:\n{result['error']}\n\nPlease fix the code and provide the full corrected version."),
        ]
        
        fix_response = await client.complete(fix_messages, tier=ModelTier.SMART, max_tokens=1500)
        
        code_match = re.search(r"```(python|bash)\n(.*?)\n```", fix_response, re.DOTALL)
        if not code_match:
            yield {"type": "error", "message": "Failed to generate a fixed code block."}
            return
        
        lang = code_match.group(1)
        code = code_match.group(2)
        
        yield {"type": "status", "message": f"Executing fixed {lang} code..."}
        result = execute_code(code, language=lang)
        
        if result["success"]:
            yield {"type": "status", "message": "Fixed code executed successfully."}
            yield {"type": "answer_chunk", "text": f"Output:\n```\n{result['output']}\n```"}
            yield {"type": "done", "iterations": 2}
        else:
            yield {"type": "error", "message": f"Code execution failed after fix attempt: {result['error']}"}
