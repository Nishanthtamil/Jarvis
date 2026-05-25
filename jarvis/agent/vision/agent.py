from __future__ import annotations
from typing import AsyncGenerator
from llm.client import get_client

async def run_vision_agent(prompt: str, image_b64: str) -> AsyncGenerator[dict, None]:
    """
    Analyzes an image and answers the user's prompt.
    """
    client = get_client()
    
    yield {"type": "status", "message": "Analyzing image with vision model..."}
    
    try:
        response = await client.complete_vision(image_b64, prompt)
        
        # Stream the response back
        # Since complete_vision is non-streaming for now, we yield it as one answer_chunk.
        yield {"type": "answer_chunk", "text": response}
        yield {"type": "done", "iterations": 1}
    except Exception as e:
        yield {"type": "error", "message": f"Vision analysis failed: {e}"}
