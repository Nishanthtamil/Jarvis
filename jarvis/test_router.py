import asyncio
import os
import sys

# Add jarvis to path
sys.path.append(os.path.join(os.getcwd(), 'jarvis'))

from agent.router import route_request, Intent

async def test_router():
    test_queries = [
        "What is the capital of France?",
        "Write a python script to sort a list of numbers.",
        "Remind me to call Mom at 5pm.",
        "Hey Jarvis, how's it going?"
    ]
    
    print("Testing Intent Router...")
    print("-" * 30)
    
    for query in test_queries:
        result = await route_request(query)
        print(f"Query: '{query}'")
        print(f"Detected Intent: {result.intent.upper()}")
        print(f"Rationale: {result.rationale}")
        print("-" * 30)

if __name__ == "__main__":
    # Ensure FAST model is set to what's available
    os.environ["MODEL_FAST"] = "gemma3:4b-it-q4_K_M"
    asyncio.run(test_router())
