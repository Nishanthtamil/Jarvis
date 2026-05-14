import asyncio
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'jarvis'))

from agent.code.agent import run_code_agent

async def test_code():
    print("Testing Code Agent (Docker Sandbox)...")
    print("-" * 30)
    
    goal = "Calculate the first 10 prime numbers and print them."
    
    async for event in run_code_agent(goal):
        if event["type"] == "status":
            print(f"[*] {event['message']}")
        elif event["type"] == "answer_chunk":
            print(f"\n[JARVIS]: {event['text']}\n")
        elif event["type"] == "error":
            print(f"[!] Error: {event['message']}")

if __name__ == "__main__":
    os.environ["MODEL_FAST"] = "gemma3:4b-it-q4_K_M"
    os.environ["MODEL_SMART"] = "gemma3:4b-it-q4_K_M"
    asyncio.run(test_code())
