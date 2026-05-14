import asyncio
import os
import sys
import httpx

async def test_ollama_json():
    print("Testing Ollama JSON mode with Gemma 3...")
    payload = {
        "model": "gemma3:4b-it-q4_K_M",
        "messages": [
            {"role": "system", "content": "Respond only with JSON: {\"test\": \"ok\"}"},
            {"role": "user", "content": "Say ok"}
        ],
        "format": "json",
        "stream": False
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://localhost:11434/api/chat", json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")

if __name__ == "__main__":
    asyncio.run(test_ollama_json())
