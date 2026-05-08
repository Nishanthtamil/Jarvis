"""
LLM client — wraps Ollama's OpenAI-compatible /v1/chat/completions endpoint.
Handles streaming, JSON-constrained generation, token counting, and retries.
"""

from __future__ import annotations

import json
import os
from typing import AsyncGenerator

import httpx
import tiktoken
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b-it-q4_K_M")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# tiktoken cl100k_base is close enough for token counting on all modern models
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens])


class LLMMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: float = OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._http = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        await self._http.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 800,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        """
        Single-shot completion. Returns the assistant's text content.
        json_mode=True instructs Ollama to emit valid JSON only.
        """
        payload: dict = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        resp = await self._http.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 1200,
        temperature: float = 0.2,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming completion — yields text chunks as they arrive.
        Use this for the final synthesis step so the UI feels responsive.
        """
        payload = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        async with self._http.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    chunk_data = json.loads(line[6:])
                    delta = chunk_data["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta

    async def complete_json(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 600,
    ) -> dict:
        """
        Calls complete() with json_mode=True and parses the result.
        Retries once with an explicit correction prompt if JSON is malformed.
        """
        raw = await self.complete(messages, max_tokens=max_tokens, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Ask the model to fix its own output
            fix_messages = messages + [
                LLMMessage(role="assistant", content=raw),
                LLMMessage(
                    role="user",
                    content="Your response was not valid JSON. Return only the corrected JSON object, no extra text.",
                ),
            ]
            raw2 = await self.complete(fix_messages, max_tokens=max_tokens, json_mode=True)
            return json.loads(raw2)


# Module-level singleton — import and use directly
_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
