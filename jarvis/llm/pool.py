"""
LLM Pool — routes requests to different models based on ModelTier.
"""

from __future__ import annotations

import json
import os
from enum import Enum
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
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# Default models for tiers
MODEL_FAST = os.getenv("MODEL_FAST", "gemma3:4b-it-q4_K_M")
MODEL_SMART = os.getenv("MODEL_SMART", "qwen2.5:14b-instruct-q4")
MODEL_VISION = os.getenv("MODEL_VISION", "llava:7b-v1.6-mistral-q4_K_M")

_enc = tiktoken.get_encoding("cl100k_base")


class ModelTier(str, Enum):
    FAST = "fast"    # Gemma 3 4B - extraction, classification, small summaries
    SMART = "smart"  # Qwen 2.5 14B - planning, synthesis, complex reasoning
    VISION = "vision"


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


class LLMPool:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http = httpx.AsyncClient(timeout=timeout)
        self.models = {
            ModelTier.FAST: MODEL_FAST,
            ModelTier.SMART: MODEL_SMART,
            ModelTier.VISION: MODEL_VISION,
        }

    async def close(self):
        await self._http.aclose()

    def _get_model(self, tier: ModelTier) -> str:
        # Fallback to FAST if SMART not configured or available? 
        # For now, just return what's in self.models.
        return self.models.get(tier, MODEL_FAST)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def complete(
        self,
        messages: list[LLMMessage],
        tier: ModelTier = ModelTier.FAST,
        max_tokens: int = 800,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": self._get_model(tier),
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            # For Ollama's OpenAI compatible endpoint, use response_format
            payload["response_format"] = {"type": "json_object"}

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
        tier: ModelTier = ModelTier.SMART,
        max_tokens: int = 1200,
        temperature: float = 0.2,
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": self._get_model(tier),
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
        tier: ModelTier = ModelTier.FAST,
        max_tokens: int = 600,
    ) -> dict:
        raw = await self.complete(messages, tier=tier, max_tokens=max_tokens, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            fix_messages = messages + [
                LLMMessage(role="assistant", content=raw),
                LLMMessage(
                    role="user",
                    content="Your response was not valid JSON. Return only the corrected JSON object, no extra text.",
                ),
            ]
            raw2 = await self.complete(fix_messages, tier=tier, max_tokens=max_tokens, json_mode=True)
            return json.loads(raw2)

    async def complete_vision(self, image_b64: str, prompt: str) -> str:
        """
        Sends to Ollama /api/chat with image support.
        """
        payload = {
            "model": self.models[ModelTier.VISION],
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64]
                }
            ],
            "stream": False
        }
        resp = await self._http.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


_pool: LLMPool | None = None


def get_pool() -> LLMPool:
    global _pool
    if _pool is None:
        _pool = LLMPool()
    return _pool
