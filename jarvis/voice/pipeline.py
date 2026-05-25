"""
Voice Pipeline — coordinates STT, Routing, and TTS.
"""

from __future__ import annotations

import logging
import base64
import numpy as np
from typing import AsyncGenerator

from voice.stt import STT
from voice.tts import TTS
from voice.vad import VAD
from agent.router import route_request, Intent
from agent.orchestrator import run_research
from agent.code.agent import run_code_agent
from agent.task.agent import run_task_agent
from llm.client import get_client, LLMMessage

logger = logging.getLogger(__name__)

class VoicePipeline:
    def __init__(self):
        self.stt = STT()
        self.tts = TTS()
        self.vad = VAD()

    async def process_audio(self, audio_data: bytes) -> AsyncGenerator[dict, None]:
        """
        Processes a chunk of audio: VAD -> STT -> Intent -> Agent -> TTS.
        """
        # 1. Convert to numpy for VAD and STT
        # Assuming 16kHz 16-bit mono PCM
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            yield {"type": "error", "message": "Invalid audio data format."}
            return

        # 2. VAD: Get speech segments
        segments = self.vad.get_speech_segments(audio_np)
        if not segments:
            yield {"type": "error", "message": "No speech detected."}
            return

        # 3. Transcribe
        # Join all speech segments for transcription
        full_text = ""
        for seg in segments:
            start, end = seg['start'], seg['end']
            segment_audio = audio_np[start:end]
            text = self.stt.transcribe(segment_audio)
            if text:
                full_text += " " + text
        
        full_text = full_text.strip()
        if not full_text:
            yield {"type": "error", "message": "Could not transcribe speech."}
            return

        yield {"type": "status", "message": f"Heard: '{full_text}'"}

        # 4. Intent Routing
        route = await route_request(full_text)
        yield {"type": "status", "message": f"Intent: {route.intent}"}

        # 5. Agent Execution
        if route.intent == Intent.RESEARCH:
            agent_gen = run_research(full_text)
        elif route.intent == Intent.CODE:
            agent_gen = run_code_agent(full_text)
        elif route.intent == Intent.TASK:
            agent_gen = run_task_agent(full_text)
        else: # Intent.CHAT
            async def chat_fallback():
                client = get_client()
                messages = [LLMMessage(role="user", content=full_text)]
                async for chunk in client.stream(messages):
                    yield {"type": "answer_chunk", "text": chunk}
                yield {"type": "done", "iterations": 1}
            agent_gen = chat_fallback()

        full_answer = ""
        async for event in agent_gen:
            if event["type"] == "answer_chunk":
                full_answer += event["text"]
            yield event
        
        # 6. TTS (Final result)
        if full_answer:
            yield {"type": "status", "message": "Generating voice response..."}
            audio_response = self.tts.speak(full_answer)
            if audio_response:
                yield {
                    "type": "audio", 
                    "data": base64.b64encode(audio_response).decode()
                }
            else:
                yield {"type": "error", "message": "TTS generation failed."}
