"""
Voice Pipeline — coordinates STT, Routing, and TTS.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from voice.stt import STT
from voice.tts import TTS
from voice.vad import VAD
from agent.router import route_request, Intent

logger = logging.getLogger(__name__)

class VoicePipeline:
    def __init__(self):
        self.stt = STT()
        self.tts = TTS()
        self.vad = VAD()

    async def process_audio(self, audio_data: bytes) -> AsyncGenerator[dict, None]:
        """
        Processes a chunk of audio: STT -> Intent -> Agent -> TTS.
        """
        # 1. Transcribe
        text = self.stt.transcribe(audio_data)
        if not text:
            yield {"type": "error", "message": "Could not hear anything."}
            return

        yield {"type": "status", "message": f"Heard: '{text}'"}

        # 2. Intent Routing
        route = await route_request(text)
        yield {"type": "status", "message": f"Intent: {route.intent}"}

        # 3. Execution (This part is tricky as agents yield streams)
        # For simplicity in voice mode, we'll collect the answer and then speak it.
        # In a more advanced version, we'd stream TTS chunks.
        
        from api.app import chat, ChatRequest
        # We can reuse the existing chat logic by calling the chat endpoint internally or refactoring.
        # For now, let's just use a simplified version.
        
        # ... (implementation would go here, possibly calling agents directly)
        yield {"type": "answer_chunk", "text": "Processing your request..."}
        
        # 4. Speak (Final result)
        # This will be handled by the caller/API which will stream audio back.
