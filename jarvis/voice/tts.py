"""
Text-to-Speech (TTS) using Kokoro-ONNX.
"""

from __future__ import annotations

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TTS:
    def __init__(self, model_path: str = "data/models/voice/kokoro-v0_19.onnx", voices_path: str = "data/models/voice/voices.bin"):
        self.model_path = model_path
        self.voices_path = voices_path
        self.kokoro = None
        
        if os.path.exists(model_path) and os.path.exists(voices_path):
            try:
                from kokoro_onnx import Kokoro
                self.kokoro = Kokoro(model_path, voices_path)
                logger.info("Kokoro-ONNX loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Kokoro-ONNX: {e}")
        else:
            logger.warning(f"Kokoro-ONNX models not found at {model_path}. TTS will be disabled or mocked.")

    def speak(self, text: str, voice: str = "af_bella") -> bytes | None:
        """
        Generate speech audio (WAV bytes) from text.
        """
        if self.kokoro is None:
            logger.info(f"[MOCK TTS] Speaking: {text}")
            return None

        try:
            samples, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
            # Convert to 16-bit PCM WAV bytes (simplified)
            import io
            import scipy.io.wavfile as wavfile
            
            byte_io = io.BytesIO()
            wavfile.write(byte_io, sample_rate, (samples * 32767).astype(np.int16))
            return byte_io.getvalue()
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None
