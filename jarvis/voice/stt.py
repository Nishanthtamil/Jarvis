"""
Speech-to-Text (STT) using Faster-Whisper.
"""

from __future__ import annotations

import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class STT:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        model_size: tiny, base, small, medium, large-v3
        device: cpu, cuda
        compute_type: int8, float16
        """
        try:
            logger.info(f"Loading Whisper model '{model_size}' on {device}...")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None

    def transcribe(self, audio: str | bytes | np.ndarray) -> str:
        """
        Transcribe audio to text.
        audio can be a file path, bytes, or numpy array.
        """
        if self.model is None:
            return ""

        segments, info = self.model.transcribe(audio, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        
        logger.info(f"Transcribed ({info.language}): {text}")
        return text
