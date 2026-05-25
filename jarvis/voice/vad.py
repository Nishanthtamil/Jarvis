"""
Voice Activity Detection (VAD) using Silero-VAD.
"""

from __future__ import annotations

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VAD:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True,
                trust_repo=True
            )
            (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        except Exception as e:
            logger.error(f"Failed to load Silero-VAD: {e}")
            self.model = None

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Check if the audio chunk contains speech.
        """
        if self.model is None:
            return False
            
        # Silero-VAD expects float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
            
        # Get probability of speech
        speech_prob = self.model(torch.from_numpy(audio_chunk), sample_rate).item()
        return speech_prob > self.threshold

    def get_speech_segments(self, audio: np.ndarray, sample_rate: int = 16000):
        """
        Get timestamps for speech segments.
        """
        if self.model is None:
            return []
        return self.get_speech_timestamps(audio, self.model, sampling_rate=sample_rate)
