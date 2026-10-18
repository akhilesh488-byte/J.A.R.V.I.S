"""
Coqui XTTS v2 TTS engine.
Provides local neural voice cloning with calibrated generation parameters
to prevent audio hallucination timeouts (long beep / stutter).
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SPEAKER_WAV = os.path.join(ROOT, "jarvis_sample.wav")


def clean_text_for_tts(text: str) -> str:
    """
    Pre-processor right before synthesis that strips all markdown (*, #, _)
    and converts all question marks (?) and exclamation points (!) into periods (.).
    This prevents XTTS from generating rising pitch curves that trigger the stutter.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[*_#`~>\[\]]", "", text)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[!]+", ".", cleaned)
    cleaned = re.sub(r"[?]+", ".", cleaned)
    cleaned = re.sub(r"\.+", ".", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned


class XTTSEngine:
    name = "xtts"

    def __init__(self, speaker_wav: Optional[str] = None):
        self.speaker_wav = speaker_wav or DEFAULT_SPEAKER_WAV
        self.tts = None
        self.device = "cpu"
        self._init_model()

    def _init_model(self) -> None:
        try:
            import torch
            from TTS.api import TTS

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        except Exception:
            self.tts = None

    def is_available(self) -> bool:
        if self.tts is not None:
            return True
        try:
            import TTS  # noqa: F401
            return True
        except ImportError:
            return False

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        temperature: float = 0.2,
        repetition_penalty: float = 1.8,
        gpt_cond_len: float = 6.0,
        speed: float = 1.05,
    ) -> str:
        """
        Synthesize speech with anti-hallucination parameters:
        - temperature = 0.2 (Lowers pitch randomness, stops sassy pitch shifts)
        - repetition_penalty = 1.8 (Prevents vocal jumps and continuous tone hallucination)
        """
        cleaned_text = clean_text_for_tts(text)
        if not cleaned_text:
            return ""

        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"xtts_{int(time.time() * 1000)}.wav")

        ref_wav = speaker_wav or self.speaker_wav
        if not os.path.exists(ref_wav):
            alt_ref = os.path.join(ROOT, "jarvis_sample.wav")
            if os.path.exists(alt_ref):
                ref_wav = alt_ref

        if self.tts is None:
            self._init_model()

        if self.tts is not None:
            self.tts.tts_to_file(
                text=cleaned_text,
                speaker_wav=ref_wav,
                language=language,
                file_path=output_path,
                gpt_cond_len=gpt_cond_len,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                speed=speed,
                enable_text_splitting=False,
            )
            return output_path
        raise RuntimeError("Coqui XTTS model is not available.")

    def tts_to_file(
        self,
        text: str,
        file_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        temperature: float = 0.2,
        repetition_penalty: float = 1.8,
        gpt_cond_len: float = 6.0,
        speed: float = 1.05,
        **kwargs,
    ) -> str:
        return self.synthesize(
            text=text,
            output_path=file_path,
            speaker_wav=speaker_wav,
            language=language,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            gpt_cond_len=gpt_cond_len,
            speed=speed,
        )


_xtts_engine = None


def get_xtts_engine() -> XTTSEngine:
    global _xtts_engine
    if _xtts_engine is None:
        _xtts_engine = XTTSEngine()
    return _xtts_engine


def synthesize(text: str, **kwargs) -> str:
    return get_xtts_engine().synthesize(text, **kwargs)


def tts_to_file(text: str, file_path: str, **kwargs) -> str:
    return get_xtts_engine().tts_to_file(text, file_path, **kwargs)
