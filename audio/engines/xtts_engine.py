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


import shutil
import wave

VOICE_CACHE_DIR = os.path.join(ROOT, "jarvis_voice_cache")
os.makedirs(VOICE_CACHE_DIR, exist_ok=True)

CACHE_ALIASES = {
    "yes_sir": "yes_sir",
    "yessir": "yes_sir",
    "command_unrecognized_sir_standing_by": "command_unrecognized_sir_standing_by",
    "command_unrecognized_sir": "command_unrecognized_sir_standing_by",
    "noted_sir": "noted_sir",
    "on_its_way_sir": "on_its_way_sir",
    "standing_by": "standing_by",
    "good_morning_sir": "good_morning_sir",
    "good_afternoon_sir": "good_afternoon_sir",
    "good_evening_sir": "good_evening_sir",
    "launching_spotify_now_sir": "launching_spotify_now_sir",
    "launching_chrome_now_sir": "launching_chrome_now_sir",
    "launching_notepad_now_sir": "launching_notepad_now_sir",
    "launching_calculator_now_sir": "launching_calculator_now_sir",
    "launching_youtube_now_sir": "launching_youtube_now_sir",
    "powering_down_systems_good_day_sir": "powering_down_systems_good_day_sir",
}


def _cache_key(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return re.sub(r"\s+", "_", cleaned)


def get_cached_audio(text: str) -> Optional[str]:
    key = _cache_key(text)
    target_key = CACHE_ALIASES.get(key, key)
    path = os.path.join(VOICE_CACHE_DIR, f"{target_key}.wav")
    if os.path.exists(path):
        return path
    return None


def sanitize_xtts_audio(wav_path: str, max_words: Optional[int] = None) -> bool:
    """
    Cleans synthesized audio without truncating normal speech:
      1. Trims trailing dead silence / carrier hum at the end of speech.
      2. If the model experiences a runaway generation (overrun beyond expected duration),
         detects where speech transitioned into a pure monotone hallucination (e.g. 750Hz beep)
         and cleanly truncates the tone tail.
      3. Applies a smooth 30ms cosine fade-out at the end.
    """
    try:
        import numpy as np

        with wave.open(wav_path, "rb") as w:
            sr = w.getframerate()
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            n_frames = w.getnframes()
            data = np.frombuffer(w.readframes(n_frames), dtype=np.int16)

        if len(data) == 0:
            return False

        dur_sec = len(data) / sr
        expected_max_sec = max(3.0, (max_words or 10) * 0.75 + 1.8) if max_words else 15.0

        win_len = int(sr * 0.04)  # 40ms window
        hop_len = int(sr * 0.02)  # 20ms hop
        n_wins = (len(data) - win_len) // hop_len

        last_audio_idx = len(data)

        # 1. Backward scan: trim trailing dead silence after speech
        for i in range(n_wins - 1, -1, -1):
            start = i * hop_len
            chunk = data[start:start + win_len]
            rms = np.sqrt(np.mean(chunk.astype(float) ** 2))
            if rms > 450:
                last_audio_idx = min(len(data), start + win_len + int(sr * 0.15))
                break

        # 2. Overrun protection: if file ran away far beyond expected word length,
        # detect if a pure continuous tone (XTTS 750Hz beep hallucination) has taken over
        if dur_sec > expected_max_sec:
            for i in range(int(sr * 1.0) // hop_len, n_wins):
                start = i * hop_len
                chunk = data[start:start + win_len]
                fft = np.abs(np.fft.rfft(chunk))
                tot = np.sum(fft ** 2)
                if tot > 0:
                    peak_ratio = np.max(fft ** 2) / tot
                    peak_freq = np.argmax(fft) * sr / (2 * len(fft))
                    if peak_ratio > 0.35 and peak_freq > 300:
                        last_audio_idx = min(last_audio_idx, max(int(sr * 1.0), start))
                        break

            if last_audio_idx == len(data):
                last_audio_idx = int(expected_max_sec * sr)

        # Preserve at least minimum speech samples
        min_samples = int(sr * min(dur_sec, 0.8))
        last_audio_idx = max(last_audio_idx, min_samples)

        cleaned = data[:last_audio_idx].copy()

        # Smooth 30ms fade-out to prevent audio pops
        fade_len = min(int(sr * 0.03), len(cleaned))
        if fade_len > 0:
            fade = np.linspace(1.0, 0.0, fade_len)
            cleaned[-fade_len:] = (cleaned[-fade_len:] * fade).astype(np.int16)

        with wave.open(wav_path, "wb") as w:
            w.setnchannels(n_channels)
            w.setsampwidth(sampwidth)
            w.setframerate(sr)
            w.writeframes(cleaned.tobytes())

        return True
    except Exception:
        return False


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
        speed: float = 1.05,
        **kwargs,
    ) -> str:
        """
        Synthesize speech with instant audio caching and anti-hallucination sanitization.
        """
        cleaned_text = clean_text_for_tts(text)
        if not cleaned_text:
            return ""

        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"xtts_{int(time.time() * 1000)}.wav")

        # Check prompt cache first (instant response, 0ms latency)
        cached_wav = get_cached_audio(cleaned_text)
        if cached_wav and os.path.exists(cached_wav):
            if os.path.abspath(output_path) != os.path.abspath(cached_wav):
                shutil.copyfile(cached_wav, output_path)
            return output_path

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
                speed=speed,
            )
            # Remove trailing tone/beep hallucinations or dead air
            sanitize_xtts_audio(output_path, max_words=len(cleaned_text.split()))

            return output_path
        raise RuntimeError("Coqui XTTS model is not available.")

    def tts_to_file(
        self,
        text: str,
        file_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        temperature: float = 0.2,
        repetition_penalty: float = 2.0,
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
