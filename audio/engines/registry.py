"""
Engine registry — returns the right engine instance by name.

Import and use:
    from audio.engines.registry import get_engine, transcribe_with_fallback
"""

from __future__ import annotations

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def _all_engines() -> dict:
    """Lazy imports so missing packages don't blow up at startup."""
    from audio.engines.faster_whisper_engine import FasterWhisperEngine
    from audio.engines.deepgram_engine import DeepgramEngine
    from audio.engines.openai_engine import OpenAIEngine
    from audio.engines.elevenlabs_engine import ElevenLabsScribeEngine
    from audio.engines.cohere_engine import CohereTranscribeEngine
    return {
        "faster_whisper": FasterWhisperEngine(),
        "deepgram": DeepgramEngine(),
        "openai": OpenAIEngine(),
        "elevenlabs_scribe": ElevenLabsScribeEngine(),
        "cohere_transcribe": CohereTranscribeEngine(),
    }


_registry: dict | None = None


def _get_registry() -> dict:
    global _registry
    if _registry is None:
        _registry = _all_engines()
    return _registry


def get_engine(name: str):
    """Return engine instance by name; raises KeyError if unknown."""
    engines = _get_registry()
    if name not in engines:
        raise KeyError(
            f"Unknown STT engine '{name}'. "
            f"Available: {list(engines.keys())}"
        )
    return engines[name]


def transcribe_with_fallback(wav_path: str, vocab: list[str] | None = None):
    """
    Try config.STT_ENGINE first, then each entry in config.STT_FALLBACK.
    Returns TranscriptionResult from the first engine that succeeds.
    Raises RuntimeError if all fail.
    """
    from audio.engines.base import TranscriptionResult
    engines_to_try = [config.STT_ENGINE] + [
        e for e in config.STT_FALLBACK if e != config.STT_ENGINE
    ]
    last_err = None
    for name in engines_to_try:
        try:
            engine = get_engine(name)
        except KeyError as e:
            print(f"[stt] {e}")
            last_err = e
            continue
        if not engine.is_available():
            print(f"[stt] Engine '{name}' not available (missing package or API key). Skipping.")
            last_err = RuntimeError(f"Engine '{name}' not available")
            continue
        try:
            result = engine.transcribe(wav_path, vocab=vocab)
            return result
        except Exception as e:
            print(f"[stt] Engine '{name}' failed: {e}. Trying next.")
            last_err = e
    raise RuntimeError(
        f"All STT engines failed. Last error: {last_err}. "
        "Check DEEPGRAM_API_KEY / OPENAI_API_KEY or ensure faster-whisper is installed."
    )
