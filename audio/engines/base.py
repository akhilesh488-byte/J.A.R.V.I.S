"""
Common interface for all STT engines.

Every engine must expose:
  name          str
  is_available() -> bool     – no side effects, safe to call at startup
  transcribe(wav_path, vocab=None) -> TranscriptionResult

TranscriptionResult is a dataclass so callers can access typed fields and
extra engine-specific data is stored in `extras` dict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranscriptionResult:
    text: str
    latency_ms: float
    engine: str
    # optional quality signals – not all engines provide these
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    language: Optional[str] = None
    extras: dict = field(default_factory=dict)


class STTEngine:
    """Abstract base; subclass and override name + is_available + _transcribe."""

    name: str = "base"

    def is_available(self) -> bool:  # pragma: no cover
        raise NotImplementedError

    def _transcribe(self, wav_path: str, vocab: list[str] | None) -> TranscriptionResult:  # pragma: no cover
        raise NotImplementedError

    def transcribe(self, wav_path: str, vocab: list[str] | None = None) -> TranscriptionResult:
        t0 = time.perf_counter()
        result = self._transcribe(wav_path, vocab)
        # Allow engine to set its own latency (it may time internal steps);
        # if it didn't, fill it in here.
        if result.latency_ms == 0.0:
            result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result
