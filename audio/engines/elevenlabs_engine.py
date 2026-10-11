"""
ElevenLabs Scribe STT engine.

Requires: pip install elevenlabs   (already in requirements-cloud.txt)
Requires: ELEVENLABS_API_KEY in environment.

Model: "scribe_v1" — ElevenLabs' current transcription model.
API verified against elevenlabs Python SDK (ElevenLabs client).

SDK call (as of elevenlabs>=1.0):
    client.speech_to_text.convert(
        file=<BinaryIO>,
        model_id="scribe_v1",
        language_code="en",
        tag_audio_events=False,
    )

Returns: an object with .text

Note: The ElevenLabs SDK version in your environment may differ. If
is_available() returns False despite having the key, check:
  python -c "import elevenlabs; print(elevenlabs.__version__)"
"""

from __future__ import annotations

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from audio.engines.base import STTEngine, TranscriptionResult

_client = None


def _get_client():
    global _client
    if _client is None:
        from elevenlabs import ElevenLabs
        _client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
    return _client


class ElevenLabsScribeEngine(STTEngine):
    name = "elevenlabs_scribe"

    def is_available(self) -> bool:
        if not config.ELEVENLABS_API_KEY:
            return False
        try:
            import elevenlabs  # noqa: F401
            return True
        except ImportError:
            return False

    def _transcribe(self, wav_path: str, vocab: list[str] | None = None) -> TranscriptionResult:
        client = _get_client()

        t0 = time.perf_counter()
        try:
            with open(wav_path, "rb") as f:
                response = client.speech_to_text.convert(
                    file=f,
                    model_id="scribe_v1",
                    language_code="en",
                    tag_audio_events=False,
                )
        except Exception as e:
            raise RuntimeError(f"ElevenLabs Scribe error: {e}") from e
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = getattr(response, "text", "") or ""
        return TranscriptionResult(
            text=text.strip(),
            latency_ms=latency_ms,
            engine=self.name,
        )
