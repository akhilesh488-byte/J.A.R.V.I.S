"""
OpenAI gpt-4o-transcribe STT engine.

Requires: pip install openai   (already in requirements-cloud.txt)
Requires: OPENAI_API_KEY in environment.

Model name verified: "gpt-4o-transcribe" is the current OpenAI audio
transcription model (distinct from the old "whisper-1").
API call: client.audio.transcriptions.create(model=..., file=..., language=...)
Prompt parameter biases decoding toward known vocabulary.

Reference: https://platform.openai.com/docs/api-reference/audio/createTranscription
(prompt is supported and documented for gpt-4o-transcribe)
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
        from openai import OpenAI
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _build_prompt(vocab: list[str] | None) -> str:
    words = set(config.APP_ALIASES.keys())
    words.update(["Jarvis", "todo", "timer", "open", "add", "mark", "done",
                  "complete", "list", "set", "remind", "note"])
    if vocab:
        words.update(vocab)
    return ", ".join(sorted(words))


class OpenAIEngine(STTEngine):
    name = "openai"

    def is_available(self) -> bool:
        if not config.OPENAI_API_KEY:
            return False
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return False

    def _transcribe(self, wav_path: str, vocab: list[str] | None = None) -> TranscriptionResult:
        client = _get_client()
        prompt = _build_prompt(vocab)

        t0 = time.perf_counter()
        try:
            with open(wav_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=f,
                    language="en",
                    prompt=prompt,
                )
        except Exception as e:
            raise RuntimeError(f"OpenAI transcription error: {e}") from e
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return TranscriptionResult(
            text=response.text.strip(),
            latency_ms=latency_ms,
            engine=self.name,
        )
