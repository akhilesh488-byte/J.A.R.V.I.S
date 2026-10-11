"""
Deepgram STT engine (nova-3 with keyterm prompting).

Requires: pip install deepgram-sdk   (already in requirements-cloud.txt)
Requires: DEEPGRAM_API_KEY in environment.

API verified against deepgram-sdk==3.x (PrerecordedOptions).
nova-3 is Deepgram's current best general model as of mid-2025.
Keyterms (the nova-3 equivalent of keywords) bias transcription toward
specific words — we pass APP_ALIASES keys + command vocabulary.

Note: keyterms is only supported on nova-3. If the model is changed to
nova-2 the field is silently ignored, so we log a warning.
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
        from deepgram import DeepgramClient
        _client = DeepgramClient(config.DEEPGRAM_API_KEY)
    return _client


class DeepgramEngine(STTEngine):
    name = "deepgram"

    def is_available(self) -> bool:
        if not config.DEEPGRAM_API_KEY:
            return False
        try:
            import deepgram  # noqa: F401
            return True
        except ImportError:
            return False

    def _transcribe(self, wav_path: str, vocab: list[str] | None = None) -> TranscriptionResult:
        from deepgram import PrerecordedOptions

        client = _get_client()

        # Build keyterms list from APP_ALIASES + vocab
        keyterms = list(config.APP_ALIASES.keys())
        keyterms += ["Jarvis", "todo", "timer", "open", "add", "mark", "done",
                     "complete", "list", "set", "remind", "note"]
        if vocab:
            keyterms += vocab
        # Deduplicate, keep only non-empty
        keyterms = list(dict.fromkeys(k for k in keyterms if k))

        with open(wav_path, "rb") as f:
            buffer_data = f.read()

        payload = {"buffer": buffer_data}
        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            language="en",
            keyterms=keyterms,   # nova-3 feature; ignored gracefully on older models
        )

        t0 = time.perf_counter()
        try:
            response = client.listen.prerecorded.v("1").transcribe_file(payload, options)
        except Exception as e:
            raise RuntimeError(f"Deepgram API error: {e}") from e
        latency_ms = (time.perf_counter() - t0) * 1000.0

        try:
            text = response.results.channels[0].alternatives[0].transcript.strip()
            confidence = response.results.channels[0].alternatives[0].confidence
        except (AttributeError, IndexError, KeyError) as e:
            raise RuntimeError(f"Unexpected Deepgram response shape: {e}") from e

        return TranscriptionResult(
            text=text,
            latency_ms=latency_ms,
            engine=self.name,
            extras={"confidence": confidence},
        )
