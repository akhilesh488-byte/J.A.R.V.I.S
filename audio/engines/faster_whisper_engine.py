"""
faster-whisper STT engine.

Verified parameter names against faster-whisper==1.2.1:
  WhisperModel.transcribe() accepts:
    language, beam_size, temperature, condition_on_previous_text,
    initial_prompt, hotwords, vad_filter, vad_parameters

  WhisperModel.__init__() accepts:
    model_size_or_path, device, compute_type, cpu_threads, num_workers

  TranscriptionInfo has: language, avg_logprob  (per segment in Segment)
  Segment has: avg_logprob, no_speech_prob

Supported WHISPER_MODEL_SIZE values (not validated here — faster-whisper
downloads them from HuggingFace on first use):
  tiny, tiny.en, base, base.en, small, small.en, medium, medium.en,
  large-v1, large-v2, large-v3, large-v3-turbo, distil-large-v3
"""

from __future__ import annotations

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from audio.engines.base import STTEngine, TranscriptionResult

_FALLBACK_ORDER = ["tiny.en", "tiny"]

_model = None


def _build_initial_prompt(vocab: list[str] | None) -> str:
    """Build a Whisper initial_prompt from vocabulary + APP_ALIASES keys."""
    words = set(config.APP_ALIASES.keys())
    # Add domain terms and hard-coded command vocabulary
    words.update([
        "gym", "to-dos", "PyTorch", "LangChain", "GitHub", "WhatsApp", "VS Code", "J.A.R.V.I.S.",
        "Jarvis", "to-do list", "todo", "timer", "open", "add", "mark", "done",
        "complete", "list", "set", "remind", "note", "screenshot", "spotify",
        "YouTube", "Chrome", "calculator", "terminal",
    ])
    if vocab:
        words.update(vocab)
    # Whisper uses the initial_prompt as a fake previous transcription;
    # a comma-separated list of relevant words primes the decoder vocabulary.
    return ", ".join(sorted(words))


def load_model(reset: bool = False):
    """Load (or reload) the faster-whisper model.  Thread-safe via GIL for simple cases."""
    global _model
    if reset:
        _model = None
    if _model is not None:
        return _model

    from bench.providers import get_whisper_pipeline
    pipeline = get_whisper_pipeline(reset=reset)
    _model = pipeline.model
    return _model


def reset_model():
    global _model
    _model = None
    from bench.providers import get_whisper_pipeline
    get_whisper_pipeline(reset=True)


class FasterWhisperEngine(STTEngine):
    name = "faster_whisper"

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _transcribe(self, wav_path: str, vocab: list[str] | None = None) -> TranscriptionResult:
        from bench.providers import get_whisper_pipeline
        pipeline = get_whisper_pipeline()
        prompt = _build_initial_prompt(vocab)
        t0 = time.perf_counter()

        text = pipeline.transcribe(
            wav_path,
            initial_prompt=prompt,
            timeout=10.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            language=config.WHISPER_LANGUAGE or None,
            beam_size=config.WHISPER_BEAM_SIZE,
            temperature=0.0,
            vad_filter=config.WHISPER_VAD_FILTER,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return TranscriptionResult(
            text=text,
            latency_ms=latency_ms,
            engine=self.name,
            avg_logprob=None,
            no_speech_prob=None,
            language=config.WHISPER_LANGUAGE or None,
        )

