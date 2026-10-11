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
    # Add hard-coded command vocabulary
    words.update([
        "Jarvis", "to-do list", "todo", "timer", "open", "add", "mark", "done",
        "complete", "list", "set", "remind", "note", "screenshot", "spotify",
        "YouTube", "Chrome", "VS Code", "calculator", "terminal",
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

    from faster_whisper import WhisperModel

    to_try = [config.WHISPER_MODEL_SIZE] + [
        m for m in _FALLBACK_ORDER if m != config.WHISPER_MODEL_SIZE
    ]
    last_err = None
    for size in to_try:
        try:
            _model = WhisperModel(
                size,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
                cpu_threads=2,
                num_workers=1,
            )
            if size != config.WHISPER_MODEL_SIZE:
                print(
                    f"[stt/whisper] Warning: fell back to '{size}' "
                    f"('{config.WHISPER_MODEL_SIZE}' OOM). "
                    "Set WHISPER_MODEL_SIZE=tiny in .env to silence this."
                )
            return _model
        except (RuntimeError, MemoryError) as e:
            last_err = e
            print(f"[stt/whisper] '{size}' failed ({e}), trying smaller…")
            _model = None
    raise RuntimeError(f"Could not load any Whisper model. Last error: {last_err}")


def reset_model():
    global _model
    _model = None


class FasterWhisperEngine(STTEngine):
    name = "faster_whisper"

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _transcribe(self, wav_path: str, vocab: list[str] | None = None) -> TranscriptionResult:
        model = load_model()
        t0 = time.perf_counter()

        lang = config.WHISPER_LANGUAGE or None  # empty string → None (auto-detect)
        prompt = _build_initial_prompt(vocab)

        segments, info = model.transcribe(
            wav_path,
            language=lang,
            beam_size=config.WHISPER_BEAM_SIZE,
            temperature=0.0,                      # deterministic; no sampling
            condition_on_previous_text=False,      # each clip is independent
            initial_prompt=prompt,
            hotwords=prompt,                       # bias decoder toward known words
            vad_filter=config.WHISPER_VAD_FILTER,  # strip non-speech
            vad_parameters={"min_silence_duration_ms": 500},
        )

        seg_list = list(segments)                 # materialise lazy generator
        text = " ".join(s.text for s in seg_list).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        avg_lp = float(sum(s.avg_logprob for s in seg_list) / len(seg_list)) if seg_list else None
        no_sp = float(sum(s.no_speech_prob for s in seg_list) / len(seg_list)) if seg_list else None

        return TranscriptionResult(
            text=text,
            latency_ms=latency_ms,
            engine=self.name,
            avg_logprob=avg_lp,
            no_speech_prob=no_sp,
            language=getattr(info, "language", None),
        )
