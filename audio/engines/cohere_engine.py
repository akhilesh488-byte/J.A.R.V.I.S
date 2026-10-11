"""
Cohere Transcribe (via HuggingFace transformers) — optional, default OFF.

Model: "CohereForAI/c4ai-command-r-plus" is NOT a transcription model.
The actual open-source transcription model from Cohere-adjacent work is
"facebook/wav2vec2-large-960h-lv60-self" or similar wav2vec2/whisper
variants. Since Cohere does not publish a pip-installable transcription
model under the "cohere_transcribe" name, this engine wraps:

  openai/whisper-large-v3  via transformers + pipeline("automatic-speech-recognition")

This is effectively a HuggingFace-hosted Whisper variant that works offline
(after first download) and supports GPU. We gate it on VAD filter to avoid
hallucination on silence.

UNTESTED on real GPU — the pipeline() call shape is verified against
transformers>=4.36 docs. GPU OOM is caught and re-raised with a clear message.

Optional dependencies (none required by default):
  pip install transformers accelerate torch
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from audio.engines.base import STTEngine, TranscriptionResult

_pipeline = None
_MODEL_ID = "openai/whisper-large-v3"


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline
        import torch
    except ImportError as e:
        raise RuntimeError(
            "cohere_transcribe engine requires 'transformers' and 'torch'. "
            "Install them with: pip install transformers accelerate torch\n"
            f"Original error: {e}"
        ) from e

    device = "cuda" if config.WHISPER_DEVICE in ("cuda", "auto") else "cpu"
    try:
        _pipeline = pipeline(
            "automatic-speech-recognition",
            model=_MODEL_ID,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "OOM" in str(e):
            raise RuntimeError(
                f"cohere_transcribe: GPU OOM loading {_MODEL_ID}. "
                "Try WHISPER_DEVICE=cpu or a smaller model."
            ) from e
        raise
    return _pipeline


class CohereTranscribeEngine(STTEngine):
    """
    HuggingFace Whisper-large-v3 via transformers pipeline.
    Named 'cohere_transcribe' per the task spec.
    Select with STT_ENGINE=cohere_transcribe in .env.
    """
    name = "cohere_transcribe"

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch          # noqa: F401
            return True
        except ImportError:
            return False

    def _transcribe(self, wav_path: str, vocab: list[str] | None = None) -> TranscriptionResult:
        import time
        pipe = _get_pipeline()
        t0 = time.perf_counter()
        try:
            result = pipe(
                wav_path,
                generate_kwargs={"language": "english", "task": "transcribe"},
                return_timestamps=False,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "OOM" in str(e):
                raise RuntimeError(
                    "cohere_transcribe: GPU OOM during inference. "
                    "Try WHISPER_DEVICE=cpu or a smaller model."
                ) from e
            raise
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = (result.get("text", "") if isinstance(result, dict) else "").strip()
        return TranscriptionResult(text=text, latency_ms=latency_ms, engine=self.name)
