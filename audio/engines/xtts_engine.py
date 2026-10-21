"""
Coqui XTTS v2 TTS engine.
Provides local neural voice cloning with calibrated generation parameters
to prevent audio hallucination timeouts (long beep / stutter).
"""

from __future__ import annotations

import os
os.environ["COQUI_TOS_AGREED"] = "1"
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



def _patch_xtts_runtime():
    try:
        import soundfile as sf
        import torch
        import torchaudio
        from transformers import GPT2LMHeadModel
        from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel

        _orig_torchaudio_load = torchaudio.load
        def _sf_load(uri, *args, **kwargs):
            try:
                return _orig_torchaudio_load(uri, *args, **kwargs)
            except Exception:
                data, sr = sf.read(uri)
                tensor = torch.from_numpy(data).float()
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                else:
                    tensor = tensor.t()
                return tensor, sr
        torchaudio.load = _sf_load

        GPT2InferenceModel.can_generate = lambda self: True
        GPT2InferenceModel.prepare_inputs_for_generation = GPT2LMHeadModel.prepare_inputs_for_generation
        GPT2InferenceModel._validate_model_kwargs = lambda self, model_kwargs: None

        _orig_fwd = GPT2InferenceModel.forward
        def _patched_fwd(self, *args, **kwargs):
            pkv = kwargs.get('past_key_values') or (args[1] if len(args) > 1 else None)
            if isinstance(pkv, tuple) and not hasattr(pkv, 'get_seq_length'):
                class PKVWrapper(tuple):
                    def get_seq_length(self):
                        return self[0][0].shape[-2]
                pkv = PKVWrapper(pkv)
                if 'past_key_values' in kwargs:
                    kwargs['past_key_values'] = pkv
                elif len(args) > 1:
                    args = list(args)
                    args[1] = pkv
                    args = tuple(args)
            return _orig_fwd(self, *args, **kwargs)
        GPT2InferenceModel.forward = _patched_fwd
    except Exception as e:
        print(f"[JARVIS] Runtime patch notice: {e}")


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

            _patch_xtts_runtime()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        except Exception as e:
            print(f"[JARVIS] XTTS Load Error: {e}")
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
        gpt_cond_len: int = 6,
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
                speed=speed,
            )
            from jarvis_local import sanitize_xtts_audio
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
        repetition_penalty: float = 1.8,
        gpt_cond_len: int = 6,
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
            gpt_cond_len=int(gpt_cond_len),
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
