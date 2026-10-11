"""
Recording and speech-to-text.

Public API (unchanged — main.py and bench scripts keep working):
    record_command(save_path=None) -> str  (wav path)
    transcribe(wav_path) -> str
    load_model()    # pre-warm the local Whisper model
    reset_model()   # drop model (bench scripts switch sizes between runs)
    warmup()        # one throwaway transcription to heat up the model

Audio capture improvements over v1:
- Pre-roll buffer: ~300 ms of audio is always captured before VAD triggers,
  so the first syllable is never cut.
- Trailing pad: ~200 ms of silence is appended after end-of-speech, so the
  last syllable isn't cut.
- Peak normalization: quiet recordings are boosted to a target peak level
  before transcription (configurable, default on).
- MIC_INDEX: pick a specific PyAudio input device via JARVIS_MIC_INDEX env.
- TTS_POST_SPEAK_DELAY: waits N seconds after TTS so the mic doesn't catch
  speaker echo (main.py must call stt.record_command() AFTER speak()).
- Debug audio: JARVIS_DEBUG_AUDIO=1 saves every recording to
  data/recordings/<timestamp>.wav and appends a JSON line to
  data/recordings/log.jsonl.

STT is now engine-based:
    STT_ENGINE=faster_whisper | deepgram | openai | elevenlabs_scribe | cohere_transcribe
    STT_FALLBACK=faster_whisper   (comma list, tried in order if primary fails)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import wave

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from audio.vad import EnergyEndpointer

RATE = 16000
CHUNK = 1024
CHANNELS = 1

# Target peak level for normalization (int16 scale: 32768 max).
# We aim for 80% of full scale = 26214.
_NORMALIZE_TARGET = 26214
_CLIP_WARN_THRESHOLD = 32000   # warn if any sample is at/above this


# ---------------------------------------------------------------------------
# Pure audio helpers (testable without a microphone)
# ---------------------------------------------------------------------------

def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """
    Peak-normalize int16 samples to _NORMALIZE_TARGET.
    Only RAISES quiet audio; already-loud audio is untouched.
    Returns a new int16 array.
    """
    peak = np.max(np.abs(samples.astype(np.float32)))
    if peak < 1.0:
        return samples  # silent — nothing to normalize
    gain = min(_NORMALIZE_TARGET / peak, 1.0)   # never clip: cap at 1.0
    normalized = (samples.astype(np.float32) * gain).clip(-32768, 32767).astype(np.int16)
    if np.max(np.abs(normalized)) >= _CLIP_WARN_THRESHOLD:
        print("[stt] Warning: audio near clipping — try lowering microphone gain")
    return normalized


def build_preroll_chunks(n_chunks: int) -> list[np.ndarray]:
    """Return a ring-buffer-sized list of zero-filled chunks (pre-roll placeholder)."""
    return [np.zeros(CHUNK, dtype=np.int16)] * n_chunks


def apply_trailing_pad(frames: list[bytes], pad_ms: int, sample_width: int = 2) -> list[bytes]:
    """Append pad_ms of silence to a list of raw audio byte frames."""
    pad_chunks = max(1, int(pad_ms / 1000 * RATE / CHUNK))
    silence = bytes(CHUNK * sample_width)
    return frames + [silence] * pad_chunks


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------

def _write_wav(path: str, frames: list[bytes], sample_width: int = 2) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))


def _default_wav_path() -> str:
    return os.path.join(tempfile.gettempdir(), "jarvis_command.wav")


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

def _debug_log(wav_path: str, result_text: str, engine: str,
               latency_ms: float, extras: dict) -> None:
    """Append one JSON line to data/recordings/log.jsonl."""
    os.makedirs(config.DEBUG_AUDIO_DIR, exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "wav": wav_path,
        "engine": engine,
        "transcript": result_text,
        "latency_ms": round(latency_ms, 1),
    }
    entry.update(extras)
    log_path = os.path.join(config.DEBUG_AUDIO_DIR, "log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_command(save_path: str | None = None) -> str:
    """
    Record audio from the microphone until end-of-speech (energy VAD).

    Improvements over v1:
    - Pre-roll ring buffer so the first syllable is captured.
    - Trailing silence pad so the last syllable isn't cut.
    - Configurable mic device via config.MIC_INDEX.
    - Peak normalization before saving (if config.AUDIO_NORMALIZE).
    - Debug save: if config.DEBUG_AUDIO, copies to data/recordings/<ts>.wav.
    """
    import pyaudio
    from collections import deque

    save_path = save_path or _default_wav_path()

    preroll_chunks = max(1, int(config.AUDIO_PREROLL_MS / 1000 * RATE / CHUNK))
    ring: deque[bytes] = deque(maxlen=preroll_chunks)

    pa = pyaudio.PyAudio()
    open_kwargs = dict(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    if config.MIC_INDEX is not None:
        open_kwargs["input_device_index"] = config.MIC_INDEX

    stream = pa.open(**open_kwargs)
    endpointer = EnergyEndpointer(
        RATE, CHUNK,
        config.VAD_RMS_THRESHOLD,
        config.VAD_SILENCE_SECONDS,
        config.VAD_MIN_SECONDS,
        config.VAD_NO_SPEECH_TIMEOUT,
        config.VAD_MAX_SECONDS,
    )
    print("Listening...")
    frames: list[bytes] = []
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            ring.append(data)            # always fill pre-roll buffer
            frames.append(data)
            samples = np.frombuffer(data, dtype=np.int16)
            stop, reason = endpointer.feed(samples)
            if reason == "" and not endpointer.heard_speech:
                # Haven't detected speech yet — keep pre-roll fresh but don't
                # accumulate frames (we'll prepend the ring buffer on trigger)
                frames = []
            if endpointer.heard_speech and not frames:
                # Speech just started — prepend the pre-roll
                frames = list(ring)
            if stop:
                break
    finally:
        sample_width = pa.get_sample_size(pyaudio.paInt16)
        stream.stop_stream()
        stream.close()
        pa.terminate()

    # Trailing pad
    frames = apply_trailing_pad(frames, config.AUDIO_TRAILING_PAD_MS, sample_width)

    # Normalize
    if config.AUDIO_NORMALIZE and frames:
        raw = np.frombuffer(b"".join(frames), dtype=np.int16)
        raw = normalize_audio(raw)
        # Re-split into CHUNK-sized byte frames
        frames = [raw[i:i+CHUNK].tobytes() for i in range(0, len(raw), CHUNK)]

    _write_wav(save_path, frames, sample_width)

    if config.DEBUG_AUDIO:
        ts = time.strftime("%Y%m%dT%H%M%S")
        debug_path = os.path.join(config.DEBUG_AUDIO_DIR, f"{ts}.wav")
        os.makedirs(config.DEBUG_AUDIO_DIR, exist_ok=True)
        import shutil
        shutil.copy2(save_path, debug_path)

    return save_path


# ---------------------------------------------------------------------------
# Transcription (public API — same signature as before)
# ---------------------------------------------------------------------------

def transcribe(wav_path: str) -> str:
    """
    Transcribe wav_path → text string.
    Uses the engine chain from config.STT_ENGINE / config.STT_FALLBACK.
    Keeps backwards compatibility with the old STT_PROVIDER flag:
      if STT_PROVIDER == "deepgram" and STT_ENGINE is default, uses deepgram.
    """
    from audio.engines.registry import transcribe_with_fallback

    result = transcribe_with_fallback(wav_path)

    if config.DEBUG_AUDIO:
        extras: dict = {}
        if result.avg_logprob is not None:
            extras["avg_logprob"] = round(result.avg_logprob, 4)
        if result.no_speech_prob is not None:
            extras["no_speech_prob"] = round(result.no_speech_prob, 4)
        if result.language:
            extras["language"] = result.language
        extras.update(result.extras)
        _debug_log(wav_path, result.text, result.engine, result.latency_ms, extras)

    return result.text


# ---------------------------------------------------------------------------
# Model lifecycle (kept for bench scripts and backwards compat)
# ---------------------------------------------------------------------------

def load_model():
    """Pre-warm the local faster-whisper model."""
    from audio.engines.faster_whisper_engine import load_model as _lm
    return _lm()


def reset_model():
    """Drop the loaded model (bench scripts switch size/device between runs)."""
    from audio.engines.faster_whisper_engine import reset_model as _rm
    _rm()


def warmup():
    """One throwaway transcription so the first timed call isn't a cold start."""
    path = os.path.join(tempfile.gettempdir(), "jarvis_warmup.wav")
    silence = np.zeros(RATE, dtype=np.int16)
    _write_wav(path, [silence.tobytes()])
    try:
        from audio.engines.faster_whisper_engine import FasterWhisperEngine
        FasterWhisperEngine().transcribe(path)
    except Exception:
        pass
