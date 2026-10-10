"""
Records a short voice command after the wake word fires, then
transcribes it locally with faster-whisper (no API key, no cost,
runs on CPU on your hardware fine for short clips).
"""

import os
import sys
import wave

import pyaudio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import tempfile

RATE = 16000
CHUNK = 1024
CHANNELS = 1
RECORD_SECONDS = 5  # length of the command window after wake word


def record_command(save_path: str = None) -> str:
    if save_path is None:
        save_path = os.path.join(tempfile.gettempdir(), "jarvis_command.wav")
    audio = pyaudio.PyAudio()
    stream = None
    frames = []
    try:
        sample_width = audio.get_sample_size(pyaudio.paInt16)
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        print("Listening for your command...")
        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            frames.append(stream.read(CHUNK, exception_on_overflow=False))
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        try:
            audio.terminate()
        except Exception:
            pass

    with wave.open(save_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return save_path


_model = None


def _transcribe_local(wav_path: str) -> str:
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type="int8",
            cpu_threads=2,
        )

    segments, _ = _model.transcribe(wav_path)
    result = " ".join(segment.text for segment in segments).strip()
    import gc
    gc.collect()
    return result


def transcribe(wav_path: str) -> str:
    """
    Dispatches to Deepgram if configured (config.STT_PROVIDER == "deepgram"),
    otherwise uses the local faster-whisper model. Same signature either way,
    so main.py doesn't need to know which one is active.
    """
    if config.STT_PROVIDER == "deepgram":
        from audio import stt_deepgram

        return stt_deepgram.transcribe(wav_path)
    return _transcribe_local(wav_path)
