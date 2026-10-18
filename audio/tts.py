import os
import queue
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ---------------------------------------------------------------------------
# pyttsx3 runs its own COM/event loop on Windows.  Calling runAndWait() from
# inside a PyAudio callback thread causes it to fight with the audio driver
# and silently crash.  The fix: one dedicated daemon thread owns the engine
# forever; speak() just puts text in a queue and returns immediately.
# ---------------------------------------------------------------------------
_tts_queue: queue.Queue = queue.Queue()
_tts_thread: threading.Thread | None = None


def _tts_worker():
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", config.TTS_RATE)
    while True:
        text = _tts_queue.get()          # blocks until something arrives
        if text is None:                 # sentinel → shut down
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[tts] pyttsx3 error: {e}")
        finally:
            _tts_queue.task_done()


def _ensure_worker():
    global _tts_thread
    if _tts_thread is None or not _tts_thread.is_alive():
        _tts_thread = threading.Thread(target=_tts_worker, daemon=True, name="tts-worker")
        _tts_thread.start()


def _speak_local(text: str) -> None:
    _ensure_worker()
    _tts_queue.put(text)
    # Block until the engine has finished speaking so the caller knows
    # the reply was delivered before we start listening again.
    _tts_queue.join()


def speak(text: str) -> None:
    """
    Dispatches to ElevenLabs if configured (config.TTS_PROVIDER == "elevenlabs"),
    Coqui XTTS v2 if configured (config.TTS_PROVIDER in ("xtts", "coqui")),
    otherwise uses local pyttsx3. Same signature either way, so main.py
    doesn't need to know which one is active.
    """
    if not text:
        return
    if config.TTS_PROVIDER == "elevenlabs":
        from audio import tts_elevenlabs
        tts_elevenlabs.speak(text)
    elif config.TTS_PROVIDER in ("xtts", "coqui", "coqui_xtts"):
        from audio.engines import xtts_engine
        out_wav = xtts_engine.synthesize(text)
        if out_wav and os.path.exists(out_wav):
            if sys.platform == "win32":
                import subprocess
                safe_path = out_wav.replace("'", "''")
                ps_cmd = f"(New-Object System.Media.SoundPlayer '{safe_path}').PlaySync()"
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
    else:
        _speak_local(text)


from audio.engines.xtts_engine import clean_text_for_tts, synthesize, tts_to_file

