import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def _speak_local(text: str) -> None:
    print(f"Jarvis: {text}")
    # On Windows, win32com SAPI.SpVoice is synchronous, rock-solid, and doesn't suffer
    # from pyttsx3's stuck event loop bug on repeated utterances.
    if sys.platform.startswith("win"):
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            # Map TTS_RATE (words per minute, default ~175) to SAPI Rate (-10 to 10, default 0)
            rate = int((config.TTS_RATE - 175) / 15)
            speaker.Rate = max(-10, min(10, rate))
            speaker.Speak(text)
            return
        except Exception:
            # Fall back to pyttsx3 below if SAPI COM initialization fails
            pass

    import pyttsx3

    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", config.TTS_RATE)
        engine.say(text)
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass
        del engine


def speak(text: str) -> None:
    """
    Dispatches to ElevenLabs if configured (config.TTS_PROVIDER ==
    "elevenlabs"), otherwise uses local speech synthesis. Same signature either
    way, so main.py doesn't need to know which one is active.
    """
    if not text:
        return
    text = str(text).strip()
    if not text:
        return

    if config.TTS_PROVIDER == "elevenlabs":
        from audio import tts_elevenlabs

        tts_elevenlabs.speak(text)
    else:
        _speak_local(text)

