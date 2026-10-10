"""
Jarvis — Phase 1

Loop: wake word -> record command -> transcribe -> route -> speak reply.

Run with:  python main.py
"""

import os

import config
from audio import wakeword, stt, tts
from core.router import handle_command
from models.model_manager import ModelManager

model_manager = ModelManager()


def on_wake():
    tts.speak("Yes?")
    wav_path = None
    try:
        wav_path = stt.record_command()
        text = stt.transcribe(wav_path)
    except Exception as e:
        print(f"Error capturing command: {e}")
        tts.speak("Sorry, I had trouble hearing that.")
        return
    finally:
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

    if not text.strip():
        tts.speak("I didn't catch that.")
        return

    print(f"You said: {text}")
    try:
        reply = handle_command(text, model_manager, tts.speak)
    except Exception as e:
        print(f"Error handling command: {e}")
        reply = f"Sorry, I encountered an issue: {e}"

    if reply:
        tts.speak(reply)


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    print("Starting Jarvis...")
    print("Configured models available:", [
        name for name in config.MODEL_PRIORITY
        if model_manager.get_client(name) and model_manager.get_client(name).is_available()
    ] or "none — add API keys to config.py")

    wakeword.listen_for_wake_word(
        on_wake,
        model_path=None,  # set this once you've trained/downloaded a "jarvis" model
        threshold=config.WAKE_WORD_THRESHOLD,
    )


if __name__ == "__main__":
    main()
