"""
Jarvis

  python main.py                 # mode from JARVIS_MODE (default: wake)
  python main.py --mode text     # type commands, no microphone needed
  python main.py --mode ptt      # press Enter, speak, it stops when you pause
  python main.py --mode wake     # always listening for the wake word

Start with text mode. Get that working, then add voice.
"""

import argparse
import os
import time

import config
from core.router import handle_command
from models.model_manager import ModelManager

model_manager = ModelManager()


def speak(text):
    from audio import tts

    tts.speak(text)


def print_dialogue(user_text=None, jarvis_text=None):
    border = "=" * 70
    if user_text:
        print(f"\n{border}", flush=True)
        print(f"  YOU    : \"{user_text}\"", flush=True)
    if jarvis_text:
        if not user_text:
            print(f"\n{border}", flush=True)
        print(f"  JARVIS : \"{jarvis_text}\"", flush=True)
        print(f"{border}\n", flush=True)


def say_and_print(text, voice=True):
    print_dialogue(jarvis_text=text)
    if voice:
        speak(text)


def run_text(voice):
    print("Text mode. Type a command (or 'quit').")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("quit", "exit"):
            break
        if text:
            print_dialogue(user_text=text)
            say_and_print(handle_command(text, model_manager, speak), voice)


def handle_spoken(voice=True):
    from audio import stt

    # Small delay so the microphone doesn't capture speaker output (echo).
    if config.TTS_POST_SPEAK_DELAY > 0:
        time.sleep(config.TTS_POST_SPEAK_DELAY)

    wav = stt.record_command()
    text = stt.transcribe(wav)
    if not text.strip():
        say_and_print("I didn't catch that, sir.", voice)
        return
    print_dialogue(user_text=text)
    say_and_print(handle_command(text, model_manager, speak), voice)


def run_ptt():
    print("Push to talk. Press Enter, then speak. Ctrl+C to quit.")
    try:
        while True:
            input("[Enter to talk] ")
            handle_spoken()
    except (EOFError, KeyboardInterrupt):
        print()


def run_wake():
    from audio import wakeword

    def on_wake():
        speak("Yes?")
        handle_spoken()

    wakeword.listen_for_wake_word(on_wake, model=config.WAKE_WORD_MODEL,
                                  threshold=config.WAKE_WORD_THRESHOLD)


def print_status():
    brains = model_manager.available()
    print("Jarvis starting.")
    print(f"  Brain : {', '.join(brains) if brains else 'none configured (see .env.example)'}")
    ears = ("Deepgram (cloud)" if config.STT_PROVIDER == "deepgram" else
            f"faster-whisper {config.WHISPER_MODEL_SIZE} on {config.WHISPER_DEVICE} (local)")
    mouth = "ElevenLabs (cloud)" if config.TTS_PROVIDER == "elevenlabs" else "pyttsx3 (local)"
    print(f"  Ears  : {ears}")
    print(f"  Voice : {mouth}")
    print(f"  Memory: {'on' if config.USE_MEMORY else 'off'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wake", "ptt", "text", "local"], default="local")
    ap.add_argument("--no-voice", action="store_true", help="text mode: don't speak replies")
    args = ap.parse_args()

    if args.mode == "local":
        import jarvis_local
        jarvis_local.main()
        return

    os.makedirs(config.DATA_DIR, exist_ok=True)
    print_status()
    try:
        if args.mode == "text":
            run_text(voice=not args.no_voice)
        elif args.mode == "ptt":
            run_ptt()
        else:
            run_wake()
    except (KeyboardInterrupt, SystemExit):
        say_and_print("Powering down systems. Good day, sir.", voice=not args.no_voice)


if __name__ == "__main__":
    main()
