"""
ElevenLabs text-to-speech with a British voice persona. Only used when
config.ELEVENLABS_API_KEY (and a voice ID) are set; otherwise tts.py
falls back to the local pyttsx3 path.

Requires an output audio player on the system:
- Linux: `sudo apt install mpg123` (or ffplay via ffmpeg)
- Windows: uses the built-in winsound-compatible playback via pydub/ffplay
"""

import os
import subprocess
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_client = None


def _play(audio_path: str) -> None:
    # ffplay first (works on Windows and Linux if ffmpeg is installed),
    # then mpg123, then on Windows hand the file to the default player.
    import shutil

    for cmd in (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path],
                ["mpg123", "-q", audio_path]):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, check=True)
                return
            except Exception:
                pass
    if os.name == "nt":
        os.startfile(audio_path)  # opens the default player; doesn't block
    else:
        print("Couldn't play audio: install ffmpeg (ffplay) or mpg123.")


def speak(text: str) -> None:
    global _client
    if not text:
        return

    if _client is None:
        from elevenlabs.client import ElevenLabs

        _client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)

    print(f"Jarvis: {text}")
    audio = _client.text_to_speech.convert(
        voice_id=config.ELEVENLABS_VOICE_ID,
        model_id="eleven_turbo_v2_5",  # low-latency model
        text=text,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        for chunk in audio:
            f.write(chunk)
        temp_path = f.name

    _play(temp_path)
    os.unlink(temp_path)
