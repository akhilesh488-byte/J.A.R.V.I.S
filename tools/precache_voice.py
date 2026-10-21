"""
Pre-cache common static Jarvis phrases using Coqui XTTS v2.
Generates clean, sanitized .wav files in jarvis_voice_cache/
for 0ms latency and zero hallucination.
"""

import os
import re
import sys
import time
import wave
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from jarvis_local import _patch_xtts_runtime, sanitize_xtts_audio

CACHE_DIR = os.path.join(ROOT, "jarvis_voice_cache")
REFERENCE_WAV = os.path.join(ROOT, "jarvis_sample.wav")

PHRASES = {
    "yes_sir": "Yes, sir.",
    "standing_by": "Standing by.",
    "noted_sir": "Noted, sir.",
    "on_its_way_sir": "On its way, sir.",
    "command_unrecognized_sir_standing_by": "Command unrecognized, sir. Standing by.",
    "good_morning_sir": "Good morning, sir.",
    "good_afternoon_sir": "Good afternoon, sir.",
    "good_evening_sir": "Good evening, sir.",
    "good_day_sir": "Good day, sir.",
    "powering_down_systems": "Powering down systems.",
    "powering_down_systems_good_day_sir": "Powering down systems. Good day, sir.",
    "all_primary_systems_are_online_and_functioning_within_nominal_parameters": "All primary systems are online and functioning within nominal parameters.",
    "how_may_i_assist_you_today_sir": "How may I assist you today, sir.",
    "launching_spotify_now_sir": "Launching Spotify now, sir.",
    "launching_chrome_now_sir": "Launching Chrome now, sir.",
    "launching_notepad_now_sir": "Launching Notepad now, sir.",
    "launching_calculator_now_sir": "Launching Calculator now, sir.",
    "launching_youtube_now_sir": "Launching YouTube now, sir.",
    "launching_whatsapp_now_sir": "Launching WhatsApp now, sir.",
    "launching_youtube_and_whatsapp_now_sir": "Launching YouTube and WhatsApp now, sir.",
    "playback_stopped_sir": "Playback stopped, sir.",
    "next_track_engaged_sir": "Next track engaged, sir.",
    "dictation_mode_active_sir_please_speak_your_note_clearly": "Dictation mode active, sir. Please speak your note clearly.",
    "no_dictation_input_was_detected_sir": "No dictation input was detected, sir.",
    "i_have_transcribed_your_dictation_and_opened_it_in_notepad_sir": "I have transcribed your dictation and opened it in Notepad, sir.",
}


def precache_all(force: bool = False):
    import torch
    from TTS.api import TTS

    os.makedirs(CACHE_DIR, exist_ok=True)
    _patch_xtts_runtime()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PRECACHE] Loading Coqui XTTS v2 on {device.upper()}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    total = len(PHRASES)
    for idx, (key, text) in enumerate(PHRASES.items(), 1):
        out_path = os.path.join(CACHE_DIR, f"{key}.wav")
        if os.path.exists(out_path) and not force:
            print(f"[{idx}/{total}] Already cached: '{text}' ({key}.wav)")
            continue

        print(f"[{idx}/{total}] Caching: '{text}' -> {key}.wav")
        t0 = time.time()
        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=REFERENCE_WAV,
                language="en",
                file_path=out_path,
                speed=1.05,
            )
            sanitize_xtts_audio(out_path, max_words=len(text.split()))
            dur = time.time() - t0
            print(f"       Done in {dur:.2f}s")
        except Exception as e:
            print(f"       Error caching '{text}': {e}")

    print("[PRECACHE] Voice cache generation complete!")


if __name__ == "__main__":
    force_run = "--force" in sys.argv
    precache_all(force=force_run)
