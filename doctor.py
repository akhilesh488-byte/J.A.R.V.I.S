"""
Checks what's installed and what's configured, so you know which pieces
of Jarvis will work before you run it.

  python doctor.py          # full check
  python doctor.py --mic    # also measure mic levels and suggest VAD_RMS_THRESHOLD
"""

import argparse
import importlib
import shutil
import sys
import urllib.request

import config

OK, WARN, NO = "[ok]  ", "[warn]", "[no]  "


def has(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def line(flag, msg):
    print(f"{flag} {msg}")


def check():
    print(f"Python {sys.version.split()[0]}\n")

    print("Local pieces (free, offline)")
    for mod, why in [("numpy", "audio maths"), ("faster_whisper", "speech-to-text"),
                     ("pyttsx3", "text-to-speech"), ("pyaudio", "microphone"),
                     ("openwakeword", "wake word"), ("onnxruntime", "wake word runtime"),
                     ("mss", "screenshots"), ("plyer", "notifications")]:
        line(OK if has(mod) else NO, f"{mod:15} {why}")
    if has("ctranslate2"):
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        line(OK if n else WARN, f"CUDA devices seen by faster-whisper: {n} "
             f"(WHISPER_DEVICE={config.WHISPER_DEVICE})")
    line(OK if shutil.which("nvidia-smi") else WARN, "nvidia-smi on PATH")
    if sys.platform != "win32" and not shutil.which("gnome-screenshot") and not has("mss"):
        line(WARN, "no screenshot tool (install gnome-screenshot on Wayland)")

    print("\nBrains (at least one needed for anything beyond simple commands)")
    for name, key in [("claude", config.ANTHROPIC_API_KEY), ("openai", config.OPENAI_API_KEY),
                      ("gemini", config.GEMINI_API_KEY)]:
        line(OK if key else WARN, f"{name}: {'key set' if key else 'no key'}")
    if config.LOCAL_LLM_MODEL:
        try:
            base = config.LOCAL_LLM_URL.replace("/v1", "")
            urllib.request.urlopen(base + "/api/tags", timeout=2)
            line(OK, f"local: Ollama reachable, model {config.LOCAL_LLM_MODEL}")
        except Exception:
            line(NO, f"local: LOCAL_LLM_MODEL={config.LOCAL_LLM_MODEL} but Ollama isn't reachable")
    else:
        line(WARN, "local: LOCAL_LLM_MODEL not set (optional)")

    print("\nOptional cloud voice / Google")
    line(OK if config.DEEPGRAM_API_KEY else WARN, "Deepgram STT" + ("" if config.DEEPGRAM_API_KEY else " (using local Whisper)"))
    line(OK if config.ELEVENLABS_API_KEY else WARN, "ElevenLabs TTS" + ("" if config.ELEVENLABS_API_KEY else " (using local voice)"))
    import os
    line(OK if os.path.exists(config.GOOGLE_CREDENTIALS_FILE) else WARN, "credentials.json for Google tools")

    print("\nWake word")
    if has("openwakeword"):
        try:
            import openwakeword
            from openwakeword.model import Model

            openwakeword.utils.download_models()
            Model(wakeword_models=[config.WAKE_WORD_MODEL], inference_framework="onnx")
            line(OK, f"wake word model '{config.WAKE_WORD_MODEL}' loads")
        except Exception as e:
            line(NO, f"wake word model '{config.WAKE_WORD_MODEL}' failed to load: {e}")
    else:
        line(NO, "openwakeword not installed (use --mode text or --mode ptt)")


def mic_test():
    import numpy as np
    import pyaudio
    from audio.vad import rms

    pa = pyaudio.PyAudio()

    # --- List all input devices ---
    print("\nInput devices:")
    default_idx = pa.get_default_input_device_info().get("index", -1)
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) < 1:
            continue
        marker = " <-- default" if i == default_idx else ""
        print(
            f"  [{i}] {info['name']!r:40s}  "
            f"rate={int(info['defaultSampleRate'])} Hz{marker}"
        )
    if config.MIC_INDEX is not None:
        print(f"\nUsing JARVIS_MIC_INDEX={config.MIC_INDEX}")
    else:
        print(f"\nUsing default device (index {default_idx}). "
              "Set JARVIS_MIC_INDEX=<index> in .env to override.")

    # --- Level measurement ---
    open_kwargs = dict(format=pyaudio.paInt16, channels=1, rate=16000,
                       input=True, frames_per_buffer=1024)
    if config.MIC_INDEX is not None:
        open_kwargs["input_device_index"] = config.MIC_INDEX

    stream = pa.open(**open_kwargs)

    def measure(seconds):
        n = int(16000 / 1024 * seconds)
        frames = []
        rms_vals = []
        peak = 0
        for _ in range(n):
            raw = np.frombuffer(stream.read(1024, exception_on_overflow=False), dtype=np.int16)
            frames.append(raw)
            rms_vals.append(rms(raw))
            peak = max(peak, int(np.max(np.abs(raw))))
        all_samples = np.concatenate(frames)
        return float(np.percentile(rms_vals, 90)), peak, all_samples

    input("\nStay quiet, then press Enter...")
    quiet_rms, quiet_peak, _ = measure(2)

    input("Press Enter, then speak normally for 3 seconds...")
    loud_rms, loud_peak, speech_samples = measure(3)

    stream.close()
    pa.terminate()

    suggested = int((quiet_rms + loud_rms) / 2)
    clipping = loud_peak >= 32000
    print(f"\n  Quiet  — RMS: {quiet_rms:.0f}   Peak: {quiet_peak}")
    print(f"  Speech — RMS: {loud_rms:.0f}   Peak: {loud_peak}")
    if clipping:
        print("  [warn] CLIPPING detected! Lower your microphone gain.")
    else:
        print("  [ok]   No clipping detected.")
    print(f"\nSuggested VAD_RMS_THRESHOLD = {suggested}   (current: {config.VAD_RMS_THRESHOLD:.0f})")
    print("Add this to your .env:  VAD_RMS_THRESHOLD=" + str(suggested))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", action="store_true")
    a = ap.parse_args()
    check()
    if a.mic:
        print()
        mic_test()
