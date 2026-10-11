"""
Always-listening wake word using openWakeWord (free, local, no key).

openWakeWord has a pretrained "hey_jarvis" model; WAKE_WORD_MODEL can also
be a path to your own .onnx file. Run `python doctor.py` to check the
model loads before you rely on this. Untested on real hardware in the
sandbox this was written in (no microphone there).
"""

import numpy as np

CHUNK = 1280  # 80 ms at 16 kHz, what openWakeWord expects
RATE = 16000


def _load_model(name_or_path):
    import openwakeword
    from openwakeword.model import Model

    try:
        openwakeword.utils.download_models()  # no-op if already downloaded
    except Exception as e:
        print(f"(couldn't check for wake word models: {e})")
    return Model(wakeword_models=[name_or_path], inference_framework="onnx")


def listen_for_wake_word(on_wake, model: str = "hey_jarvis", threshold: float = 0.28):
    import pyaudio

    oww = _load_model(model)
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                     frames_per_buffer=CHUNK)
    print(f"Listening for the wake word ({model})... Ctrl+C to stop.")
    try:
        while True:
            chunk = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            scores = oww.predict(chunk)
            if any(score > threshold for score in scores.values()):
                stream.stop_stream()
                try:
                    on_wake()
                except Exception as e:
                    import traceback
                    print(f"[wakeword] Error during command handling: {e}")
                    traceback.print_exc()
                finally:
                    # Always reset and resume listening, even if something crashed
                    oww.reset()
                    stream.start_stream()
                    print(f"Listening for the wake word ({model})... Ctrl+C to stop.")
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        pa.terminate()
