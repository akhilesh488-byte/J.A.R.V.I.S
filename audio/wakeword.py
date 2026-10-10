"""
Always-listening wake word detection using openWakeWord (free, local,
no API key). Listens on the default microphone and calls on_wake()
whenever it hears "jarvis" (or whichever pretrained model you pick).

Note: openWakeWord ships pretrained models for a handful of words
(e.g. "hey jarvis" is NOT built in by default — the closest built-in
is generic; for an exact "Jarvis" wake word you'll want to train a
custom model with their free training notebook, which takes ~15 min
on Google Colab and needs no ML expertise). This file works with
either the pretrained "alexa" style models for testing, or your own
custom "jarvis" model once you've trained one — just drop the .onnx
file in and point MODEL_PATH at it.
"""

import numpy as np
import pyaudio

CHUNK = 1280  # openWakeWord expects 80ms chunks at 16kHz
RATE = 16000


def _open_stream(mic):
    return mic.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )


def listen_for_wake_word(on_wake, model_path: str = None, threshold: float = 0.5):
    from openwakeword.model import Model

    wakeword_models = [model_path] if model_path else ["hey_jarvis"]
    oww_model = Model(wakeword_models=wakeword_models, inference_framework="onnx")

    mic = pyaudio.PyAudio()
    stream = _open_stream(mic)

    print("Jarvis is listening for the wake word...")
    try:
        while True:
            try:
                audio_chunk = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
                predictions = oww_model.predict(audio_chunk)
            except OSError as e:
                # Recover from transient Windows audio endpoint changes (e.g. when games/Spotify launch)
                print(f"Audio stream reset ({e}), reconnecting microphone...")
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
                import time
                time.sleep(0.5)
                stream = _open_stream(mic)
                continue
            except MemoryError:
                import gc, time
                gc.collect()
                time.sleep(0.2)
                continue

            for model_name, score in predictions.items():
                if score > threshold:
                    print(f"Wake word detected! ({model_name} confidence: {score:.2f})")
                    # Close wake stream so record_command has exclusive mic access
                    try:
                        stream.stop_stream()
                    except Exception:
                        pass
                    try:
                        stream.close()
                    except Exception:
                        pass
                    try:
                        on_wake()
                    finally:
                        oww_model.reset()
                        import gc
                        gc.collect()
                        stream = _open_stream(mic)
                    break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stream.stop_stream()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        try:
            mic.terminate()
        except Exception:
            pass
