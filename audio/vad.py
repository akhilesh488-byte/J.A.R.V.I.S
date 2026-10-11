"""
Energy-based end-of-speech detection. Pure numpy, no audio device needed,
so it can be unit-tested with synthetic audio. Replaces the fixed 5-second
recording window: recording stops after the speaker pauses.
"""

import numpy as np


def rms(samples) -> float:
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


class EnergyEndpointer:
    def __init__(self, rate, chunk, rms_threshold, silence_s, min_s, no_speech_s, max_s):
        self.chunk_s = chunk / float(rate)
        self.rms_threshold = rms_threshold
        self.silence_s = silence_s
        self.min_s = min_s
        self.no_speech_s = no_speech_s
        self.max_s = max_s
        self.elapsed = 0.0
        self.silence_run = 0.0
        self.heard_speech = False

    def feed(self, samples):
        """Feed one chunk. Returns (stop, reason); reason is '' while recording."""
        self.elapsed += self.chunk_s
        if rms(samples) > self.rms_threshold:
            self.heard_speech = True
            self.silence_run = 0.0
        elif self.heard_speech:
            self.silence_run += self.chunk_s

        if self.elapsed >= self.max_s:
            return True, "max_length"
        if self.heard_speech and self.silence_run >= self.silence_s and self.elapsed >= self.min_s:
            return True, "end_of_speech"
        if not self.heard_speech and self.elapsed >= self.no_speech_s:
            return True, "no_speech"
        return False, ""
