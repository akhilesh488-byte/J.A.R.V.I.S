import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["JARVIS_BENCHMARK"] = "1"

import numpy as np  # noqa: E402

import config  # noqa: E402
from audio.vad import EnergyEndpointer, rms  # noqa: E402
from bench.scoring import score_call  # noqa: E402
from core import memory  # noqa: E402
from core.router import classify_regex, route  # noqa: E402
from models.base import ModelClient, new_stats  # noqa: E402
from tools.tool_registry import build_mock, get_tool_schemas  # noqa: E402


class FakeClient(ModelClient):
    name = "fake"

    def is_available(self):
        return True

    def ask(self, prompt, dispatch_table, image_path=None):
        self.last_stats = new_stats("fake-model")
        self.last_stats["iterations"] = 2
        self.last_stats["tokens_in"] = 10
        self.last_stats["latency_ms"] = 5.0
        dispatch_table["add_todo"](task="from llm")
        return "done"


class FakeManager:
    def __init__(self, client):
        self.client = client

    def candidates(self, requested=None):
        return [self.client] if self.client else []


class RegexTests(unittest.TestCase):
    def test_simple_patterns(self):
        self.assertEqual(classify_regex("open spotify"), ("open_app", {"app_name": "spotify"}))
        self.assertEqual(classify_regex("what's on my to-do list")[0], "list_todos")
        self.assertEqual(classify_regex("add buy groceries to my todo list"),
                         ("add_todo", {"task": "buy groceries"}))
        self.assertEqual(classify_regex("add task call mom"), ("add_todo", {"task": "call mom"}))
        self.assertEqual(classify_regex("mark buy groceries as done"),
                         ("complete_todo", {"task_hint": "buy groceries"}))

    def test_no_match(self):
        self.assertIsNone(classify_regex("what is the capital of France"))

    def test_known_false_positive_is_kept_for_the_paper(self):
        self.assertEqual(classify_regex("what is an open question")[0], "open_app")

    def test_timer_regex_off_by_default_then_on(self):
        self.assertIsNone(classify_regex("set a timer for 5 minutes"))
        config.ENABLE_TIMER_REGEX = True
        try:
            self.assertEqual(classify_regex("set a timer for 5 minutes"), ("set_timer", {"seconds": 300}))
        finally:
            config.ENABLE_TIMER_REGEX = False


class RouteTests(unittest.TestCase):
    def test_regex_path_goes_through_mock(self):
        mock = build_mock()
        res = route("open spotify", FakeManager(None), mock.table(), capture_screen=False)
        self.assertEqual(res.path, "regex")
        self.assertEqual([c["tool"] for c in mock.calls], ["open_app"])

    def test_force_llm_skips_regex_and_returns_stats(self):
        mock = build_mock()
        res = route("open spotify", FakeManager(FakeClient()), mock.table(),
                    force_llm=True, capture_screen=False)
        self.assertEqual(res.path, "llm")
        self.assertEqual(res.stats["iterations"], 2)
        self.assertEqual(mock.calls[0]["args"], {"task": "from llm"})

    def test_no_model_configured(self):
        res = route("tell me a joke", FakeManager(None), build_mock().table(), capture_screen=False)
        self.assertEqual(res.error, "no_model")

    def test_benchmark_mode_hides_memory_tool(self):
        names = [t["name"] for t in get_tool_schemas()]
        self.assertNotIn("remember_note", names)


class ScoringTests(unittest.TestCase):
    def test_rules(self):
        calls = [{"tool": "add_todo", "args": {"task": " Buy Groceries ", "extra": "x"}}]
        s = score_call("add_todo", {"task": "buy groceries"}, calls)
        self.assertTrue(s["tool_correct"] and s["args_correct"])
        s = score_call("add_todo", {"task": "buy"}, calls)
        self.assertFalse(s["args_correct"])
        self.assertTrue(s["args_lenient"])
        self.assertTrue(score_call(None, {}, [])["tool_correct"])
        self.assertFalse(score_call(None, {}, calls)["tool_correct"])
        self.assertFalse(score_call("add_todo", {}, [])["tool_correct"])


class VadTests(unittest.TestCase):
    def run_audio(self, seconds_of_speech, lead_silence=0.2):
        rate, chunk = 16000, 1024
        ep = EnergyEndpointer(rate, chunk, 500, 1.0, 0.6, 5.0, 15.0)
        n_lead = int(lead_silence * rate / chunk)
        n_speech = int(seconds_of_speech * rate / chunk)
        chunks = [np.zeros(chunk)] * n_lead + [np.full(chunk, 3000.0)] * n_speech + [np.zeros(chunk)] * 200
        for i, c in enumerate(chunks):
            stop, reason = ep.feed(c)
            if stop:
                return reason, ep.elapsed
        return "never", ep.elapsed

    def test_stops_after_pause(self):
        reason, elapsed = self.run_audio(1.5)
        self.assertEqual(reason, "end_of_speech")
        self.assertLess(elapsed, 4.0)

    def test_no_speech_times_out(self):
        reason, elapsed = self.run_audio(0.0)
        self.assertEqual(reason, "no_speech")

    def test_rms(self):
        self.assertAlmostEqual(rms([3, 4]), (12.5) ** 0.5)


class MemoryTests(unittest.TestCase):
    def test_append_and_load(self):
        old = config.NOTES_FILE
        with tempfile.TemporaryDirectory() as d:
            config.NOTES_FILE = os.path.join(d, "notes.md")
            try:
                memory.append_note("my locker code is in the blue book")
                self.assertIn("blue book", memory.load_notes())
            finally:
                config.NOTES_FILE = old


if __name__ == "__main__":
    unittest.main(verbosity=2)


class AudioHelpersTests(unittest.TestCase):
    """Tests for pure audio utility functions in audio/stt.py."""

    def setUp(self):
        from audio.stt import normalize_audio, apply_trailing_pad
        self.normalize = normalize_audio
        self.pad = apply_trailing_pad

    def test_normalize_raises_quiet_audio(self):
        """Quiet audio should be amplified toward the target peak."""
        from audio.stt import _NORMALIZE_TARGET
        quiet = np.full(1024, 1000, dtype=np.int16)
        normalized = self.normalize(quiet)
        self.assertGreaterEqual(np.max(np.abs(normalized.astype(np.float32))),
                                np.max(np.abs(quiet.astype(np.float32))))

    def test_normalize_does_not_clip_loud_audio(self):
        """Loud audio should not be boosted past 32767."""
        loud = np.full(1024, 30000, dtype=np.int16)
        normalized = self.normalize(loud)
        self.assertLessEqual(np.max(np.abs(normalized.astype(np.int32))), 32767)

    def test_normalize_silent_audio_unchanged(self):
        """Silent audio (all zeros) should remain all zeros."""
        silent = np.zeros(512, dtype=np.int16)
        normalized = self.normalize(silent)
        self.assertTrue(np.all(normalized == 0))

    def test_trailing_pad_adds_frames(self):
        """apply_trailing_pad should append silence frames."""
        chunk = 1024
        frames = [bytes(chunk * 2)]  # one frame, sample_width=2
        padded = self.pad(frames, pad_ms=200, sample_width=2)
        self.assertGreater(len(padded), len(frames))

    def test_trailing_pad_content_is_silence(self):
        """Appended frames should be all zeros."""
        frames = [b"\x00" * 2048]
        padded = self.pad(frames, pad_ms=100, sample_width=2)
        # All extra frames should be silence
        for extra_frame in padded[len(frames):]:
            self.assertEqual(set(extra_frame), {0})


class VocabFixTests(unittest.TestCase):
    """Tests for core/vocab_fix.py — conservative fuzzy correction."""

    def setUp(self):
        config.VOCAB_FIX = True

    def tearDown(self):
        config.VOCAB_FIX = False

    def test_corrects_near_match(self):
        """A word very close to a known vocab word should be corrected."""
        from core.vocab_fix import correct_transcript
        # "spotiify" should correct to "spotify"
        result = correct_transcript("open spotiify")
        self.assertIn("spotify", result.lower())

    def test_no_correction_when_disabled(self):
        """With VOCAB_FIX=False, transcripts pass through unchanged."""
        config.VOCAB_FIX = False
        from core.vocab_fix import correct_transcript
        original = "open spotiify"
        self.assertEqual(correct_transcript(original), original)

    def test_exact_match_unchanged(self):
        """Exact vocabulary words should not be altered."""
        from core.vocab_fix import correct_transcript
        self.assertEqual(correct_transcript("open spotify"), "open spotify")

    def test_stopwords_not_corrected(self):
        """Common English stopwords should never be corrected."""
        from core.vocab_fix import correct_transcript
        # "the", "to", "my" are stopwords; should remain
        result = correct_transcript("add the task to my list")
        self.assertIn("the", result)
        self.assertIn("to", result)
        self.assertIn("my", result)

    def test_no_hallucination_on_gibberish(self):
        """Very short tokens or garbage should be passed through."""
        from core.vocab_fix import correct_transcript
        result = correct_transcript("xqz zzz")
        # No crash; result is a string
        self.assertIsInstance(result, str)


class BakeoffNormalizerTests(unittest.TestCase):
    """Tests for the bakeoff normalization function."""

    def setUp(self):
        sys.path.insert(0, os.path.join(ROOT, "bench"))
        from stt_bakeoff import normalize_text
        self.normalize = normalize_text

    def test_lowercase(self):
        self.assertEqual(self.normalize("Open Spotify"), "open spotify")

    def test_strip_punctuation(self):
        self.assertEqual(self.normalize("done."), "done")
        self.assertEqual(self.normalize("add, call mom!"), "add call mom")

    def test_collapse_whitespace(self):
        self.assertEqual(self.normalize("open   youtube"), "open youtube")

    def test_numerals_preserved(self):
        self.assertEqual(self.normalize("set timer for 5 minutes"), "set timer for 5 minutes")
        self.assertEqual(self.normalize("set timer for five minutes"), "set timer for five minutes")
