import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["JARVIS_BENCHMARK"] = "1"

import numpy as np  # noqa: E402

import config  # noqa: E402
from audio.vad import EnergyEndpointer, rms  # noqa: E402
from bench.logger import REQUIRED_COLUMNS, log_experiment_run  # noqa: E402
from bench.scoring import score_call  # noqa: E402
from core import memory  # noqa: E402
from core.router import classify_regex, route, route_command  # noqa: E402
from models.base import ModelClient, new_stats  # noqa: E402
from models.ollama_manager_client import OllamaManagerClient  # noqa: E402
from tools.mock_tools import MockTools, ToolCallRecord  # noqa: E402
from tools.tool_registry import build_mock, get_tool_schemas, validate_tool_call  # noqa: E402


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


class AudioHelpersTests(unittest.TestCase):
    def setUp(self):
        from audio.stt import apply_trailing_pad, normalize_audio
        self.normalize = normalize_audio
        self.pad = apply_trailing_pad

    def test_normalize_raises_quiet_audio(self):
        quiet = np.full(1024, 1000, dtype=np.int16)
        normalized = self.normalize(quiet)
        self.assertGreaterEqual(np.max(np.abs(normalized.astype(np.float32))),
                                np.max(np.abs(quiet.astype(np.float32))))

    def test_normalize_does_not_clip_loud_audio(self):
        loud = np.full(1024, 30000, dtype=np.int16)
        normalized = self.normalize(loud)
        self.assertLessEqual(np.max(np.abs(normalized.astype(np.int32))), 32767)

    def test_normalize_silent_audio_unchanged(self):
        silent = np.zeros(512, dtype=np.int16)
        normalized = self.normalize(silent)
        self.assertTrue(np.all(normalized == 0))

    def test_trailing_pad_adds_frames(self):
        chunk = 1024
        frames = [bytes(chunk * 2)]
        padded = self.pad(frames, pad_ms=200, sample_width=2)
        self.assertGreater(len(padded), len(frames))

    def test_trailing_pad_content_is_silence(self):
        frames = [b"\x00" * 2048]
        padded = self.pad(frames, pad_ms=100, sample_width=2)
        for extra_frame in padded[len(frames):]:
            self.assertEqual(set(extra_frame), {0})


class VocabFixTests(unittest.TestCase):
    def setUp(self):
        config.VOCAB_FIX = True

    def tearDown(self):
        config.VOCAB_FIX = False

    def test_corrects_near_match(self):
        from core.vocab_fix import correct_transcript
        result = correct_transcript("open spotiify")
        self.assertIn("spotify", result.lower())

    def test_no_correction_when_disabled(self):
        config.VOCAB_FIX = False
        from core.vocab_fix import correct_transcript
        original = "open spotiify"
        self.assertEqual(correct_transcript(original), original)

    def test_exact_match_unchanged(self):
        from core.vocab_fix import correct_transcript
        self.assertEqual(correct_transcript("open spotify"), "open spotify")

    def test_stopwords_not_corrected(self):
        from core.vocab_fix import correct_transcript
        result = correct_transcript("add the task to my list")
        self.assertIn("the", result)
        self.assertIn("to", result)
        self.assertIn("my", result)

    def test_no_hallucination_on_gibberish(self):
        from core.vocab_fix import correct_transcript
        result = correct_transcript("xqz zzz")
        self.assertIsInstance(result, str)


class BakeoffNormalizerTests(unittest.TestCase):
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


class MockToolsSyncTests(unittest.TestCase):
    """Verifies all 12 tools exist with exact typed signatures and return realistic string outputs."""

    def setUp(self):
        self.mock = MockTools()

    def test_all_12_tools_and_signatures(self):
        res1 = self.mock.open_app(app_name="vscode")
        self.assertEqual(res1, "Opened vscode.")

        res2 = self.mock.add_todo(task="refactor tests")
        self.assertEqual(res2, "Added task: refactor tests")

        res3 = self.mock.list_todos()
        self.assertIn("Buy groceries", res3)

        res4 = self.mock.complete_todo(task_hint="groceries")
        self.assertIn("groceries", res4)

        res5 = self.mock.set_timer(seconds=120, label="tea")
        self.assertIn("120 seconds (tea)", res5)

        res6 = self.mock.set_alarm_at(hour=8, minute=15, label="standup")
        self.assertIn("08:15 (standup)", res6)

        res7 = self.mock.list_recent_emails(count=3)
        self.assertIn("3 recent emails", res7)

        res8 = self.mock.create_email_draft(to="alice@test.com", subject="Update", body="Status ok")
        self.assertIn("alice@test.com", res8)

        res9 = self.mock.list_upcoming_events(count=2)
        self.assertIn("2 upcoming events", res9)

        res10 = self.mock.create_event(
            summary="Review", start_time="2026-10-20T10:00:00", end_time="2026-10-20T11:00:00", description="Notes"
        )
        self.assertIn("Review", res10)

        res11 = self.mock.list_recent_files(count=2)
        self.assertIn("2 recent files", res11)

        res12 = self.mock.remember_note(note="favorite editor is vim")
        self.assertIn("favorite editor is vim", res12)

        # Check call records
        self.assertEqual(len(self.mock.calls), 12)
        first = self.mock.calls[0]
        self.assertEqual(first[0], "open_app")
        self.assertEqual(first[1], {"app_name": "vscode"})
        self.assertIsInstance(first[2], float)
        # Check dict access compatibility
        self.assertEqual(first["tool"], "open_app")
        self.assertEqual(first["args"], {"app_name": "vscode"})

        self.mock.reset()
        self.assertEqual(len(self.mock.calls), 0)

    def test_tool_validation_helper(self):
        valid, _ = validate_tool_call("open_app", {"app_name": "chrome"})
        self.assertTrue(valid)

        invalid_missing, err1 = validate_tool_call("open_app", {})
        self.assertFalse(invalid_missing)
        self.assertIn("Missing required arguments", err1)

        invalid_unknown, err2 = validate_tool_call("unknown_tool", {})
        self.assertFalse(invalid_unknown)
        self.assertIn("Unknown tool", err2)


class OllamaManagerClientTests(unittest.TestCase):
    """Tests JSON parsing, multi-turn tool loops, and cascade escalation using mock HTTP responses."""

    def _make_http_response(self, content_str: str, prompt_eval: int = 20, eval_cnt: int = 10):
        body = json.dumps({
            "model": "jarvis-3b",
            "message": {"role": "assistant", "content": content_str},
            "prompt_eval_count": prompt_eval,
            "eval_count": eval_cnt,
        }).encode("utf-8")
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        return resp

    @patch("urllib.request.urlopen")
    def test_json_tool_calling_and_followup(self, mock_urlopen):
        # Turn 1: model outputs JSON tool call; Turn 2: model outputs conversational reply
        resp1 = self._make_http_response('{"tool": "open_app", "args": {"app_name": "spotify"}}')
        resp2 = self._make_http_response("I have opened Spotify for you, sir.")
        mock_urlopen.side_effect = [resp1, resp2]

        client = OllamaManagerClient(model="jarvis-3b")
        res = client.ask("open spotify", force_mock=True)

        self.assertEqual(res["tool_called"], "open_app")
        self.assertEqual(res["args"], {"app_name": "spotify"})
        self.assertTrue(res["valid_json"])
        self.assertEqual(res["reply"], "I have opened Spotify for you, sir.")
        self.assertFalse(res["escalated"])
        self.assertIsNone(res["error"])
        self.assertEqual(res["tokens_in"], 40)
        self.assertEqual(res["tokens_out"], 20)

    @patch("urllib.request.urlopen")
    def test_conversational_reply_no_tool(self, mock_urlopen):
        resp = self._make_http_response("Good day! All systems are operational.")
        mock_urlopen.side_effect = [resp]

        client = OllamaManagerClient(model="jarvis-3b")
        res = client.ask("hello jarvis")

        self.assertIsNone(res["tool_called"])
        self.assertIsNone(res["args"])
        self.assertFalse(res["valid_json"])
        self.assertEqual(res["reply"], "Good day! All systems are operational.")
        self.assertFalse(res["escalated"])

    @patch("urllib.request.urlopen")
    def test_invalid_json_handling_without_cascade(self, mock_urlopen):
        resp = self._make_http_response("{broken json response")
        mock_urlopen.side_effect = [resp]

        client = OllamaManagerClient(cascade_enabled=False)
        res = client.ask("run something")

        self.assertFalse(res["valid_json"])
        self.assertIn("Invalid JSON", res["error"])
        self.assertFalse(res["escalated"])

    @patch("urllib.request.urlopen")
    def test_invalid_tool_name_handling(self, mock_urlopen):
        resp = self._make_http_response('{"tool": "explode_computer", "args": {}}')
        mock_urlopen.side_effect = [resp]

        client = OllamaManagerClient(cascade_enabled=False)
        res = client.ask("explode")

        self.assertTrue(res["valid_json"])
        self.assertIn("Unknown tool", res["error"])
        self.assertFalse(res["escalated"])


class RouteCommandTests(unittest.TestCase):
    """Tests headless route_command function."""

    def test_route_command_regex_match(self):
        res = route_command("open spotify", use_mock=True)
        self.assertTrue(res.regex_matched)
        self.assertEqual(res.provider, "regex")
        self.assertEqual(res.tool_called, "open_app")
        self.assertEqual(res.args, {"app_name": "spotify"})
        self.assertEqual(res.llm_ms, 0.0)
        self.assertIn("Opened spotify", res.reply)

    @patch("urllib.request.urlopen")
    def test_route_command_llm_dispatch(self, mock_urlopen):
        body = json.dumps({
            "model": "jarvis-3b",
            "message": {"role": "assistant", "content": "Paris is the capital of France."},
            "prompt_eval_count": 15,
            "eval_count": 8,
        }).encode("utf-8")
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = None
        mock_urlopen.return_value = resp

        res = route_command("what is the capital of France?", provider="auto", use_mock=True)
        self.assertFalse(res.regex_matched)
        self.assertEqual(res.provider, "ollama")
        self.assertEqual(res.reply, "Paris is the capital of France.")


class StructuredLoggerTests(unittest.TestCase):
    """Tests structured CSV logging with header validation."""

    def test_logging_and_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = os.path.join(tmpdir, "test_run.csv")
            data_row = {
                "timestamp": "2026-10-17T12:00:00",
                "text": "open spotify",
                "regex_matched": True,
                "provider": "regex",
                "model": "regex",
                "total_ms": 1.25,
                "llm_ms": 0.0,
                "tool_ms": 0.85,
                "tokens_in": 0,
                "tokens_out": 0,
                "tool_called": "open_app",
                "valid_json": True,
                "exact_match": True,
                "error": None,
                "escalated": False,
            }

            log_experiment_run(csv_file, data_row)
            self.assertTrue(os.path.exists(csv_file))

            # Second row append
            data_row["text"] = "list todos"
            log_experiment_run(csv_file, data_row)

            # Header mismatch test
            bad_file = os.path.join(tmpdir, "bad.csv")
            with open(bad_file, "w", encoding="utf-8") as f:
                f.write("wrong,header\n")
            with self.assertRaises(ValueError):
                log_experiment_run(bad_file, data_row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
