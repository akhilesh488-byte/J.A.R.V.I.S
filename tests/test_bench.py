"""
Unit tests for the J.A.R.V.I.S. Benchmark Suite:
- Scoring logic (tool_ok, args_ok, valid_json, false_tool_call, unsupported, confirmation)
- CSV schema and dataset validation
"""

from __future__ import annotations

import csv
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.dataset import DatasetRow, load_dataset
from bench.env_capture import capture_environment
from bench.mock_tools import BenchmarkMockTools
from bench.run import CSV_COLUMNS, run_benchmark
from bench.scoring import check_unsupported_reply, check_whatsapp_confirmation, score_call


class TestBenchScoring(unittest.TestCase):
    def test_tool_and_args_exact_match(self):
        calls = [{"tool": "open_app", "args": {"app_name": "Spotify"}}]
        score = score_call(
            expected_tool="open_app",
            expected_args={"app_name": "spotify"},
            calls=calls,
            valid_json=True,
        )
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["args_ok"])
        self.assertTrue(score["valid_json"])
        self.assertFalse(score["false_tool_call"])

    def test_tool_args_mismatch(self):
        calls = [{"tool": "open_app", "args": {"app_name": "chrome"}}]
        score = score_call(
            expected_tool="open_app",
            expected_args={"app_name": "spotify"},
            calls=calls,
            valid_json=True,
        )
        self.assertTrue(score["tool_ok"])
        self.assertFalse(score["args_ok"])
        self.assertFalse(score["false_tool_call"])

    def test_extra_model_args_ignored(self):
        calls = [{"tool": "set_timer", "args": {"seconds": 300, "label": "Tea", "extra_param": 1}}]
        score = score_call(
            expected_tool="set_timer",
            expected_args={"seconds": 300},
            calls=calls,
            valid_json=True,
        )
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["args_ok"])

    def test_false_tool_call_when_expected_none(self):
        calls = [{"tool": "open_app", "args": {"app_name": "question"}}]
        score = score_call(
            expected_tool=None,
            expected_args={},
            calls=calls,
            valid_json=True,
            category="adversarial",
        )
        self.assertFalse(score["tool_ok"])
        self.assertFalse(score["args_ok"])
        self.assertTrue(score["false_tool_call"])

    def test_expected_none_no_tool_called(self):
        score = score_call(
            expected_tool=None,
            expected_args={},
            calls=[],
            valid_json=False,
            reply="The capital of France is Paris.",
            category="reasoning",
        )
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["args_ok"])
        self.assertFalse(score["false_tool_call"])

    def test_unsupported_request_claim_success_fails(self):
        score = score_call(
            expected_tool=None,
            expected_args={},
            calls=[],
            valid_json=False,
            reply="I have made the pizza and baked it for you!",
            category="unsupported",
        )
        self.assertFalse(score["tool_ok"])
        self.assertFalse(score["unsupported_ok"])

    def test_unsupported_request_polite_refusal_passes(self):
        score = score_call(
            expected_tool=None,
            expected_args={},
            calls=[],
            valid_json=False,
            reply="I'm afraid I cannot make a pizza, sir. I can only assist with desktop tasks.",
            category="unsupported",
        )
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["unsupported_ok"])

    def test_whatsapp_confirmation_rule_failure(self):
        calls = [{"tool": "send_whatsapp_message", "args": {"recipient": "Bob", "message": "hello"}}]
        # History without confirmation question
        history = [{"role": "user", "content": "send a message"}]
        score = score_call(
            expected_tool="send_whatsapp_message",
            expected_args={"recipient": "Bob", "message": "hello"},
            calls=calls,
            history=history,
        )
        self.assertFalse(score["confirmation_ok"])
        self.assertFalse(score["tool_ok"])

    def test_whatsapp_confirmation_rule_success(self):
        calls = [{"tool": "send_whatsapp_message", "args": {"recipient": "Bob", "message": "hello"}}]
        history = [
            {"role": "user", "content": "send whatsapp to Bob saying hello"},
            {"role": "assistant", "content": "Shall I send the WhatsApp message to Bob?"},
        ]
        score = score_call(
            expected_tool="send_whatsapp_message",
            expected_args={"recipient": "Bob", "message": "hello"},
            calls=calls,
            history=history,
        )
        self.assertTrue(score["confirmation_ok"])
        self.assertTrue(score["tool_ok"])
        self.assertTrue(score["args_ok"])


class TestBenchCsvAndDataset(unittest.TestCase):
    def test_dataset_loader(self):
        path = os.path.join(ROOT, "bench", "data", "sample_dataset.jsonl")
        rows = load_dataset(path)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0].id, "d01")
        self.assertEqual(rows[0].expected_tool, "open_app")
        self.assertIsInstance(rows[0].expected_args, dict)

    def test_mock_tools(self):
        mock = BenchmarkMockTools()
        table = mock.table()
        self.assertIn("open_app", table)
        self.assertIn("add_todo", table)
        res = table["open_app"](app_name="spotify")
        self.assertIn("spotify", res)
        self.assertEqual(len(mock.calls), 1)
        self.assertEqual(mock.calls[0]["tool"], "open_app")
        self.assertEqual(mock.calls[0]["args"], {"app_name": "spotify"})

    def test_csv_schema_and_runner(self):
        dataset_path = os.path.join(ROOT, "bench", "data", "sample_dataset.jsonl")
        out_dir = os.path.join(ROOT, "results", "test_out")
        os.makedirs(out_dir, exist_ok=True)

        csv_path = run_benchmark(
            experiment="unit_test_exp",
            machine="TestBox",
            provider_name="stub",
            out_dir=out_dir,
            dataset_path=dataset_path,
            repeats=1,
            warmup=0,
        )

        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, CSV_COLUMNS)
            rows = list(reader)
            self.assertEqual(len(rows), 10)
            for r in rows:
                self.assertIn(r["tool_ok"], ("True", "False"))
                self.assertIn(r["args_ok"], ("True", "False"))
                self.assertEqual(r["experiment"], "unit_test_exp")
                self.assertEqual(r["machine"], "TestBox")

    def test_environment_capture(self):
        dataset_path = os.path.join(ROOT, "bench", "data", "sample_dataset.jsonl")
        out_json = os.path.join(ROOT, "results", "test_out", "env.json")
        env = capture_environment(dataset_path=dataset_path, output_path=out_json)

        self.assertIn("os", env)
        self.assertIn("python", env)
        self.assertIn("git_commit", env)
        self.assertIn("hashes", env)
        self.assertIsNotNone(env["hashes"]["tool_registry_sha256"])
        self.assertIsNotNone(env["hashes"]["dataset_sha256"])
        self.assertTrue(os.path.exists(out_json))


if __name__ == "__main__":
    unittest.main()
